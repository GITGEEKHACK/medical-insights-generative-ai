import os
import re
import json
import time
import traceback
from fuzzywuzzy import fuzz

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.bedrock import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app import logger
from app.constant import BotoClient
from app.constant import MedicalInsights
from app.service.medical_document_insights.nlp_extractor import bedrock_client, get_llm_input_tokens


class HistoryExtractor:
    def __init__(self):
        os.environ['AWS_DEFAULT_REGION'] = BotoClient.AWS_DEFAULT_REGION
        self.bedrock_client = bedrock_client
        self.model_id_llm = 'anthropic.claude-3-sonnet-20240229-v1:0'
        self.model_embeddings = 'amazon.titan-embed-text-v1'
        self.anthropic_llm = BedrockChat(
            model_id=self.model_id_llm,
            model_kwargs={
                "max_tokens": 4000,
                "temperature": 0,
                "top_p": 0,
                "top_k": 0,
                "stop_sequences": [],
            },
            client=self.bedrock_client,
        )
        self.titan_llm = Bedrock(model_id=self.model_embeddings, client=self.bedrock_client)
        self.bedrock_embeddings = BedrockEmbeddings(model_id=self.model_embeddings, client=self.bedrock_client)

    async def __data_formatter(self, json_data):
        """ This method is used to format the data and prepare chunks """

        raw_text = ""
        list_of_page_contents = []
        page_indexes_dict = []
        page_start_index = 0
        for page, content in json_data.items():
            raw_text += content
            list_of_page_contents.append(Document(page_content=content, metadata={'page': int(page.split('_')[1])}))
            page_indexes_dict.append([page, [int(page_start_index), int(page_start_index) + len(content) - 1]])
            page_start_index += len(content)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=20000, chunk_overlap=200
        )

        texts = text_splitter.split_text(raw_text)

        for text in texts:
            threshold = self.anthropic_llm.get_num_tokens(text)
            if threshold > 5000:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=10000, chunk_overlap=200
                )
                texts = text_splitter.split_text(raw_text)
                break

        docs = []
        chunk_start_index = 0
        last_found = 0
        overlap = 0
        max_overlap = 200
        previous_text = ''
        for text in texts:
            current_text = text
            if len(previous_text) != 0:
                min_overlap = min(len(previous_text), len(current_text))
                possible_overlap = min(max_overlap, min_overlap)
                for i in range(1, possible_overlap + 1):
                    if previous_text[-i:] == current_text[:i]:
                        overlap = i
            chunk_start_index = chunk_start_index + len(previous_text) - overlap
            chunk_indexes = [int(chunk_start_index), int(chunk_start_index) + len(current_text) - 1]
            previous_text = current_text

            start_page, end_page, last_found = await self.__find_page_numbers(page_indexes_dict, chunk_indexes,
                                                                              last_found)
            # Create multiple documents
            docs.append(Document(page_content=text,
                                 metadata={'start_page': start_page, 'end_page': end_page}))
        return docs, list_of_page_contents

    async def __find_page_numbers(self, page_indexes_dict, chunk_indexes, last_found):
        start_index, end_index = chunk_indexes
        start_page, end_page = None, None
        for page, indexes in page_indexes_dict[last_found:]:
            if indexes[0] <= start_index <= indexes[1]:
                start_page = int(page.split('_')[1])
            if indexes[0] <= end_index <= indexes[1]:
                end_page = int(page.split('_')[1])

                curr_last_found = end_page - 1
                for x in page_indexes_dict[curr_last_found::-1]:
                    diff = end_index - x[1][0]
                    last_found = int(x[0].split('_')[1]) - 1
                    if diff >= 200:
                        break

            if start_page and end_page:
                break
        return start_page, end_page, last_found

    async def __post_processing(self, text):
        """ This method is used to post-process the LLM response """

        start_index = text.find('{')
        end_index = text.rfind('}') + 1
        json_str = text[start_index:end_index]
        json_str = re.sub(r',\s*\n\s*]', '\n  ]', json_str)
        return json.loads(json_str)

    async def __get_page_number(self, date_with_event, list_of_page_contents, relevant_chunks):
        if len(relevant_chunks) == 1:
            most_similar_chunk_index = 0
        else:
            cosine_similarities = []
            for chunk in relevant_chunks:
                cosine_similarities.append(fuzz.token_set_ratio(date_with_event, chunk.page_content))

            most_similar_chunk_index = cosine_similarities.index(max(cosine_similarities))

        most_similar_chunk = relevant_chunks[most_similar_chunk_index]
        start_page = most_similar_chunk.metadata['start_page']
        end_page = most_similar_chunk.metadata['end_page']
        relevant_pages = list_of_page_contents[start_page - 1: end_page]

        cosine_similarities = []
        for page in relevant_pages:
            cosine_similarities.append(fuzz.token_set_ratio(date_with_event, page.page_content))

        most_similar_page_index = cosine_similarities.index(max(cosine_similarities))
        most_similar_page = [page.metadata['page'] for page in relevant_pages][most_similar_page_index]

        return most_similar_page

    async def get_social_and_family_history(self, vectorstore_faiss, list_of_page_contents):
        llm_start_time = time.time()
        query = MedicalInsights.Prompts.HISTORY_PROMPT
        prompt_template = MedicalInsights.Prompts.PROMPT_TEMPLATE

        logger.info(f'[Medical-Insights][History][{self.model_embeddings}] Embedding tokens for LLM call: '
                    f'{self.titan_llm.get_num_tokens(query) + self.titan_llm.get_num_tokens(prompt_template)}')

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        qa = RetrievalQA.from_chain_type(
            llm=self.anthropic_llm,
            chain_type="stuff",
            retriever=vectorstore_faiss.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        answer = qa.invoke({"query": query})

        response = answer['result']
        relevant_chunks = answer['source_documents']

        output = {'social_history': {'page_no': None, 'values': {'Smoking': 'No', 'Alcohol': 'No', 'Tobacco': 'No'}},
                  'family_history': {'page_no': None, 'values': {}}}
        try:
            processed_response = await self.__post_processing(response)
            output['social_history']['values'] = processed_response.get('Social_History', {})
            output['family_history']['values'] = processed_response.get('Family_History', {})
        except Exception as e:
            logger.warning(f'{e} -> {traceback.format_exc()}')

        if output['family_history']['values']:
            page_family = await self.__get_page_number(json.dumps(output['family_history']['values']),
                                                       list_of_page_contents, relevant_chunks)
            output['family_history']['page_no'] = page_family
        page_social = await self.__get_page_number(json.dumps(output['social_history']['values']),
                                                   list_of_page_contents, relevant_chunks)
        output['social_history']['page_no'] = page_social

        input_tokens = get_llm_input_tokens(self.anthropic_llm, answer) + self.anthropic_llm.get_num_tokens(
            prompt_template)
        output_tokens = self.anthropic_llm.get_num_tokens(response)

        logger.info(f'[Medical-Insights][History][{self.model_id_llm}] Input tokens: {input_tokens} '
                    f'Output tokens: {output_tokens} LLM execution time: {time.time() - llm_start_time}')

        for parent_key in list(output):
            for key in list(output[parent_key]['values']):
                output[parent_key]['values'][key.lower()] = output[parent_key]['values'].pop(key)

        return output

    async def get_psychiatric_injury(self, vectorstore_faiss, list_of_page_contents):
        llm_start_time = time.time()
        query = MedicalInsights.Prompts.PSYCHIATRIC_INJURY_PROMPT
        prompt_template = MedicalInsights.Prompts.PROMPT_TEMPLATE

        logger.info(f'[Medical-Insights][Injury][{self.model_embeddings}] Embedding tokens for LLM call: '
                    f'{self.titan_llm.get_num_tokens(query) + self.titan_llm.get_num_tokens(prompt_template)}')

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        qa = RetrievalQA.from_chain_type(
            llm=self.anthropic_llm,
            chain_type="stuff",
            retriever=vectorstore_faiss.as_retriever(
                search_type="similarity", search_kwargs={"k": 8}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        answer = qa.invoke({"query": query})
        response = answer['result']
        relevant_chunks = answer['source_documents']

        output = {'psychiatric_injury': {'page_no': None, 'values': []}}
        try:
            processed_response = await self.__post_processing(response)
            output['psychiatric_injury']['values'] = processed_response.get('Psychiatric_Injury', [])
        except Exception as e:
            logger.warning(f'{e} -> {traceback.format_exc()}')

        if output['psychiatric_injury']['values']:
            page_injury = await self.__get_page_number(json.dumps(output['psychiatric_injury']['values']),
                                                       list_of_page_contents, relevant_chunks)
            output['psychiatric_injury']['page_no'] = page_injury

        input_tokens = get_llm_input_tokens(self.anthropic_llm, answer) + self.anthropic_llm.get_num_tokens(
            prompt_template)
        output_tokens = self.anthropic_llm.get_num_tokens(response)

        logger.info(f'[Medical-Insights][Injury][{self.model_id_llm}] Input tokens: {input_tokens} '
                    f'Output tokens: {output_tokens} LLM execution time: {time.time() - llm_start_time}')
        return output

    async def get_history(self, document):
        """ This method is used to generate the encounters """

        data = document['page_wise_text']
        pdf_text = "".join(data.values()).strip()

        if not pdf_text:
            template_data = MedicalInsights.TemplateResponse.HISTORY_TEMPLATE_RESPONSE
            return {"general_history": template_data, "document_name": document['name']}
        else:
            output = {}
            t1 = time.time()
            docs, list_of_page_contents = await self.__data_formatter(data)

            emb_tokens = 0
            for i in docs:
                emb_tokens += self.titan_llm.get_num_tokens(i.page_content)

            t2 = time.time()
            logger.info(f'[Medical-Insights][History] Chunk Preparation Time: {t2 - t1}')

            vectorstore_faiss = FAISS.from_documents(
                documents=docs,
                embedding=self.bedrock_embeddings,
            )
            logger.info(f'[Medical-Insights][History][{self.model_embeddings}] Input Embedding tokens: {emb_tokens} '
                        f'and Generation time: {time.time() - t2}')

            history_output = await self.get_social_and_family_history(vectorstore_faiss, list_of_page_contents)
            output.update(history_output)

            injury_output = await self.get_psychiatric_injury(vectorstore_faiss, list_of_page_contents)
            output.update(injury_output)

            return {"general_history": output, "document_name": document['name']}
