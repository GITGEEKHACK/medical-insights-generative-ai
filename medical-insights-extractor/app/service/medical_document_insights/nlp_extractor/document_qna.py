import os
import time
import glob
import json
import copy

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app import logger
from app.constant import BotoClient
from app.constant import MedicalInsights
from app.common.utils import update_file_path
from app.service.medical_document_insights.nlp_extractor import bedrock_client, get_llm_input_tokens


class DocumentQnA:
    def __init__(self):
        os.environ['AWS_DEFAULT_REGION'] = BotoClient.AWS_DEFAULT_REGION
        self.bedrock_client = bedrock_client
        self.model_id_llm = 'anthropic.claude-3-haiku-20240307-v1:0'
        self.model_embeddings = 'amazon.titan-embed-text-v1'

        self.anthropic_llm = BedrockChat(
            model_id=self.model_id_llm,
            model_kwargs={
                "max_tokens": 4000,
                "temperature": 0.75,
                "top_p": 0.01,
                "top_k": 0,
                "stop_sequences": [],
            },
            client=self.bedrock_client,
        )

        self.titan_llm = BedrockChat(model_id=self.model_embeddings, client=self.bedrock_client)
        self.bedrock_embeddings = BedrockEmbeddings(model_id=self.model_embeddings, client=self.bedrock_client)
        self.prompt = self.__create_prompt_template()

    def __create_prompt_template(self):
        prompt_template = """
        Human: You are a Medical Assistant that provides concise answers to the questions related to the medical text context given to you. Strictly answer the questions related to the following information 
        <context>
        {context}
        </context>
        to answer in a helpful manner. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Question: {question}

        Medical Assistant:"""

        self.prompt_template_tokens = self.anthropic_llm.get_num_tokens(prompt_template)

        prompt = PromptTemplate(
            input_variables=["context", "question"], template=prompt_template
        )

        return prompt

    async def __prepare_data(self, project_path):
        project_response_path = project_path.replace(MedicalInsights.REQUEST_FOLDER_NAME, MedicalInsights.RESPONSE_FOLDER_NAME)
        project_response_file_path = os.path.join(project_response_path, 'embeddings.pkl')
        if os.path.exists(project_response_file_path):
            vectored_data = FAISS.load_local(project_response_path, self.bedrock_embeddings, index_name='embeddings',
                                             allow_dangerous_deserialization=True)
        else:
            raw_text = ""
            document_list = glob.glob(os.path.join(project_path, '*'))
            for document in document_list:
                pdf_name, output_dir = await update_file_path(document)
                dir_name = os.path.join(output_dir, 'textract_response')
                with open(f'{dir_name}/{pdf_name}_text.json', 'r') as file:
                    data = json.loads(file.read())
                    raw_text = raw_text + "".join(data.values())
            if len(raw_text.strip()) != 0:
                docs = await self.__data_formatter(raw_text)
                vectored_data = await self.__prepare_embeddings(docs, project_response_path)
            else:
                vectored_data = None
        return vectored_data

    async def __data_formatter(self, raw_text):
        """ This method is used to format the data and prepare chunks """

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=15000, chunk_overlap=200
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

        docs = [Document(page_content=t) for t in texts]
        return docs

    async def __prepare_embeddings(self, docs, project_response_path):
        x = time.time()
        emb_tokens = 0
        for i in docs:
            emb_tokens += self.titan_llm.get_num_tokens(i.page_content)

        y = time.time()
        logger.info(f'[Medical-Insights][QnA-Embeddings] Chunk Preparation Time: {y - x}')

        vector_embeddings = FAISS.from_documents(
            documents=docs,
            embedding=self.bedrock_embeddings,
        )
        vector_embeddings.save_local(project_response_path, index_name='embeddings')
        logger.info(
            f'[Medical-Insights][QnA-Embeddings][{self.model_embeddings}] Input embedding tokens: {emb_tokens}'
            f'and Generation time: {time.time() - y}')
        return vector_embeddings

    async def __create_conversation_chain(self, vectored_data, prompt_template):

        qa = RetrievalQA.from_chain_type(
            llm=self.anthropic_llm,
            chain_type="stuff",
            retriever=vectored_data.as_retriever(
                search_type="similarity", search_kwargs={"k": 6}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template},
        )

        return qa

    async def get_query_response(self, query, project_path):

        x = time.time()
        vectored_data = await self.__prepare_data(project_path)
        logger.info(f"[Medical-Insights-QnA] Input data preparation for LLM is completed in {time.time() - x} seconds.")

        if vectored_data is None:
            logger.warning("[Medical-Insights-QnA] Empty Document Found for QnA !!")
            response = copy.deepcopy(MedicalInsights.TemplateResponse.QNA_EMPTY_DOC_RESPONSE)
            response['query'] = query
            return response

        else:
            x = time.time()
            conversation_chain = await self.__create_conversation_chain(vectored_data, self.prompt)
            answer = conversation_chain({'query': query})

            input_tokens = get_llm_input_tokens(self.anthropic_llm, answer) + self.prompt_template_tokens
            output_tokens = self.anthropic_llm.get_num_tokens(answer['result'])

            logger.info(f'[Medical-Insights-QnA][{self.model_embeddings}] Embedding tokens for LLM call: '
                        f'{self.titan_llm.get_num_tokens(query) + self.prompt_template_tokens}')

            logger.info(f'[Medical-Insights-QnA][{self.model_id_llm}] Input tokens: {input_tokens} '
                        f'Output tokens: {output_tokens} LLM execution time: {time.time() - x}')

            logger.info(f"[Medical-Insights-QnA] LLM generated response for input query in {time.time() - x} seconds.")

            return answer
