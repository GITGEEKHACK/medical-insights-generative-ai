import json
import os
import time

from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from app import logger
from app.constant import BotoClient
from app.constant import MedicalInsights
from app.service.medical_document_insights.nlp_extractor import bedrock_client, get_llm_input_tokens


class DocTypeExtractor:

    def __init__(self):
        os.environ['AWS_DEFAULT_REGION'] = BotoClient.AWS_DEFAULT_REGION

        self.bedrock_client = bedrock_client
        self.model_id_llm = 'anthropic.claude-3-haiku-20240307-v1:0'
        self.model_embeddings = 'amazon.titan-embed-text-v1'

        self.anthropic_llm = BedrockChat(
            model_id=self.model_id_llm,
            model_kwargs={
                "max_tokens": 4000,
                "temperature": 0,
                "top_p": 0.01,
                "top_k": 1,
            },
            client=self.bedrock_client,
        )

        self.titan_llm = BedrockChat(model_id=self.model_embeddings, client=self.bedrock_client)
        self.bedrock_embeddings = BedrockEmbeddings(model_id=self.model_embeddings, client=self.bedrock_client)

    async def __process_document_type(self, output_text, document):

        start_index = output_text.find('{')
        end_index = output_text.rfind('}') + 1
        json_str = output_text[start_index:end_index]

        if len(json_str) == 0:
            return {"document_type": ""}

        data = json.loads(json_str)
        return {"document_type": data.get("document_type", ''), "document_name": document['name']}

    async def __classify_document_type(self, vectorstore_faiss, document):

        query = MedicalInsights.Prompts.DOC_TYPE_PROMPT
        prompt_template = MedicalInsights.Prompts.PROMPT_TEMPLATE

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        qa = RetrievalQA.from_chain_type(
            llm=self.anthropic_llm,
            chain_type="stuff",
            retriever=vectorstore_faiss.as_retriever(
                search_type="similarity", search_kwargs={"k": 1}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        answer = qa({"query": query})
        response = answer['result']
        final_response = await self.__process_document_type(response, document)
        return final_response

    async def __data_formatter(self, document):
        json_data = document['page_wise_text']
        raw_text = "".join(json_data.values())
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=20000, chunk_overlap=200
        )
        texts = text_splitter.split_text(raw_text)
        for text in texts:
            threshold = self.anthropic_llm.get_num_tokens(text)
            if threshold > 7000:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=10000, chunk_overlap=200
                )
                texts = text_splitter.split_text(raw_text)
                break
        docs = [Document(page_content=t) for t in texts]
        return docs

    async def __get_docs_embeddings(self, document):
        """ This method is used to prepare the embeddings and returns it """

        x = time.time()
        docs = await self.__data_formatter(document)

        emb_tokens = 0
        for i in docs:
            emb_tokens += self.titan_llm.get_num_tokens(i.page_content)

        y = time.time()
        logger.info(f'[Medical-Insights][Document-Type] Chunk Preparation Time: {y - x}')

        vector_embeddings = FAISS.from_documents(
            documents=docs,
            embedding=self.bedrock_embeddings,
        )
        logger.info(f'[Medical-Insights][Document-Type][{self.model_embeddings}] Input embedding tokens: {emb_tokens}'
                    f'and Generation time: {time.time() - y}')

        return vector_embeddings


    async def extract_document_type(self, document):
        """ This is expose method of the class """

        pdf_text = "".join(document['page_wise_text'].values()).strip()
        if not pdf_text:
            return {"document_type": "", "document_name": document['name']}
        else:
            t = time.time()
            embeddings = await self.__get_docs_embeddings(document)
            logger.info(f"[Medical-Insights][Document-Type] Embedding Generation for Document Type is completed in {time.time() - t} seconds.")

            t = time.time()
            document_type = await self.__classify_document_type(embeddings, document)
            logger.info(f"[Medical-Insights][Document-Type] Document Type Extraction is completed in {time.time() - t} seconds.")

            return document_type



