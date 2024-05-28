import time

from app import logger
from app.service.medical_document_insights.nlp_extractor.document_qna import DocumentQnA


async def get_query_response(query, project_path):

    x = time.time()
    document_qna = DocumentQnA()
    response = await document_qna.get_query_response(query, project_path)
    logger.info(f"[Medical-Insights-QnA] Query response generated in {time.time() - x} seconds.")
    return response
