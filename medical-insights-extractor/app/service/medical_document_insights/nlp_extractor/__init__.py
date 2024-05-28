import boto3
from botocore.config import Config
from app.constant import BotoClient

bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=BotoClient.AWS_DEFAULT_REGION,
                              config=Config(read_timeout=BotoClient.read_timeout,
                                            connect_timeout=BotoClient.connect_timeout))


def get_llm_input_tokens(llm, llm_response):
    input_tokens = 0
    input_tokens += llm.get_num_tokens(llm_response['query'])

    for i in llm_response['source_documents']:
        input_tokens += llm.get_num_tokens(i.page_content)

    return input_tokens
