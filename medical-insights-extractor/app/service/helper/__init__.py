import boto3
from botocore.config import Config
from app.constant import BotoClient

textract_client = boto3.client(service_name='textract', region_name=BotoClient.AWS_DEFAULT_REGION, config=Config(BotoClient.read_timeout))