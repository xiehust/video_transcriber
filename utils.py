import os
import random
import time
from dotenv import load_dotenv

import boto3
from botocore.config import Config


load_dotenv()
config = Config(
       connect_timeout=1000,
    read_timeout=1000,
)

def call_converse(model_id,
                  messages,
                  system_prompt,
                  max_tokens=4096,
                  temperature=0.01,
                  stop_sequences=[],
                  retry=20):
    if os.getenv('AWS_ACCESS_KEY_ID_LLM') and os.getenv('AWS_SECRET_ACCESS_KEY_LLM'):
        session = boto3.session.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID_LLM'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY_LLM'),
            region_name=os.getenv('AWS_REGION','us-east-1')
        )
    else:
        session = boto3.session.Session(
            region_name=os.getenv('AWS_REGION','us-east-1')
        )

    bedrock_runtime = session.client(service_name='bedrock-runtime',
                                config=config)


    for attempt in range(retry):
        try:
            system = [{'text': system_prompt}]
            inf_params = {
                'maxTokens': max_tokens,
                'temperature': temperature,
                'stopSequences': stop_sequences,
            }
            if 'claude-3-5-haiku' in model_id:
                stream = bedrock_runtime.converse_stream(
                    modelId=model_id,
                    messages=messages,
                    system=system,
                    inferenceConfig=inf_params,
                    performanceConfig={
                        'latency': 'optimized'
                    }).get('stream')
            else:
                stream = bedrock_runtime.converse_stream(
                    modelId=model_id,
                    messages=messages,
                    system=system,
                    inferenceConfig=inf_params).get('stream')

            msg = ''
            invoke_metrics = None
            if stream:
                for event in stream:
                    if 'contentBlockDelta' in event:
                        msg += event['contentBlockDelta']['delta']['text']
                    if 'metadata' in event:
                        invoke_metrics = {
                            **event['metadata']['usage'],
                            **event['metadata']['metrics']
                        }
            return msg, invoke_metrics

        except Exception as e:
            print(f'[WARNING] Retry {attempt + 1}/{retry} due to error: {e}')
            time.sleep(random.randint(1, 3))

    raise RuntimeError('Calling BedRock failed after retrying for '
                       f'{retry} times.')



def call_retrieve(knowledge_base_id, query, number_of_results=5):
    if os.getenv('AWS_ACCESS_KEY_ID_LLM') and os.getenv('AWS_SECRET_ACCESS_KEY_LLM'):
        session = boto3.session.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID_LLM'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY_LLM'),
            region_name=os.getenv('AWS_REGION','us-east-1')
        )
    else:
        session = boto3.session.Session(
            region_name=os.getenv('AWS_REGION','us-east-1')
        )

    bedrock_runtime = session.client(service_name='bedrock-agent-runtime',
                                config=config)

    all_info = []
    try:
        response = bedrock_runtime.retrieve(knowledgeBaseId=knowledge_base_id,
                                            retrievalQuery={'text': query},
                                            retrievalConfiguration={'vectorSearchConfiguration':{'numberOfResults': number_of_results}})
        return response['retrievalResults']

    except Exception as e:
        print(f'An error occurred: {str(e)}')

    return all_info

