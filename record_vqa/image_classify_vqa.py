import boto3
import json
from pathlib import Path
import os
from botocore.config import Config
from dotenv import load_dotenv
import argparse
import tqdm
import logging
import time
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import pandas as pd
import uuid
from json.decoder import JSONDecodeError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.basename(__file__))

# Load environment variables
load_dotenv()
config = Config(
    connect_timeout=1000,
    read_timeout=1000,
)

PRO_MODEL_ID = "us.amazon.nova-pro-v1:0"
LITE_MODEL_ID = "us.amazon.nova-lite-v1:0"
CLAUDE_SONNET_35_MODEL_ID = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

SYSTEM_PROMPT = "你是一位专业的汽车配件识别专家。你正在对一辆汽车的图片进行识别，判断其拍摄位置及包含的零部件。"

USER_PROMPT = """
<role>
你是一位专业的汽车配件识别专家。你正在对一辆汽车的图片进行识别，判断其拍摄位置及包含的零部件。
<role>

<workflow>
1. 仔细查看输入的汽车拍摄照片，对拍摄位置和图片包含的汽车零部件类别进行思考；
2. 预测拍摄位置("<shooting>"罗列了候选的拍摄位置)；
3. 识别图片中包含的汽车零部件，并给出具体类别("<category>"罗列了候选的零部件类别，请直接忽略没有包含在内的物体)和置信度(0-100的整数)；
4. 参考<output_format>进行输出。
</workflow>

<rules>
- 在输出零部件类别时，请仅输出你能确认的零部件，而忽略模棱两可的。
</rules>

<shooting>
- 汽车前方
- 汽车侧方
- 汽车后方
- 汽车顶部
- 汽车底部
- 汽车内部
</shooting>

<category>
- 轮胎
- 座椅
- 车灯
- 后备箱
- 车顶
- 车门
- 车窗
- 内饰
- 后视镜
- 前挡风玻璃
- 后挡风玻璃
- 发动机舱
- 前舱盖
- 钥匙
- 仪表盘
- 方向盘
- 中控台
- 控制面板
- 车内顶棚
- 遮阳板
- 充电口
- 车身VIN
</category>

<output_format>
直接输出以下json文本，不要增加任何解释。
```json
{
    "思考": "思考信息",
    "拍摄位置": "拍摄位置(参考<shooting>)",
    "零部件": {
        "零部件1(参考<category>)": 置信度,
        "零部件2(参考<category>)": 置信度,
        ...
    }
}
```
</output_format>
"""

def parse_objects(objects, shooting):
    objects_str = ', '.join(objects.keys())
    return f"{shooting}: {objects_str}"

def clean_results(results, threshold):
    cleaned_results = []
   
    for result in results:
        if result is not None and result['拍摄位置'] != "其他":
            filtered_objects = {k: v for k, v in result['零部件'].items() if v >= threshold}
            
            if filtered_objects:
                result['零部件'] = filtered_objects
                result['key_text'] = parse_objects(filtered_objects, result['拍摄位置'])
                cleaned_results.append(result)
    
    return cleaned_results

class ImageClassifier:
    def __init__(self, model_id=LITE_MODEL_ID):
        self.model_id = model_id
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

        self.bedrock_runtime = session.client(service_name='bedrock-runtime', 
                                              config=config)

    def process(self, image_path, max_retry=100) -> str:
        for attempt in range(max_retry):
            try:
                with open(image_path, "rb") as file:
                    media_bytes = file.read()
                    messages = [ 
                        { "role": "user", "content": [
                            {"image": {"format": "jpeg", "source": {"bytes": media_bytes}}},
                            {"text": USER_PROMPT}
                        ]},
                        {
                            "role": "assistant", "content": [
                                {"text": "```json"}
                            ]
                        }
                    ]

                    response = self.bedrock_runtime.converse(
                        modelId=self.model_id,
                        messages=messages,
                        inferenceConfig={"temperature": 0.0, 'stopSequences': ['```']},
                        system=[{"text": SYSTEM_PROMPT}]
                    )
                    
                    return response['output']['message']['content'][0]['text'].strip('```')
            except Exception as e:
                logger.warning(f'Retry {attempt + 1}/{max_retry} due to error: {e}')
                time.sleep(random.randint(2, 5))
        raise RuntimeError(f'Calling BedRock failed after retrying for {max_retry} times.')

def upload_to_s3(file_path, bucket_name, s3_key):
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        return f"s3://{bucket_name}/{s3_key}"
    except Exception as e:
        logger.error(f"Error uploading {file_path} to S3: {e}")
        return None

def process_dataset(data, output_dir, s3_bucket):
    uuid_str = str(uuid.uuid4())
    job_name = f"{int(time.time())}-{uuid_str}"
    for index, row in enumerate(data):
        csv_filename = f'{index:08}.csv'
        csv_path = os.path.join(output_dir, csv_filename)

        image_path = os.path.join(row['folder'], row['image_file'])
        s3_key_image = f"{job_name}/images/{row['image_file']}"
        s3_key_csv = f"{job_name}/knowledge_base/{csv_filename}"

        s3_path = upload_to_s3(image_path, s3_bucket, s3_key_image)

        if s3_path:
            row['s3_path'] = s3_path

            pd.DataFrame([row]).to_csv(csv_path, index=False)

            metadata = {
                'metadataAttributes': {},
                'documentStructureConfiguration': {
                    'type': 'RECORD_BASED_STRUCTURE_METADATA',
                    'recordBasedStructureMetadata': {
                        'contentFields': [{
                            'fieldName': 'key_text'
                        }],
                        'metadataFieldsSpecification': {
                            'fieldsToInclude': [{
                                'fieldName': 's3_path'
                            }],
                            'fieldsToExclude': []
                        }
                    }
                }
            }

            metadata_filename = f'{csv_filename}.metadata.json'
            metadata_path = os.path.join(output_dir, metadata_filename)

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            upload_to_s3(csv_path, s3_bucket, s3_key_csv)
            upload_to_s3(metadata_path, s3_bucket, s3_key_csv + '.metadata.json')
    return job_name

def process_image(image_classifier, folder, image_name):
    if image_name.endswith('.jpg'):
        file_path = os.path.join(folder, image_name)
        response = image_classifier.process(file_path)
        while True:
            try:
                json_obj = json.loads(response)
                break
            except JSONDecodeError as e:
                response = image_classifier.process(file_path)
                logger.info(f'Error decode JSON: {e}')
        logger.info(image_name)
        logger.info(json_obj)
        return {"image_file": image_name, "folder": folder, **json_obj}
    return None

def process_images(folder, threshold, s3_bucket):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    results_dir = Path(folder) / "output" / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    image_classifier = ImageClassifier(model_id=PRO_MODEL_ID)

    image_names = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])
    process_func = partial(process_image, image_classifier, folder)

    # with ThreadPoolExecutor() as executor:
    #     results = list(tqdm.tqdm(executor.map(process_func, image_names), total=len(image_names)))
    results = []
    for image_name in tqdm.tqdm(image_names):
        results.append(process_func(image_name))

    results = clean_results(results, threshold)
    job_name = process_dataset(results, results_dir, s3_bucket)

    return job_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Enter the path to your image files")
    parser.add_argument("--s3_bucket", help="S3 bucket name for image upload")
    parser.add_argument("--threshold", type=int, default=90)
    args = parser.parse_args()

    job_name = process_images(args.folder, args.threshold, args.s3_bucket)
    print(f"Job completed. Job name: {job_name}")
