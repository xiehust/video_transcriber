import boto3
import json
import base64
import logging
import os
from botocore.config import Config
from dotenv import load_dotenv
import argparse
import logging
import time
import random


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.basename(__file__))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables
load_dotenv()
config = Config(
       connect_timeout=1000,
    read_timeout=1000,
)



PRO_MODEL_ID = "us.amazon.nova-pro-v1:0"
LITE_MODEL_ID = "us.amazon.nova-lite-v1:0"
CLAUDE_SONNET_35_MODEL_ID = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

SYSTEM = \
"""
你现在是一个专业的汽车配件识别专家。你现在正在给一辆汽车做划痕，外观的识别检测，我会给你一张汽车配件的图片，请根据以下分类标准进行识别和判断：

配件分类标准：
1. 三大项留痕：
    - 水箱框架
    - 左前减震器座
    - 右前减震器座
    - 左前纵梁
    - 右前纵梁
    - 后备箱底板
    - 右后纵梁
    - 左后纵梁
    - 左侧底边梁封边
    - 右侧底边梁封边
    - 后备箱导水槽及封边
    - 右侧车顶边梁封边
    - 右侧门槛梁封边
    - 左侧车顶边梁封边
    - 左侧门槛梁封边
    - 点烟器底座
    - 踏板/踏板支架
    - OBD接口
    - 车内保险丝盒
    - 右前座椅滑轨及安装螺丝
    - 左前座椅滑轨及安装螺丝
    - 右侧门槛梁内侧及空腔
    - 左侧门槛梁内侧及空腔
    - 右侧地板线束及插接器
    - 左侧地板线束及插接器
    - 动力电池箱底板
    - 遮阳板
    - 车顶
    - 排气管(三元)
    - 充电口开启状态
    - 电池冷却液补水壶
    - 动力电池箱底护板

2. 细节留痕：
    - 车头下部留痕
    - 车尾下部留痕
    - 左前门铰链
    - 右前门铰链
    - 左后门铰链
    - 右后门铰链
    - 前挡风玻璃
    - 左前大灯
    - 右前大灯
    - 气门室盖垫
    - 主驾驶安全气囊
    - 后防撞梁
    - 前保险杠留痕
    - 后保险杠留痕
    - 中控亮屏照

3. 外观内饰留痕：
    - 正前留痕
    - 正后留痕
    - 左侧前门内饰
    - 左侧后门内饰
    - 右侧后门内饰
    - 右侧前门内饰
    - 车内顶棚留痕
    - 左侧前半部
    - 右侧前半部
    - 左侧后半部
    - 右侧后半部

4. 外观图片：
    - 左前45度
    - 左前大灯
    - 左前轮毂轮胎
    - 正前
    - 右前45度
    - 右侧前座椅
    - 右后轮毂轮胎
    - 右后45度
    - 右后尾灯
    - 正后
    - 后备箱
    - 左侧后排座椅
    - 左侧面
    - 左侧前座椅
    - 驾驶位
    - 车顶
    - 发动机舱

5. 内饰图片:
    - 钥匙
    - 着车仪表
    - 里程特写
    - 方向盘
    - 中控台
    - 空调音响面板
    - 变速杆
    - 车内顶棚

6. 证件照片:
    - 铭牌照片
    - 风挡或车身VIN

#注意事项：
1. 如果图片不清晰，请说明无法准确识别的原因
2. 如果看到的配件不在分类列表中，请说明具体情况
3. 如有特殊安装位置或使用要求，请一并说明

#请按以下json格式输出识别结果：
```json
{
"category":(从上述4个大类中选择),
"sub_category":(从相应类别中选择具体配件)
"confidence": 整数(0-100), 越大置信度越高
"comments":(如果图片不清晰，请说明无法准确识别的原因; 如有特殊安装位置或使用要求，请一并说明;如有其他可能，列出top3, 不要带换行符)
}
```





"""
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

        self.bedrock_runtime = session.client(service_name = 'bedrock-runtime', 
                                 config=config)

    def process(self,image_path, max_retry=20) -> str:
        for attempt in range(max_retry):
            try:
                with open(image_path, "rb") as file:
                    media_bytes = file.read()
                    user_message = "请对图片进行分类."
                    messages = [ 
                        { "role": "user", "content": [
                            {"image": {"format": "jpeg", "source": {"bytes": media_bytes}}},
                            {"text": user_message}
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
                        inferenceConfig={"temperature": 0.0},
                        system=[{"text":SYSTEM}]
                    )
                    logger.info(response['usage'])
                    return response['output']['message']['content'][0]['text']
            except Exception as e:
                logger.warning(f'Retry {attempt + 1}/{max_retry} due to error: {e}')
                time.sleep(random.randint(1, 3))
        raise RuntimeError(f'Calling BedRock failed after retrying for {max_retry} times.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Enter the path to your image files")
    args = parser.parse_args()
    image_classifier = ImageClassifier(model_id=LITE_MODEL_ID)
    results = []
    for image_path in os.listdir(args.folder):
        #if image_path is folder
        if os.path.isdir(os.path.join(args.folder, image_path)):
            for sub_image_path in os.listdir(os.path.join(args.folder, image_path)):
                if sub_image_path.endswith(('.jpg')):
                    file_path = os.path.join(args.folder, image_path, sub_image_path)
                    print(f"process image file:{sub_image_path}")
                    response = image_classifier.process(file_path)
                    # response = response.replace("\n","\\n")
                    print(response)
                    json_obj = json.loads(response[:-3])
                    # print(json_obj)
                    results.append({**json_obj,"image_file":file_path})

    #example response in json
    #{"category": "内饰类别", "sub_category": "发动机盖", "confidence": 90, "comments": "图片显示了汽车发动机舱的内部，包含发动机盖和一些管道及连接件。", "image_file":""}
