import boto3
import json
import base64
import logging
import os
from botocore.config import Config
from dotenv import load_dotenv
import argparse


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

SYSTEM = \
"""
你现在是一个专业的汽车配件识别专家。我会给你一张汽车配件的图片，请根据以下分类标准进行识别和判断：

配件分类标准：
1. 三大项围罩类：
   - 水箱前围罩
   - 左前挡泥板总成
   - 右前挡泥板总成
   - 右侧裙板
   - 左侧裙板总成
   - 右侧裙板
   - 左侧裙板
   - 左前翼子板防擦贴片
   - 右前翼子板防擦贴片
   - 后保角防水胶贴片
   - 前保防擦贴片
   - 右侧门槛装饰板
   - 左侧门槛装饰板
   - 前保/前保支架
   - OBD接口
   - 车内仪表总成
   - 右前座椅侧外壳安装架
   - 左前座椅侧外壳安装架
   - 右前门撞击传感器支架
   - 左前门撞击传感器支架
   - 左前门电动玻璃升降器
   - 右前门电动玻璃升降器

2. 前中网格类：
   - 前气罩(三元)
   - 外壳口门后饰板
   - 电池安全开关外壳
   - 左气罩前围罩
   - 右气罩前围罩
   - 左前门饰板
   - 右前门饰板
   - 前保热风导板
   - 右前大灯
   - 左前大灯
   - 主驾驶安全气囊
   - 前防撞梁
   - 前保饰板上围罩
   - 中央空调面板
   - 左前大灯

3. 外观中网格类：
   - 左后围罩
   - 左前门开内把手
   - 左前门扣手
   - 右前门开内把手
   - 右前门扣手
   - 左前门开关
   - 右前门开关
   - 左前门锁
   - 右前门锁
   - 左前AS柱板
   - 右前AS柱板
   - 左前轮毂轮胎
   - 右前轮毂轮胎
   - 左前AS柱
   - 右前AS柱
   - 左右轮毂轮胎
   - 主驾门锁
   - 后保角
   - 左前座椅调节器
   - 左车窗
   - 右车窗
   - 驾驶位
   - 仪表盘
   - 变速杆
   - 发动机盖
   - 格栅

4. 内饰类别：
   - 蓄电化装置
   - 水箱
   - 方向盘
   - 中控显示屏
   - 空调系统面板
   - 车内顶棚
   - 后排座椅

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
"comments":(如果图片不清晰，请说明无法准确识别的原因; 如有特殊安装位置或使用要求，请一并说明;如有其他可能，列出top3)
}
```





"""
class ImageClassifier:
    def __init__(self, model_id=LITE_MODEL_ID):
        self.model_id = model_id
        session = boto3.session.Session(region_name=os.getenv('AWS_REGION','us-east-1'))
        self.bedrock_runtime = session.client(service_name = 'bedrock-runtime', 
                                 config=config)

    def process(self,image_path) -> str:
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
                return response['output']['message']['content'][0]['text']
        except Exception as e:
            logger.error(e)
            return str(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Enter the path to your image files")
    args = parser.parse_args()
    image_classifier = ImageClassifier()
    results = []
    for image_path in os.listdir(args.folder):
        #if image_path is folder
        if os.path.isdir(os.path.join(args.folder, image_path)):
            for sub_image_path in os.listdir(os.path.join(args.folder, image_path)):
                if sub_image_path.endswith(('.jpg')):
                    file_path = os.path.join(args.folder, image_path, sub_image_path)
                    print(f"process image file:{sub_image_path}")
                    response = image_classifier.process(file_path)
                    json_obj = json.loads(response[:-3])
                    print(json_obj)
                    results.append({**json_obj,"image_file":file_path})

    #example response in json
    #{"category": "内饰类别", "sub_category": "发动机盖", "confidence": 90, "comments": "图片显示了汽车发动机舱的内部，包含发动机盖和一些管道及连接件。", "image_file":""}
