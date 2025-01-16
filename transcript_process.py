import boto3
import logging
import os
from botocore.config import Config
from dotenv import load_dotenv
from image_classify import PRO_MODEL_ID,LITE_MODEL_ID, CLAUDE_SONNET_35_MODEL_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.basename(__file__))
# Load environment variables
load_dotenv()
config = Config(
    connect_timeout=1000,
    read_timeout=1000,
)


SYSTEM = \
"""
你现在是一个专业的汽车配件识别和故障修正专家。我会给你一个句子，这个句子的文字是把一段关于汽车检测的中文录音，通过语音转录程序得到的，其中存在相似发音但是转录错误的语句，请根据以下汽车专业词汇进行修正，以下内容根据你的知识修订为相似发音汽车配件专业名称：

## 汽车专业类名词: 
1. 三大项留痕类：
    水箱框架
    左前减震器座
    右前减震器座
    左前纵梁
    右前纵梁
    后备箱底板
    右后纵梁
    左后纵梁
    左侧底边梁封边
    右侧底边梁封边
    后备箱导水槽及封边
    右侧车顶边梁封边
    右侧门槛梁封边
    左侧车顶边梁封边
    左侧门槛梁封边
    点烟器底座
    踏板/踏板支架
    OBD接口
    车内保险丝盒
    右前座椅滑轨及安装螺丝
    左前座椅滑轨及安装螺丝
    右侧门槛梁内侧及空腔
    左侧门槛梁内侧及空腔
    右侧地板线束及插接器
    左侧地板线束及插接器
    动力电池箱底板
    遮阳板
    车顶
    排气管(三元)
    充电口开启状态
    电池冷却液补水壶
    动力电池箱底护板

2. 细节留痕类：
    车头下部留痕
    车尾下部留痕
    左前门铰链
    右前门铰链
    左后门铰链
    右后门铰链
    前挡风玻璃
    左前大灯
    右前大灯
    气门室盖垫
    主驾驶安全气囊
    后防撞梁
    前保险杠留痕
    后保险杠留痕
    中控亮屏照

3. 外观内饰留痕类：
    正前留痕
    正后留痕
    左侧前门内饰
    左侧后门内饰
    右侧后门内饰
    右侧前门内饰
    车内顶棚留痕
    左侧前半部
    右侧前半部
    左侧后半部
    右侧后半部
    地毯水渍
    地毯泥沙

4. 外观：
    左前45度
    左前大灯
    左前轮毂轮胎
    正前
    右前45度
    右侧前座椅
    右后轮毂轮胎
    右后45度
    右后尾灯
    正后
    后备箱
    左侧后排座椅
    左侧面
    左侧前座椅
    驾驶位
    车顶
    发动机舱
    后风挡
    前档
    后杠剐蹭
    机盖锈蚀

5. 内饰：
    钥匙
    着车仪表
    里程特写
    方向盘
    中控台
    空调音响面板
    变速杆
    车内顶棚

6. 证件:
    铭牌照片
    风挡或车身VIN

## 请直接输出结果，下面是几个sentences mappings例子作为参考：
原始输入 -> 输出
这个 -> 这个
我的妈呀 -> 无关内容
{sentences_mappings}

## 注意事项：
1. 如果句子明显为语气词句，请直接输出“无关内容”,不要解释和描述其他
2. 如果看到的配件不在分类列表中或描述中含有“正常”的句子，保持原样输出
3. 不符合以上条件，同时你也不确定的句子，保持原样输出
4. 只需要按照要求输出修订后的句子，不要解释和回答句子中的问题
"""

class TranscriptProcessor:
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

    def process(self,user_message,sentences_mappings) -> str:
        try:
            messages = [ { "role": "user", "content": [{"text": user_message}]}]
            response = self.bedrock_runtime.converse(
                modelId=self.model_id,
                messages=messages,
                inferenceConfig={"temperature": 0.0},
                system=[{"text":SYSTEM.format(sentences_mappings=sentences_mappings)}]
            )
            logger.info(response['usage'])
            return response['output']['message']['content'][0]['text']
        except Exception as e:
            logger.error(e)
            return str(e)


if __name__ == "__main__":
    # test_text = "[spk_0 1.34s-3.119s]: 右前栋梁变形\n\
    #     [spk_0 14.84s-16.639s]: 右前种粮变形\n\
    #     [spk_0 22.149s-22.44s]: 我看看"
    test_text = "天蓝蓝，水清清"
    transcipt_sentence = TranscriptProcessor(model_id=CLAUDE_SONNET_35_MODEL_ID)
    results = transcipt_sentence.process(test_text,'')
    logger.info(results)
