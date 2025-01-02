import boto3
import logging
import os
from botocore.config import Config
from dotenv import load_dotenv
from image_classify import PRO_MODEL_ID,LITE_MODEL_ID, CLAUDE_SONNET_35_MODEL_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables
load_dotenv()
config = Config(
    connect_timeout=1000,
    read_timeout=1000,
)


SYSTEM = \
"""
你现在是一个专业的汽车配件识别和故障修正专家。我会给你一个句子，请根据以下分类标准进行修正：

配件分类标准：
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

4. 外观图片：
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

4. 内饰图片：
    钥匙
    着车仪表
    里程特写
    方向盘
    中控台
    空调音响面板
    变速杆
    车内顶棚
    铭牌照片
    风挡或车身VIN

   #注意事项：
1. 如果句子明显为语气词句或与上述配件分类标准无关的内容，请直接输出“无关内容”
2. 如果看到的配件不在分类列表中，保持原样输出
3. 只需要按照要求输出修订后的句子，不要解释和回答句子中的问题

#请按以下原始格式输出结果，下面是两个例子作为参考：
例如当你收到的句子是：这个
你修订后的输出应该为： 无关内容
例如当你收到的句子是：右前种粮变形 
你修订后的输出应该为： 右前纵梁变形
例如当你收到的输入为：我的妈呀
你修订后的输出应该为：无关内容
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

    def process(self,user_message) -> str:
        try:
            messages = [ { "role": "user", "content": [{"text": user_message}]}]
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
    # test_text = "[spk_0 1.34s-3.119s]: 右前栋梁变形\n\
    #     [spk_0 14.84s-16.639s]: 右前种粮变形\n\
    #     [spk_0 22.149s-22.44s]: 我看看"
    test_text = "天蓝蓝，水清清"
    transcipt_sentence = TranscriptProcessor(model_id=CLAUDE_SONNET_35_MODEL_ID)
    results = transcipt_sentence.process(test_text)
    logger.info(results)