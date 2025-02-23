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
你现在是一个专业的汽车配件识别和故障修正专家。我会给你一个句子，这个句子的文字是把一段关于汽车检测的中文录音，通过语音转录程序得到的，其中存在相似发音但是转录错误的语句，请根据汽车专业词汇进行修正.

## 下面是几个sentences mappings例子作为参考：
{sentences_mappings}

## 注意事项：
1. 如果句子明显为语气词句，请直接输出“无关内容”,不要解释和描述其他
2. 如果看到的配件不在分类列表中或描述中含有“正常”的句子，保持原样输出
3. 不符合以上条件，同时你也不确定的句子，保持原样输出
4. 只需要按照要求输出修订后的句子，不要解释和回答句子中的问题
5. 如果句子已经在`正确文本`列中，则保持原样输出
6. 如果句子是正常的汽车相关用语，例如`左前轮毂刮蹭`, 保持原样输出
6. 直接输出结果,不要任何解释
"""

class TranscriptProcessor:
    def __init__(self, model_id=PRO_MODEL_ID):
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
            messages = [ { "role": "user", "content": [{"text": user_message}]},
                        { "role": "assistant", "content": [{"text": "纠正后结果是:"}]}]
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
