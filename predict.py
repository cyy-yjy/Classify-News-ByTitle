import os
import torch
import torch.nn as nn
from werkzeug.wrappers import BaseResponse
from pytorch_pretrained_bert import BertModel, BertTokenizer
from flask import Flask, request, jsonify

app = Flask(__name__)
# 识别的类型
key = {0: 'finance',
       1: 'realty',
       2: 'stocks',
       3: 'education',
       4: 'science',
       5: 'society',
       6: 'politics',
       7: 'sports',
       8: 'game',
       9: 'entertainment'
       }


class Config:
    """配置参数"""

    def __init__(self):
        cru = os.path.dirname(__file__)
        self.class_list = [str(i) for i in range(len(key))]  # 类别名单
        self.save_path = 'THUCNews/saved_dict/bert.ckpt'
        self.device = torch.device('cpu')
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 3  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768

    def build_dataset(self, text):
        lin = text.strip()
        pad_size = len(lin)
        token = self.tokenizer.tokenize(lin)
        token = ['[CLS]'] + token
        token_ids = self.tokenizer.convert_tokens_to_ids(token)
        mask = []
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
        return torch.tensor([token_ids], dtype=torch.long), torch.tensor([mask])


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[1]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out


config = Config()
model = Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path, map_location='cpu'))

def prediction_model(text):
    """输入一句问话预测"""
    data = config.build_dataset(text)
    with torch.no_grad():
        outputs = model(data)
        num = torch.argmax(outputs)
    return key[int(num)]
@app.route("/predict", methods=['POST'])
def predict():
    print(1)
    response = {
        "response": {
            "isError": True,
            "msg": "", }
    }
    try:
        data = request.get_json()
        text = data.get('text')
        print("接收到post请求", text)
        ans=prediction_model(text)
        response['response']['isError'] =False
        response['response']['data'] = ans
    except Exception as e:
        response['response']['msg'] = str(e)
    return jsonify(response)

if __name__ == '__main__':
    print("已经进入main函数")
    print(prediction_model('锌价难续去年辉煌'))
    app.run(
        host='0.0.0.0',
        port=6000,
        debug=True
    )
    # 用while循环不断输入句子，进行预测
    # while 1:
    #     text = input('请输入句子：')
    #     if text == 'exit':
    #         break
    #     print(prediction_model(text))