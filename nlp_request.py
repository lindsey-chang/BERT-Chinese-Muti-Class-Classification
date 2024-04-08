from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, BertForSequenceClassification, \
    BertTokenizerFast, pipeline
# API接口的URL
from flask import Flask, jsonify,request
app = Flask(__name__)
model_path = "model/MDT-text-classification-model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
# {'label': '打开WIFI', 'score': 0.39187902212142944}]
print(nlp("123")[0]["label"])
print(nlp("123")[0]["score"])

def classify_text(text):
    # 在这里实现你的分类模型或其他逻辑
    # 假设你已经有一个函数，可以接收文本，返回标签和分数
    a = nlp(text)[0]
    label = a["label"]
    score = a["score"]
    return label, score

@app.route('/api/nlp', methods=['POST'])
def hello():
    data = request.get_json()
    text = data['text']
    label,score = classify_text(text)
    response = {
        'label': label,
        'score': score
    }
    return jsonify(response)
if __name__ == '__main__':
    app.run()