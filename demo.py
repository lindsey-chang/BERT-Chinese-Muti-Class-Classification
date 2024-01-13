from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, BertForSequenceClassification, \
    BertTokenizerFast, pipeline

model_path = "model/MDT-text-classification-model"

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

print(nlp("运转车顶灯，好吗？"))
print(nlp("小爱同学，给我把遮阳帘打开。"))
print(nlp("开启顶灯。"))
