import torch, os
import pandas as pd
from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
from torch.utils.data import Dataset
import logging
from CDataLoader import CDataLoader

from torch import cuda
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

device = 'cuda' if cuda.is_available() else 'cpu'
device

# 配置日志记录器
logging.basicConfig(
    filename='runtime.txt',
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
)

logging.info(f'{device}')

df_org = pd.read_csv("data/MDT-dataset.csv")

df_org = df_org.sample(frac=1.0, random_state=42)

# 删除某些category它对应的text过少的行
df_org = df_org[~df_org['category'].isin(['帮我下单换电申请换电', '打开倒车侧向辅助'])]

labels = df_org['category'].unique().tolist()
labels = [s.strip() for s in labels]

NUM_LABELS = len(labels)

id2label = {id: label for id, label in enumerate(labels)}

label2id = {label: id for id, label in enumerate(labels)}

# 创建一个新列以数字形式表示类别
df_org['labels_num'] = pd.factorize(df_org.category)[0]
df_org["labels"] = df_org.category.map(lambda x: label2id[x.strip()])

tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese", max_length=50)
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=NUM_LABELS, id2label=id2label,
                                                      label2id=label2id)
model.to(device)

SIZE = df_org.shape[0]

train_texts = list(df_org.text[:SIZE // 2])
val_texts = list(df_org.text[SIZE // 2:(3 * SIZE) // 4])
test_texts = list(df_org.text[(3 * SIZE) // 4:])
train_labels = list(df_org.labels[:SIZE // 2])
val_labels = list(df_org.labels[SIZE // 2:(3 * SIZE) // 4])
test_labels = list(df_org.labels[(3 * SIZE) // 4:])

logging.debug(f"{len(train_texts)}, {len(val_texts)}, {len(test_texts)}")

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings  = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataloader = CDataLoader(train_encodings, train_labels)
val_dataloader = CDataLoader(val_encodings, val_labels)
test_dataset = CDataLoader(test_encodings, test_labels)



# train
def compute_metrics(pred):
    """
    Computes accuracy, F1, precision, and recall for a given set of predictions.

    Args:
        pred (obj): An object containing label_ids and predictions attributes.
            - label_ids (array-like): A 1D array of true class labels.
            - predictions (array-like): A 2D array where each row represents
              an observation, and each column represents the probability of
              that observation belonging to a certain class.

    Returns:
        dict: A dictionary containing the following metrics:
            - Accuracy (float): The proportion of correctly classified instances.
            - F1 (float): The macro F1 score, which is the harmonic mean of precision
              and recall. Macro averaging calculates the metric independently for
              each class and then takes the average.
            - Precision (float): The macro precision, which is the number of true
              positives divided by the sum of true positives and false positives.
            - Recall (float): The macro recall, which is the number of true positives
              divided by the sum of true positives and false negatives.
    """
    # Extract true labels from the input object
    labels = pred.label_ids

    # Obtain predicted class labels by finding the column index with the maximum probability
    preds = pred.predictions.argmax(-1)

    # Compute macro precision, recall, and F1 score using sklearn's precision_recall_fscore_support function
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

    # Calculate the accuracy score using sklearn's accuracy_score function
    acc = accuracy_score(labels, preds)

    # Return the computed metrics as a dictionary
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

training_args = TrainingArguments(
    # The output directory where the model predictions and checkpoints will be written
    output_dir='"/content/drive/MyDrive/998/TTC4900Model',
    do_train=True,
    do_eval=True,
    #  The number of epochs, defaults to 3.0
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    # Number of steps used for a linear warmup
    warmup_steps=100,
    weight_decay=0.01,
    logging_strategy='steps',
   # TensorBoard log directory
    logging_dir='/content/drive/MyDrive/998/multi-class-logs',
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    fp16=True,
    load_best_model_at_end=True
)

trainer = Trainer(
    # the pre-trained model that will be fine-tuned
    model=model,
     # training arguments that we defined above
    args=training_args,
    train_dataset=train_dataloader,
    eval_dataset=val_dataloader,
    compute_metrics = compute_metrics
)

trainer.train()

q=[trainer.evaluate(eval_dataset=df_org) for df_org in [train_dataloader, val_dataloader, test_dataset]]

print(pd.DataFrame(q, index=["train","val","test"]).iloc[:,:5])
logging.debug(pd.DataFrame(q, index=["train","val","test"]).iloc[:,:5])

def predict(text):
    """
    Predicts the class label for a given input text

    Args:
        text (str): The input text for which the class label needs to be predicted.

    Returns:
        probs (torch.Tensor): Class probabilities for the input text.
        pred_label_idx (torch.Tensor): The index of the predicted class label.
        pred_label (str): The predicted class label.
    """
    # Tokenize the input text and move tensors to the GPU if available
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")

    # Get model output (logits)
    outputs = model(**inputs)

    probs = outputs[0].softmax(1)
    """ Explanation outputs: The BERT model returns a tuple containing the output logits (and possibly other elements depending on the model configuration). In this case, the output logits are the first element in the tuple, which is why we access it using outputs[0].

    outputs[0]: This is a tensor containing the raw output logits for each class. The shape of the tensor is (batch_size, num_classes) where batch_size is the number of input samples (in this case, 1, as we are predicting for a single input text) and num_classes is the number of target classes.

    softmax(1): The softmax function is applied along dimension 1 (the class dimension) to convert the raw logits into class probabilities. Softmax normalizes the logits so that they sum to 1, making them interpretable as probabilities. """

    # Get the index of the class with the highest probability
    # argmax() finds the index of the maximum value in the tensor along a specified dimension.
    # By default, if no dimension is specified, it returns the index of the maximum value in the flattened tensor.
    pred_label_idx = probs.argmax()

    # Now map the predicted class index to the actual class label
    # Since pred_label_idx is a tensor containing a single value (the predicted class index),
    # the .item() method is used to extract the value as a scalar
    pred_label = model.config.id2label[pred_label_idx.item()]

    return probs, pred_label_idx, pred_label

# Test with a example text
text = "开一下WIFI，谢谢！"
predict(text)

# save model
model_path = "model/MDT-text-classification-model-New"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)