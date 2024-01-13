# BERT-Chinese-Muti-Class-Classification
UOW CSIT998 Professional Capstone Project(Partial code)

## Environment

```bash
conda create -n bert-multi-class python=3.10.13
conda activate bert-multi-class
pip install torch pandas transformers datasets
pip install --upgrade transformers
pip install accelerate -U
pip install "transformers[torch]"
pip install scikit-learn
```


## Demo
- Step 1: Download the fine-tuned model from Google Drive to folder `model`(Because the model is too large).
- Step 2: In `demo.py` you can run the simple demo project.