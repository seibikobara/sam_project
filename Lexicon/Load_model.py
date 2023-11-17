!pip install simpletransformers
from simpletransformers.classification import ClassificationModel

model = ClassificationModel("bert", "/content/drive/MyDrive/BMI 550/Final_project/BERT_large_model3")

test_text=list(df_test['text'])
predictions, raw_outputs = model.predict(test_text)
