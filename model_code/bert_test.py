import os
import shutil
import datetime
import random
from tqdm import tqdm
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import tensorflow as tf
from transformers import BertTokenizer, get_linear_schedule_with_warmup, BertForSequenceClassification, EncodecModel, AutoProcessor
from torch.nn import CrossEntropyLoss
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, precision_score, recall_score, auc
from sklearn.metrics import confusion_matrix

# Bert
from datasets import load_dataset, Audio

# Setting
from transformers import EncodecModel, AutoProcessor

# Calculate
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, precision_score, recall_score, auc,roc_curve
from sklearn.metrics import confusion_matrix

# Encodec
from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor

# cuda
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"



<<<<<<< HEAD
with open("../dataset/dataset_load_2_1:1/asvspoof_test_dataset.pkl", "rb") as f:
    test_dataset = pickle.load(f)


Batch =1
=======
with open("dataset/dataset_load_2_1:1/asvspoof_test_dataset.pkl", "rb") as f:
    test_dataset = pickle.load(f)


Batch =32
>>>>>>> 4 commit

#Test dataset
testloader = DataLoader(test_dataset, batch_size=Batch, shuffle=True)

device_ids = [0, 1, 2]
device = torch.device('cuda')

#model 호출 
<<<<<<< HEAD
model = BertForSequenceClassification.from_pretrained('../run/saved_models/dataset_load_2_1:1/best_model_epoch_77',num_labels=2)
=======
model = BertForSequenceClassification.from_pretrained('run/saved_models/dataset_load_2_1:1/best_model_epoch_77',num_labels=2)
>>>>>>> 4 commit
model = DataParallel(model, device_ids=device_ids)


model.to(device) 
model.eval()

total_correct = 0
total_samples = len(testloader)
total_loss = 0

true_labels_test = []
predicted_labels_test = []
probs_list=[]
result = []

with torch.no_grad():
    for data in tqdm(testloader, desc="Processing Data"):
        audio_code = data["audio_code"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["label"].to(device)
        
        
        outputs = model(audio_code, token_type_ids=None, attention_mask=attention_mask)

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        positive_logits = logits[:, 1]
        probs = 1 / (1 + np.exp(-positive_logits))
        # accuracy

        audio_label = labels.to('cpu').numpy()
        pred_labels = np.argmax(logits, axis=1).flatten()
        true_labels_test.extend(audio_label)
        predicted_labels_test.extend(pred_labels)
        probs_list.extend(probs)
        
<<<<<<< HEAD
        
=======

#eer 수치         
>>>>>>> 4 commit
fpr, tpr, thresholds = roc_curve(true_labels_test, probs_list, pos_label=1)

eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, thresholds)(eer)

    
# 정확도
accuracy_test = accuracy_score(true_labels_test, predicted_labels_test)
print(f'accuracy_test: {accuracy_test}')

<<<<<<< HEAD

=======
>>>>>>> 4 commit
#recall precision AUPRC
recall_test=recall_score(true_labels_test, predicted_labels_test)
precision_test=precision_score(true_labels_test,  predicted_labels_test)


# 클래스 0에 대한 정확도
class_0_indices = [i for i in range(len(predicted_labels_test)) if predicted_labels_test[i] == 0]
true_label_class_0_test = [true_labels_test[i] for i in class_0_indices]
predicted_label_class_0_test = [predicted_labels_test[i] for i in class_0_indices]
class0_accuracy_test = accuracy_score(true_label_class_0_test, predicted_label_class_0_test)

# 클래스 1에 대한 정확도
class_1_indices = [i for i in range(len(predicted_labels_test)) if predicted_labels_test[i] == 1]
true_label_class_1_test = [true_labels_test[i] for i in class_1_indices]
predicted_label_class_1_test = [predicted_labels_test[i] for i in class_1_indices]
class1_accuracy_test = accuracy_score(true_label_class_1_test, predicted_label_class_1_test)

#균형 정확도
balanced_accuracy_test=(class0_accuracy_test+class1_accuracy_test)/2

result.append({
    'Test Accuracy': accuracy_test,
    'Test balanced Accuracy':balanced_accuracy_test,
    'Test Recall':recall_test,
    'Test Precision': precision_test,
    'Test class0 Accuracy':class0_accuracy_test,
    'Test class1 Accuracy':class1_accuracy_test,
    'EER':eer
})

<<<<<<< HEAD

df = pd.DataFrame(result)
df.to_csv('test_result_csv/dataset_load_2_1:1.csv', index=True)
=======
df = pd.DataFrame(result)
csv_dir="run/test_result_csv"

if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
df.to_csv('run/test_result_csv/dataset_load_2_1:1.csv', index=True)
>>>>>>> 4 commit


