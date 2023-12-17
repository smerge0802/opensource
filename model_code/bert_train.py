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
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, precision_score, recall_score, auc
from sklearn.metrics import confusion_matrix

# Encodec
from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor

# cuda
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

#dataset load
<<<<<<< HEAD
with open("../dataset/dataset_load_2_1:1/asvspoof_train_dataset.pkl", "rb") as f:
    train_dataset = pickle.load(f)

with open("../dataset/dataset_load_2_1:1/asvspoof_validation_dataset.pkl", "rb") as f:
=======
with open("dataset/dataset_load_2_1:1/asvspoof_train_dataset.pkl", "rb") as f:
    train_dataset = pickle.load(f)

with open("dataset/dataset_load_2_1:1/asvspoof_validation_dataset.pkl", "rb") as f:
>>>>>>> 4 commit
    validation_dataset = pickle.load(f)


Batch = 32
trainloader = DataLoader(train_dataset, batch_size=Batch, shuffle=True)
validationloader = DataLoader(validation_dataset, batch_size=Batch, shuffle=True)

# cuda
device_ids = [0, 1, 2]
device = torch.device('cuda')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
model = DataParallel(model, device_ids=device_ids)
model.to(device)


epochs = 100

#optimizer
optimizer = Adam(model.parameters(), lr=1e-7)
total_steps = len(trainloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

#SEED
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

#accuracy
best_val_accuracy = 0.0

# tensorboard
<<<<<<< HEAD
log_dir = '../run/logs/dataset_load_2_1:1'
=======
log_dir = 'run/logs/dataset_load_2_1:1'
>>>>>>> 4 commit
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
writer = SummaryWriter(log_dir=log_dir)
model.zero_grad()

# loss funtion
best_loss = 130
best_model_path=''
best_model_img=''

# csv
data = []

for epoch_i in range(0, epochs):
    total_loss = 0
    model.train()
    
    true_labels_train = []
    predicted_labels_train = []

    # 트레이닝 데이터로더를 이용해 배치 단위로 학습
    for batch in tqdm(trainloader, desc="Processing Data"):
        audio_code = batch["audio_code"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # 모델에 입력 데이터를 전달하여 출력을 얻음
        outputs = model(audio_code, token_type_ids=None, attention_mask=attention_mask, labels=labels)
        
        # Accuracy 계산
        logits = outputs.logits.detach().cpu().numpy()
        audio_label = labels.to('cpu').squeeze().numpy()
        pred_labels = np.argmax(logits, axis=1).flatten()
        true_labels_train.extend(audio_label)
        predicted_labels_train.extend(pred_labels)

        # Loss 계산
        loss = outputs[0]
        total_loss += loss.mean()

        # 역전파 및 가중치 갱신
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        
        
    avg_train_loss = total_loss / len(trainloader)
    print(f'avg_train_loss: {avg_train_loss}')
    writer.add_scalar("Train Loss", avg_train_loss, epoch_i)

    # accuracy
    accuracy_train = accuracy_score(true_labels_train, predicted_labels_train)
    writer.add_scalar("Train Accuracy", accuracy_train, epoch_i)
    print(f'accuracy_train: {accuracy_train}')

    #recall precision AUPRC
    recall_train=recall_score(true_labels_train, predicted_labels_train)
    precision_train=precision_score(true_labels_train, predicted_labels_train)
    
    # class 0 accuracy
    class_0_indices = [i for i in range(len(predicted_labels_train)) if predicted_labels_train[i] == 0]
    true_label_class_0_train = [true_labels_train[i] for i in class_0_indices]
    predicted_label_class_0_train = [predicted_labels_train[i] for i in class_0_indices]
    class0_accuracy_train = accuracy_score(true_label_class_0_train, predicted_label_class_0_train)
  
    # class 1 accuracy
    class_1_indices = [i for i in range(len(predicted_labels_train)) if predicted_labels_train[i] == 1]
    true_label_class_1_train = [true_labels_train[i] for i in class_1_indices]
    predicted_label_class_1_train = [predicted_labels_train[i] for i in class_1_indices]
    class1_accuracy_train = accuracy_score(true_label_class_1_train, predicted_label_class_1_train)

    # balanced_accuracy
    balanced_accuracy_train = (class1_accuracy_train + class0_accuracy_train)/2
    
    
    # test
    model.eval()
    total_loss = 0
    
    true_labels_test = []
    predicted_labels_test = []

    with torch.no_grad():
        for batch in tqdm(validationloader, desc="Processing Data"):
            audio_code = batch["audio_code"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # 모델에 입력 데이터를 전달하여 출력을 얻음
            outputs = model(audio_code, token_type_ids=None, attention_mask=attention_mask, labels=labels)

            # Accuracy 계산
            logits = outputs.logits.detach().cpu().numpy()
            audio_label = labels.to('cpu').squeeze().numpy()
            pred_labels = np.argmax(logits, axis=1).flatten()
            true_labels_test.extend(audio_label)
            predicted_labels_test.extend(pred_labels)

            # Loss 계산
            loss = outputs[0]
            total_loss += loss.mean()

            
    # Loss
    avg_test_loss = total_loss / len(validationloader)
    print(f'avg_test_loss: {avg_test_loss}')
    writer.add_scalar("Validtaion Loss", avg_test_loss, epoch_i)

    # Acccuracy
    accuracy_test = accuracy_score(true_labels_test, predicted_labels_test)
    print(f'accuracy_test: {accuracy_test}')
    writer.add_scalar("Validation Accuracy", accuracy_test, epoch_i)
    
    #recall precision AUPRC
    recall_test=recall_score(true_labels_test, predicted_labels_test)
    precision_test=precision_score(true_labels_test, predicted_labels_test)
    

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
    
    # balanced Accourcy
    balanced_accuracy_test = (class1_accuracy_test + class0_accuracy_test)/2


    #validation 단계에서 Loss가 가장 모델 저장하기 
    if avg_test_loss < best_loss:

        if os.path.exists(best_model_path):
            shutil.rmtree(best_model_path)

        save_dir = f'best_model_epoch_{epoch_i + 1}'
<<<<<<< HEAD
        model_path = os.path.join('../run/saved_models/dataset_load_2_1:1/' + save_dir)
=======
        model_path = os.path.join('run/saved_models/dataset_load_2_1:1/' + save_dir)
>>>>>>> 4 commit
        os.makedirs(model_path, exist_ok=True)
        model.module.save_pretrained(model_path)
        best_loss=avg_test_loss
        best_model_path=model_path

        if os.path.exists(best_model_img):
            os.remove(best_model_img)

        cm = confusion_matrix(true_labels_test, predicted_labels_test)
        class_names = ["Bonafide", "Spoof"]  # 클래스 이름
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        
<<<<<<< HEAD
        img_dir="../run/confusion_matrix/dataset_load_2_1:1"
=======
        img_dir="run/confusion_matrix/dataset_load_2_1:1"
>>>>>>> 4 commit
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
            
        image_path = os.path.join(img_dir, f'confusion_matrix_epoch_{epoch_i + 1}.png')
        plt.savefig(image_path, dpi=300, bbox_inches="tight")
        best_model_img=image_path


    #마지막 epoch에서 모델 저장하기 
    if (epoch_i + 1)==epochs:
        cm = confusion_matrix(true_labels_test, predicted_labels_test)
        class_names = ["Bonafide", "Spoof"]  # 클래스 이름
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        
<<<<<<< HEAD
        img_dir="../run/confusion_matrix/dataset_load_2_1:1"
=======
        img_dir="run/confusion_matrix/dataset_load_2_1:1"
>>>>>>> 4 commit
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        image_path = os.path.join(img_dir, f'confusion_matrix_epoch_{epoch_i + 1}.png')
        plt.savefig(image_path, dpi=300, bbox_inches="tight")


        save_dir=f'epoch_{epoch_i + 1}'
<<<<<<< HEAD
        model_path=os.path.join('../run/saved_models/dataset_load_2_1:1/'+save_dir)
=======
        model_path=os.path.join('run/saved_models/dataset_load_2_1:1/'+save_dir)
>>>>>>> 4 commit
        os.makedirs(model_path, exist_ok=True)
        model.module.save_pretrained(model_path)
        
    #CSV 파일에 저장하기 
    data.append({
        'Epoch': epoch_i,
        'Train Loss': avg_train_loss,
        'Train Accuracy': accuracy_train,
        'Train balanced Accuracy':balanced_accuracy_train,
        'Train Recall':recall_train,
        'Train Precision': precision_train,
        'Train class0 Accuracy':class0_accuracy_train,
        'Train class1 Accuracy':class1_accuracy_train,
        'Validation Loss': avg_test_loss,
        'Validation Accuracy': accuracy_test,
        'validation balanced Accuracy':balanced_accuracy_test,
        'Validation Recall':recall_test,
        'Validation Precision': precision_test,
        'Validation class0 Accuracy':class0_accuracy_test,
        'Validation class1 Accuracy':class1_accuracy_test,
    })

    df = pd.DataFrame(data)
<<<<<<< HEAD
    csv_dir='../run/result_csv'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    df.to_csv('../run/result_csv/dataset_load_2_1:1.csv', index=False)
=======
    csv_dir='run/result_csv'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    df.to_csv('run/result_csv/dataset_load_2_1:1.csv', index=False)
>>>>>>> 4 commit

writer.close()


