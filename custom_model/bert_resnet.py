# Bert
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.cuda
import torch.nn as nn
# from torchvision.models import resnet18
from torch.nn import DataParallel
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch.nn.functional as F
import tensorflow as tf
from transformers import AutoModel
from transformers.modeling_utils import ModuleUtilsMixin

# encodec
from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor

# tensorboard
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

# setting
import pickle
import datetime
import random
from tqdm import tqdm

# calculate
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, precision_score, \
    recall_score, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

# cuda
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6'


# encodec_model
model_audio = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# transform train dataset to load
class ASVspoofDataset(Dataset):
    def __init__(self, data, max_length):
        self.audio_code_list = []
        self.attention_mask_list = []
        self.labels_list = []

        for code in tqdm(data, desc="Processing Data"):
            audio_array = code['audio']['array']
            audio = processor(raw_audio=audio_array, sampling_rate=processor.sampling_rate, return_tensors="pt")
            audio_codes = model_audio(audio["input_values"], audio["padding_mask"]).audio_codes
            audio_tensor = audio_codes.squeeze()

            tensor_shape = audio_tensor.size()

            if tensor_shape[1] > max_length:
                audio_tensor = audio_tensor[:, :max_length]
                audio_tensor_classification = audio_tensor[0]
                self.audio_code_list.append(audio_tensor_classification)

                encoded_list = audio_tensor.numpy()
                attention_mask = [[1] * len(sequence) + [0] * (max_length - len(sequence)) for sequence in encoded_list]
                attention_tensor = torch.tensor(attention_mask)
                attention_tensor = attention_tensor[0]
                self.attention_mask_list.append(attention_tensor)
            else:
                # 먼저 attention mask를 만들어주고 패딩하기기
                encoded_list = audio_tensor.numpy()
                attention_mask = [[1] * len(sequence) + [0] * (max_length - len(sequence)) for sequence in encoded_list]
                attention_tensor = torch.tensor(attention_mask)
                attention_tensor = attention_tensor[0]

                self.attention_mask_list.append(attention_tensor)

                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, max_length - audio_tensor.size(1)))
                audio_tensor = audio_tensor[0]

                self.audio_code_list.append(audio_tensor)

            audio_label = code['key']
            audio_label = torch.tensor(audio_label).unsqueeze(0)
            # audio_label = torch.cat([audio_label.unsqueeze(0), audio_label.unsqueeze(0)])
            self.labels_list.append(audio_label)

    def __len__(self):
        return len(self.audio_code_list)

    def __getitem__(self, idx):
        return {
            "audio_code": self.audio_code_list[idx],
            "attention_mask": self.attention_mask_list[idx],
            "label": self.labels_list[idx],
        }

with open("dataset/dataset_load_2_1:1/asvspoof_train_dataset.pkl", "rb") as f:
    train_dataset = pickle.load(f)

with open("dataset/dataset_load_2_1:1/asvspoof_test_dataset.pkl", "rb") as f:
    test_dataset = pickle.load(f)

with open("dataset/dataset_load_2_1:1/asvspoof_validation_dataset.pkl", "rb") as f:
    validation_dataset = pickle.load(f)

# dataset load
Batch = 32
trainloader = DataLoader(train_dataset, batch_size=Batch, shuffle=True)
validationloader = DataLoader(validation_dataset, batch_size=Batch, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=Batch, shuffle=True)



class CustomResNetForBinaryClassification(nn.Module):
    def __init__(self, input_size=768):
        super(CustomResNetForBinaryClassification, self).__init__()

        """# ResNet의 기본 구조를 정의합니다.
        self.resnet = nn.Sequential(
            nn.Conv2d(768, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            self._make_layer(64, 64, 2), 
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )"""
        self.resnet = nn.Sequential(
            nn.Conv2d(input_size, 128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            self._make_layer(128, 128, 2), 
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
            self._make_layer(512, 1024, 2, stride=2),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        # 이진 분류를 위한 Fully Connected 레이어
        #self.fc = nn.Linear(512, 1)
        self.fc = nn.Linear(1024, 1)
        
        # 시그모이드 활성화 함수
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        B,N,C = x.shape
        x = x.view(B, 23, 23, 768) 
        x = x.permute(0,3,1,2)
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# cuda
class KSM(ModuleUtilsMixin,nn.Module):
    def __init__(self,embed,encoder,classifier, config):
        super(KSM, self).__init__()
        self.a = embed
        self.b = encoder
        self.inter = nn.Linear(512, 529)
        self.c = classifier
        self.config = config

    def forward(self, x,asd,mask):
        
        input_shape = x.size()
        x = self.a(x)
        mask = self.get_extended_attention_mask(mask, input_shape)
        x = self.b(x,attention_mask=mask)
        x = x[0]
        x = x.permute(0,2,1)
        x = self.inter(x)
        x = x.permute(0,2,1)
        x = self.c(x)
        return x
    
    
device_ids = [0, 1, 2]
device = torch.device('cuda')
model_bert = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
embed = model_bert.bert.embeddings
encoder = model_bert.bert.encoder
config = config = BertConfig.from_pretrained('bert-base-multilingual-cased')
classifier = CustomResNetForBinaryClassification(input_size=768)
model =  KSM(embed, encoder, classifier, config)
print(model)
quit()
model = DataParallel(model, device_ids=device_ids)
model.to(device)

# optimizer
optimizer = Adam(model.parameters(), lr=1e-7)
epochs = 130                                                                                                                                                                                 

total_steps = len(trainloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# accuracy
best_val_accuracy = 0.0

# tensorboard
log_dir = 'logs/dataset_load_2_1:1_crossEntropy'
writer = SummaryWriter(log_dir=log_dir)
model.zero_grad()

# loss funtion
loss_fn = nn.BCEWithLogitsLoss()
#loss_fn = nn.CrossEntropyLoss()


best_loss = 100
best_model_path = ''
best_model_img = ''
torch_model_path=''
# csv 파일에 저장
data = []

for epoch_i in range(0, epochs):
    total_loss = 0
    model.train()
    true_labels_train = []
    predicted_labels_train = []

    # train
    for batch in tqdm(trainloader, desc="Processing Data"):
        audio_code = batch["audio_code"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to('cpu')
        
        
        outputs = model(audio_code,None,attention_mask)
        
        
        # accuracy_label
        logits = outputs.to('cpu')
        
        logits_label = logits.detach().cpu().numpy()
        
        audio_label = labels.to('cpu').numpy()
        
        true_label=audio_label.flatten()
        pred_labels = (logits_label > 0.5).astype(int).flatten() 
        
        
        true_labels_train.extend(true_label)
        predicted_labels_train.extend(pred_labels)

        print(logits)
        print(labels)
        
        # Loss
        labels = labels.float()
        loss=loss_fn(logits,labels)
        print(loss)
        total_loss += loss.mean()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
   

    avg_train_loss = total_loss / len(trainloader)
    print(f'avg_train_loss: {avg_train_loss}')
    writer.add_scalar("Train Loss", avg_train_loss, epoch_i)

    # 정확도
    accuracy_train = accuracy_score(true_labels_train, predicted_labels_train)
    writer.add_scalar("Train Accuracy", accuracy_train, epoch_i)
    
    # 균형정확도
    balanced_accuracy_train = balanced_accuracy_score(true_labels_train, predicted_labels_train)
    print(f'accuracy_train: {accuracy_train}')

    # recall precision AUPRC
    recall_train = recall_score(true_labels_train, predicted_labels_train)
    precision_train = precision_score(true_labels_train, predicted_labels_train)
    
    # 클래스 0에 대한 정확도
    class_0_indices = [i for i in range(len(predicted_labels_train)) if predicted_labels_train[i] == 0]
    true_label_class_0_train = [true_labels_train[i] for i in class_0_indices]
    predicted_label_class_0_train = [predicted_labels_train[i] for i in class_0_indices]
    class0_accuracy_train = accuracy_score(true_label_class_0_train, predicted_label_class_0_train)

    # 클래스 1에 대한 정확도
    class_1_indices = [i for i in range(len(predicted_labels_train)) if predicted_labels_train[i] == 1]
    true_label_class_1_train = [true_labels_train[i] for i in class_1_indices]
    predicted_label_class_1_train = [predicted_labels_train[i] for i in class_1_indices]
    class1_accuracy_train = accuracy_score(true_label_class_1_train, predicted_label_class_1_train)

    # test
    model.eval()
    total_loss = 0
    true_labels_test = []
    predicted_labels_test = []

    for batch in tqdm(validationloader, desc="Processing Data"):
        audio_code = batch["audio_code"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to('cpu')

        

        # 모델에 넣기
        with torch.no_grad():
            outputs = model(audio_code, None, attention_mask)

        logits = outputs

        # accuracy_label
        logits = outputs.to('cpu')
        logits_label = logits.detach().cpu().numpy()
        
        audio_label = labels.to('cpu').numpy()
        true_label=audio_label.flatten()
        pred_labels = (logits_label > 0.5).astype(int).flatten() 
        
        true_labels_test.extend(true_label)
        predicted_labels_test.extend(pred_labels)
        
        # Loss
        labels = labels.float()
        loss=loss_fn(logits,labels)
        total_loss += loss.mean()
    
        
    

    # Loss
    avg_test_loss = total_loss / len(testloader)
    print(f'avg_test_loss: {avg_test_loss}')
    writer.add_scalar("Validtaion Loss", avg_test_loss, epoch_i)

    # 정확도
    accuracy_test = accuracy_score(true_labels_test, predicted_labels_test)
    print(f'accuracy_test: {accuracy_test}')
    writer.add_scalar("Validation Accuracy", accuracy_test, epoch_i)

    # 균형 정확도
    balanced_accuracy_test = balanced_accuracy_score(true_labels_test, predicted_labels_test)

    # recall precision AUPRC
    recall_test = recall_score(true_labels_test, predicted_labels_test)
    precision_test = precision_score(true_labels_test, predicted_labels_test)

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

    if avg_test_loss < best_loss:

        save_dir = f'best_model_epoch_{epoch_i + 1}.pth'
        
        if os.path.exists(torch_model_path):
            os.remove(torch_model_path)
        
        model_path='saved_models/dataset_load_2_layer/'
        os.makedirs(model_path, exist_ok=True)
        torch_model_path=os.path.join(model_path+save_dir)
        torch.save(model.state_dict(), torch_model_path)
        
        best_loss = avg_test_loss

        if os.path.exists(best_model_img):
            os.remove(best_model_img)

        cm = confusion_matrix(true_labels_test, predicted_labels_test)
        class_names = ["Bonafide", "Spoof"]  # 클래스 이름
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        image_path = os.path.join('confusion_matrix/dataset_load_2_layer',
                                  f'confusion_matrix_epoch_{epoch_i + 1}.png')
        plt.savefig(image_path, dpi=300, bbox_inches="tight")
        best_model_img = image_path

    if (epoch_i + 1) == epochs:
        cm = confusion_matrix(true_labels_test, predicted_labels_test)
        class_names = ["Bonafide", "Spoof"]  # 클래스 이름
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        image_path = os.path.join('confusion_matrix/dataset_load_2_layer',
                                  f'confusion_matrix_epoch_{epoch_i + 1}.png')
        plt.savefig(image_path, dpi=300, bbox_inches="tight")

        save_dir = f'epoch_{epoch_i + 1}.pth'

        torch_model_path=os.path.join('saved_models/dataset_load_2_layer/'+save_dir)
        torch.save(model.state_dict(), torch_model_path)
        

    data.append({
        'Epoch': epoch_i,
        'Train Loss': avg_train_loss,
        'Train Accuracy': accuracy_train,
        'Train balanced Accuracy': balanced_accuracy_train,
        'Train Recall': recall_train,
        'Train Precision': precision_train,
        'Train class0 Accuracy': class0_accuracy_train,
        'Train class1 Accuracy': class1_accuracy_train,
        'Validation Loss': avg_test_loss,
        'Validation Accuracy': accuracy_test,
        'validation balanced Accuracy': balanced_accuracy_test,
        'Validation Recall': recall_test,
        'Validation Precision': precision_test,
        'Validation class0 Accuracy': class0_accuracy_test,
        'Validation class1 Accuracy': class1_accuracy_test,
    })

    df = pd.DataFrame(data)
    df.to_csv('result_csv/dataset_load_2_layer.csv', index=False)

writer.close()
