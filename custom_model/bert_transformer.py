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
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"

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

with open("dataset/dataset_load_concat3_1:1/asvspoof_train_dataset.pkl", "rb") as f:
    train_dataset = pickle.load(f)

with open("dataset/dataset_load_concat3_1:1/asvspoof_test_dataset.pkl", "rb") as f:
    test_dataset = pickle.load(f)

with open("dataset/dataset_load_concat3_1:1/asvspoof_validation_dataset.pkl", "rb") as f:
    validation_dataset = pickle.load(f)

# dataset load
Batch = 32
trainloader = DataLoader(train_dataset, batch_size=Batch, shuffle=True)
validationloader = DataLoader(validation_dataset, batch_size=Batch, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=Batch, shuffle=True)



"""#tranformer_classification 
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, num_classes, num_heads=4, hidden_size=256, num_layers=3): # num head -> 멀티헤드 어텐션을 몇개 헤드가 있냐 레이어는 레이어 갯수 
        super(TransformerClassifier, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)

        # Transformer layers
        transformer_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
        )
        self.transformer = nn.TransformerEncoder(transformer_layers, num_layers)
        # Output layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x.to(torch.long))

        x = self.transformer(x)
        # Global average pooling
        x = x.mean(dim=1)
        x = self.fc(x)
        return x"""
        
    
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()

        # Multi-Head Self Attention Layer
        self.self_attention = nn.MultiheadAttention(input_size, num_heads)

        # Feedforward Layer
        self.feedforward = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.layer_norm2 = nn.LayerNorm(input_size)

        # Fully Connected Layer for Classification
        self.fc = nn.Linear(input_size, 1)
        
        #function 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        att_output, _ = self.self_attention(x, x, x)

        x = self.layer_norm1(x + att_output)

        ff_output = self.feedforward(x)

        x = self.layer_norm2(x + ff_output)

        # Global Average Pooling
        x = torch.mean(x, dim=1)

        # Fully Connected Layer for Classification
        x = self.fc(x)
        
        x = self.sigmoid(x)
        
        return x

input_size = 768  # 임베딩의 크기 혹은 단어 집합의 크기에 따라 다름
num_classes = 2  # 분류할 클래스의 개수에 따라 다름


class KSM(ModuleUtilsMixin, nn.Module):
    def __init__(self, embed, encoder, classifier, config):
        super(KSM, self).__init__()
        self.a = embed
        self.b = encoder
        self.c = classifier
        self.config = config

    def forward(self, x, asd, mask):
        input_shape = x.size()
        x = self.a(x)
        mask = self.get_extended_attention_mask(mask, input_shape)
        x = self.b(x, attention_mask=mask)
        x = x[0]

        x = self.c(x)
        return x


# cuda
device_ids = [0, 1, 2]
device = torch.device('cuda')
model_bert = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
embed = model_bert.bert.embeddings
encoder = model_bert.bert.encoder
config = BertConfig.from_pretrained('bert-base-multilingual-cased')

num_classes = 2  # Change this based on the number of classes in your classification task
input_size = 768
classifier = TransformerClassifier(input_size, hidden_size=768, num_heads=8, num_layers=6, num_classes=num_classes)
model =  KSM(embed, encoder, classifier, config)


#print(model)

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
log_dir = 'logs/dataset_load_2_transformer'
writer = SummaryWriter(log_dir=log_dir)
model.zero_grad()

# loss funtion
loss_fn = nn.BCEWithLogitsLoss()
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

        outputs = model(audio_code, None, attention_mask)
        

        # accuracy
        logits = outputs.to('cpu')
        
        logits_label = logits.detach().cpu().numpy()
        
        audio_label = labels.to('cpu').numpy()
        
        true_label=audio_label.flatten()
        pred_labels = (logits_label > 0.5).astype(int).flatten() 
        
        true_labels_train.extend(true_label)
        predicted_labels_train.extend(pred_labels)

        # Loss
        labels = labels.float()
        loss=loss_fn(logits,labels)
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

        """
       # Loss
        logits_flattened = logits.view(-1, 2)
        labels_flattened = labels.squeeze().view(-1)
        loss = loss_fnt(logits_flattened, labels_flattened)
        total_loss += loss
        """

        # accuracy
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
        
        model_path='saved_models/dataset_load_2_transformer/'
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
        image_path = os.path.join('confusion_matrix/dataset_load_2_transformer',
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
        image_path = os.path.join('confusion_matrix/dataset_load_2_transformer',
                                  f'confusion_matrix_epoch_{epoch_i + 1}.png')
        plt.savefig(image_path, dpi=300, bbox_inches="tight")

        save_dir = f'epoch_{epoch_i + 1}.pth'

        torch_model_path=os.path.join('saved_models/dataset_load_2_transformer/'+save_dir)
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
    df.to_csv('result_csv/dataset_load_2_transformer.csv', index=False)

writer.close()
