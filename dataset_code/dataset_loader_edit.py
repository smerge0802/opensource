import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import datetime
import random

# Encodec
from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor

# Custom dataset
from dataset import ASVspoofDataset

# Pickle
import pickle


# encodec_model
model_audio = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# ASVspoof_2019_train
train_data = load_dataset("LanceaKing/asvspoof2019", split='train')
train_data = train_data.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))

# ASVspoof_2019_test
test_data = load_dataset("LanceaKing/asvspoof2019", split='test')
test_data = test_data.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))

# ASVspoof_2019_validation
validation_data = load_dataset("LanceaKing/asvspoof2019", split='validation')
validation_data = validation_data.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))



# 생성 음성과 진짜 음성 갯수 1:1 
def reduce(dataset_list):
    label_0_samples = [sample for sample in dataset_list if sample["label"] == 0]
    label_1_samples = [sample for sample in dataset_list if sample["label"] == 1]
    num_label_0 = len(label_0_samples)
    print(num_label_0)
    num_label_1 = len(label_1_samples)
    print(num_label_1)

    if num_label_0 < num_label_1:
        sampled_label_1 = random.sample(label_1_samples, num_label_0)
        balanced_dataset = label_0_samples + sampled_label_1
    else:
        sampled_label_0 = random.sample(label_0_samples, num_label_1)
        balanced_dataset = sampled_label_0 + label_1_samples

    dataset = balanced_dataset
    return dataset


#  [2,512] 평균구하는 코드
"""def mean(dataset_list):
    new_tensor_list=[]
    for t in tqdm(dataset_list, desc="Processing Data"):
        t['audio_code']=((t['audio_code'][0]+t['audio_code'][1])//2)
        t['attention_mask']=t['attention_mask'][0]
        t['label']=t['label'][0]
        new_tensor_list.append(t)
        
    return new_tensor_list"""

# [2, 512] 패딩이 된 텐서에서 그냥 concat함    
"""def concat(dataset_list):
    new_tensor_list=[]
    for t in tqdm(dataset_list, desc="Processing Data"):
        t['audio_code']=torch.cat((t['audio_code'][0],t['audio_code'][1]),dim=0)
        t['attention_mask']=torch.cat((t['attention_mask'][0],t['attention_mask'][1]),dim=0)
        t['label']=t['label'][0]
        new_tensor_list.append(t)
    return new_tensor_list"""



#datase 디렉토리 생성  
dataset_directory = "dataset/dataset_load_2_1:1" #디렉토리 생성 
if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)



#train
with open("dataset/dataset_load_2/asvspoof_train_dataset.pkl", "rb") as f: #encodec을 사용해 전처리한 음성 데이터셋 
    train_dataset = pickle.load(f)

train_dataset=reduce(train_dataset)
print(len(train_dataset))

with open("dataset/dataset_load_2_1:1/asvspoof_train_dataset.pkl", "wb") as f: # 생성 음성과 가짜 음성의 비율을 1:1로 만들어서 새로 저장 
    pickle.dump(train_dataset, f)


#test
with open("dataset/dataset_load_2/asvspoof_test_dataset.pkl", "rb") as f:
    test_dataset = pickle.load(f)

test_dataset=reduce(test_dataset)
print(len(test_dataset))

with open("dataset/dataset_load_2_1:1/asvspoof_test_dataset.pkl", "wb") as f:
    pickle.dump(test_dataset, f)


#validation
with open("dataset/dataset_load_2/asvspoof_validation_dataset.pkl", "rb") as f:
    validation_dataset = pickle.load(f)

validation_dataset=reduce(validation_dataset)
print(len(validation_dataset))

with open("dataset/dataset_load_2_1:1/asvspoof_validation_dataset.pkl", "wb") as f:
    pickle.dump(validation_dataset, f)


