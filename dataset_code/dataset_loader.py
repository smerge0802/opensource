# Dataset 
import os 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# encodec
from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor

# write
import pickle

#tdqm
from tqdm import tqdm

#ASVspoofDataset
from dataset import ASVspoofDataset

# encodec_model
model_audio = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# ASVspoof_2019_train
train_data = load_dataset("LanceaKing/asvspoof2019", split='train')

# ASVspoof_2019_test
test_data = load_dataset("LanceaKing/asvspoof2019", split='test')

# ASVspoof_2019_validation
validation_data = load_dataset("LanceaKing/asvspoof2019", split='validation')



max_length = 512

#dataset 디렉토리 생성  
dataset_directory = "dataset/dataset_load_2" #디렉토리 생성 
if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)

# train_data
train_dataset = ASVspoofDataset(train_data, max_length)
with open("dataset/dataset_load_2/asvspoof_train_dataset.pkl", "wb") as f:
    pickle.dump(train_dataset, f)

# test_data
test_dataset = ASVspoofDataset(test_data, max_length)
with open("dataset/dataset_load_2/asvspoof_test_dataset.pkl", "wb") as f:
    pickle.dump(test_dataset, f)

# validation_data
validation_dataset = ASVspoofDataset(validation_data, max_length)
with open("dataset/dataset_load_2/asvspoof_validation_dataset.pkl", "wb") as f:
    pickle.dump(validation_dataset, f)
