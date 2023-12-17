# Dataset 
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


# encodec_model
model_audio = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")


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
                audio_tensor_classification=audio_tensor[1]
                self.audio_code_list.append(audio_tensor_classification)

                encoded_list = audio_tensor.numpy()
                attention_mask = [[1] * len(sequence) + [0] * (max_length - len(sequence)) for sequence in encoded_list]
                attention_tensor = torch.tensor(attention_mask)
                attention_tensor=attention_tensor[0]
                self.attention_mask_list.append(attention_tensor)
            else:
                #먼저 attention mask를 만들어주고 패딩하기기
                encoded_list = audio_tensor.numpy()
                attention_mask = [[1] * len(sequence) + [0] * (max_length - len(sequence)) for sequence in encoded_list]
                attention_tensor = torch.tensor(attention_mask)
                attention_tensor=attention_tensor[0]

                self.attention_mask_list.append(attention_tensor)

                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, max_length - audio_tensor.size(1)))
                audio_tensor=audio_tensor[1]
        
                self.audio_code_list.append(audio_tensor)


            audio_label = code['key']
            audio_label = torch.tensor(audio_label).unsqueeze(0)
            self.labels_list.append(audio_label)

    def __len__(self):
        return len(self.audio_code_list)

    def __getitem__(self, idx):
        return {
            "audio_code": self.audio_code_list[idx],
            "attention_mask": self.attention_mask_list[idx],
            "label": self.labels_list[idx],
        }


