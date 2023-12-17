import librosa
from transformers import AutoProcessor, EncodecModel, BertForSequenceClassification
import torch
import numpy as np
import sys

def main(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=24000)

    # Load Encodec model and processor
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

    # Process audio input
    inputs = processor(raw_audio=y, sampling_rate=sr, return_tensors="pt")

    # Encode audio
    audio_tensor = model.encode(inputs["input_values"], inputs["padding_mask"]).audio_codes.squeeze()

    # Prepare attention mask and pad audio if needed
    tensor_shape = audio_tensor.size()
    max_length = 512

    if tensor_shape[1] > max_length:
        audio_tensor = audio_tensor[:, :max_length]
        audio_code = audio_tensor[1]
        encoded_list = audio_tensor.numpy()
        attention_mask = [[1] * len(sequence) + [0] * (max_length - len(sequence)) for sequence in encoded_list]
        attention_tensor = torch.tensor(attention_mask)
        attention_mask = attention_tensor[0]
    else:
        encoded_list = audio_tensor.numpy()
        attention_mask = [[1] * len(sequence) + [0] * (max_length - len(sequence)) for sequence in encoded_list]
        attention_mask = torch.tensor(attention_mask)
        attention_mask = attention_mask[0]
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, max_length - audio_tensor.size(1)))
        audio_code = audio_tensor[1]

    # Load BertForSequenceClassification model
    clf_model = BertForSequenceClassification.from_pretrained('run/saved_models/dataset_load_2_1:1/best_model_epoch_77', num_labels=2)

    # Make prediction
    outputs = clf_model(audio_code.unsqueeze(0), token_type_ids=None, attention_mask=attention_mask.unsqueeze(0))
    pred_labels = np.argmax(outputs.logits.detach().numpy(), axis=1).flatten()
    
    # Print prediction result
    if pred_labels == 1:
        print("Prediction: Spoof")
    else:
        print("Prediction: Bonafide")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inferen.py /path/to/audio/file")
        sys.exit(1)

    audio_path = sys.argv[1]
    main(audio_path)
