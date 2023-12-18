 # Voice Spoof Detection
The feature extraction model we propose is as follows: By utilizing the intermediate layer of the codec model Encodec, we represent speech as positive integer values. The preprocessed speech is then used as the input for BERT to extract features from the speech.


![1](https://github.com/smerge0802/opensource/assets/149349542/e2a94413-316b-4049-af36-d4f7c6383182)



We conducted experiments using the proposed feature extraction model as follows. Utilizing each layer and leveraging the input values from each layer, we carried out various experiments.


![2](https://github.com/smerge0802/opensource/assets/149349542/4983306d-ad95-4e89-88cd-7705aee6b2f1)



The experimental results are as follows:


![3](https://github.com/smerge0802/opensource/assets/149349542/4e77cef9-ed54-43d9-b752-9b798840a0ff)

## The open-source tools and APIs used.
### model
- <https://huggingface.co/docs/transformers/model_doc/bert>
- <https://huggingface.co/docs/transformers/model_doc/encodec>

### dataset
- <https://huggingface.co/datasets/LanceaKing/asvspoof2019>

## The experimental environment is as follows


- The experimental environment.
  
```
system_info.txt
```


- Python 3.8 was used for the experiments.
 

- Other environmental configurations are as follows:

```
pip install -r requirement.txt
```


# Data preprocessing.
- Using Encodec, the speech is preprocessed and stored as positive integer values.
```
python dataset_code/dataset_loader.py
```


- The real-to-fake ratio of the preprocessed speech, represented as positive integer values, is adjusted to 1:1.
```
python dataset_code/dataset_loader_edit.py
```


>If using different data, you need to modify the paths in the code accordingly.


# Train, Test, Inference
```
python model_code/bert_train.py

python model_code/bert_test.py

python inference.py
```

