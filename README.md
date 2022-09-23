Framework for pytorch transformers
===
*incomplete*
* This repository is a framework for pytorch-based huggingface transformers. It is a code written so that it can be used as it is by modifying the code according to your own dataset and task.
* Also, I created a logging file so you can save your training history.

## 1. Usage
### Git clone
```
git clone https://github.com/kookeej/pytorch_transformers_framework.git
```

### Preprocessing
* Preprocessing your dataset via `preprocessing.py`
* Then, some `pickle` files will be generated in `data/`, including train, val, and test dataloader.
```
python preprocessing.py   --path PATH                 # dataset path
                          --max_length MAX_LENGTH     # token max length (default: 256)
```

### Training code
```
python train.py   --epochs EPOCHS       # epochs (default: 10)
                  --lr LR               # learning rate (default: 1e-4)
                  --mode MODE           # classification or regression (default: classification)
                  --logger LOGGER       # logger name (default: test)
```

### Inference code
```
python inference.py   --path PATH                 # test dataloader path
                      --model_path MODEL_PATH     # saved model path  
```


### An output sample
* terminal
![image](https://user-images.githubusercontent.com/74829786/191977665-4679965d-d5da-4c33-b3b0-c3b7681c7034.png)    

* logging file
![image](https://user-images.githubusercontent.com/74829786/191977721-691832bf-8f61-4a22-a50a-1a4f7904ad3e.png)


***

## 2. Framework Design
