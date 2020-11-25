# ASM x64 Function classifier

Homework for Machine Learning course @ Sapienza University of Rome.
Create a model for classifying assembly x64 functions in 4 function categories:
* Encryption
* Sorting
* Math
* String

In this project I built two model:
1. SVM Classifier
2. Decision Tree Classifier

The second one seems to perform better, so I suggest to use it.

## How to use it

First check requirements:

```bash
pip install -r requirements
```

then to train and test the model just run the `main.py` file:

```bash
python3 main.py
```
