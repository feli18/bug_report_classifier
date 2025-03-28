
# Performance Bug Classification Tool 

This project implements an automated classification system for performance-related software bug reports across major deep learning frameworks. It compares traditional machine learning models (Naïve Bayes, SVM) with pre-trained language models (ALBERT, ALBERT+TF-IDF fusion) to explore trade-offs in accuracy, robustness, and computational cost.

##  Project Overview

In large open-source projects such as TensorFlow, PyTorch, and MXNet, performance-related bugs (PRBs) often have low frequency but high maintenance cost. This tool leverages modern NLP techniques to automatically identify PRBs from textual bug reports, improving triage efficiency and supporting intelligent software maintenance.

##  Folder Structure

```
.
├── albert_classifier_repeat.py   # ALBERT with classification head (final model)
├── albert_idf.py                 # ALBERT + TF-IDF fusion model
├── nb_idf.py                     # Naïve Bayes + TF-IDF (baseline)
├── svm.py                        # SVM + TF-IDF
├── data/
│   └── caffe.csv  
│   └── incubator-mxnet.csv  
│   └── keras.csv  
│   └── pytorch.csv  
│   └── tensorflow.csv                     
├── results/
│   └── *.csv                     # Saved experimental results
├── replication.pdf               # Reproduction instructions
├── requirements.pdf             # Python dependencies
└── manual.pdf                   # User manual
```

##  Models Implemented

| Model              | Feature Encoding     | Notes |
|--------------------|----------------------|-------|
| Naïve Bayes        | TF-IDF               | Fast, simple baseline |
| SVM                | TF-IDF               | High performance with low cost |
| ALBERT             | Pretrained embedding | Best performance, GPU recommended |
| ALBERT + TF-IDF    | Fusion embedding     | Combines sparse + semantic features |

## 📊 Evaluation Metrics

Each model is evaluated on 5 datasets with:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **AUC**

10 repeated runs were performed per model for statistical robustness.

Statistical significance testing (Friedman + Nemenyi) was applied to verify differences between models.

## Datasets

The tool uses manually labeled bug reports from the following deep learning projects:

- TensorFlow
- PyTorch
- Keras
- MXNet
- Caffe

Each sample includes a title, body, and binary class label (1 = performance-related, 0 = not).

## ▶️ How to Run

You can run any of the models individually:

```bash
python nb_idf.py
python svm.py
python albert_classifier_repeat.py
python albert_idf.py
```

Ensure your dataset path is correctly set in each script.

## 📦 Dependencies

Install dependencies from `requirements.pdf` or use:

```bash
pip install -r requirements.txt  # if you convert requirements.pdf to .txt
```

Ensure that your environment has access to:
- Python 3.8+
- PyTorch
- Transformers
- Scikit-learn
- NLTK
- Pandas / NumPy

## Reproduction

Refer to [`replication.pdf`](./replication.pdf) for complete instructions to replicate all experimental results, including model training, evaluation, and statistical testing.

## Manual

See [`manual.pdf`](./manual.pdf) for a detailed usage guide of the tool, including optional arguments, expected inputs/outputs, and tips for adapting the tool to new datasets.

## Citation

If you use this tool in your work, please consider citing the original authors of ALBERT and related resources used in this project.

