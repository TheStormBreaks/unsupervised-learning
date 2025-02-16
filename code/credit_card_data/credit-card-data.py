#main
import numpy as np
import pandas as pd
import os
import gdown

#data vis
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import matplotlib as plt

#data prep
from sklearn import preprocessing as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

#algos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

#read
#current_path = os.getcwd()
#file = 'https://drive.google.com/file/d/1NfLMXdIlGG5u0TjfuImxVsBK2osqcaI8/view?usp=sharing'
#data = pd.read_csv(current_path + file)


#file_id = "1NfLMXdIlGG5u0TjfuImxVsBK2osqcaI8"
#url = f"https://drive.google.com/uc?export=download&id={file_id}"
#data = pd.read_csv(url)


file_id = "1NfLMXdIlGG5u0TjfuImxVsBK2osqcaI8"  # Replace with your actual file ID
url = f"https://drive.google.com/uc?export=download&id={file_id}"
output = "creditcard.csv"
gdown.download(url, output, quiet=False)

data = pd.read_csv(output)
print(data.head())
data.describe()

