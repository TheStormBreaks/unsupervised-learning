#main
import numpy as np
import pandas as pd
import os

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
current_path = os.getcwd()
file = '\\datasets\\credit_card_data\\correlationMatrix.csv'
data = pd.read_csv(current_path + file)

print(data.head())
#ata.describe()
