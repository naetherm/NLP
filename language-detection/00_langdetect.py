# Import submodules
import pandas as pd
import numpy as np
from langdetect import detect, detect_langs
from sklearn import metrics
from tqdm import tqdm
import pycountry

# Helpers
# Conversion table between dataset and langdetect

# Load the dataset
print(">> Loading the dataset ...")
lang_set = pd.read_csv('sentences.csv', sep='\t')
lang_set = lang_set.dropna()
print(">> done.")

print(">> Rearranging everything ...")
X, Y = [], []
for no, ln in enumerate(lang_set.cmn.unique()):
  langs = lang_set.loc[lang_set.cmn == ln]
  if langs.shape[0] < 500:
    continue
  langs = langs.iloc[:500, -1].tolist()
  X.extend(langs)
  Y.extend([ln] * len(langs))
print(">> done.")

# Get the labels -> required for classification_report
labels = np.unique(Y, return_counts=True)[0]

P = []
exception_count = 0
for x, y in tqdm(zip(X, Y)):
    # Get the top language
    try:
        detected = detect(x)
        lang_code = pycountry.languages.get(alpha_2=detected.split('-')[0]).alpha_3.lower()
        P.append(lang_code if lang_code in labels else 'eng')
    except:
        exception_count += 1
        P.append('eng')
      
print(f"Ran into {exception_count} exception of 64500 in total")

print(metrics.classification_report(Y, P, target_names=labels))
