import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump

train = pd.read_json(sys.argv[2], lines=True)
print(train.head(5))
train = train[["label","words_final"]]
print(train["words_final"].head(5))
for i in range(100):
    train[f"words_{i}"]=train["words_final"].apply(lambda x: x['size'])
print(train.head(5))
log_reg = LogisticRegression()
model = log_reg.fit(train.iloc[:,2:], train.iloc[:,0])
dump(model, sys.argv[4])