import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump

train = pd.read_json(sys.argv[2], lines=True)
print(train.head(5))
words = []
for el in train['words_final']:
    l = []
    if len(el['values']) == 100:
        words.append(el['values'])
    else:
        for i in range(100):
            if i in el['indices']:
                l.append(el['values'][el['indices'].index(i)])
            else:
                l.append(0)
        words.append(l)
df = pd.DataFrame(words)
log_reg = LogisticRegression()
model = log_reg.fit(df, train["label"])
dump(model, sys.argv[4])