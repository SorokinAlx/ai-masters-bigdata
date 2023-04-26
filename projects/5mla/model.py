from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

#
# Dataset fields
#
numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)] + ["day_number"]

# without categorical features
fields = ["id", "label"] + numeric_features + categorical_features

#
# Model pipeline
#

class ColumnDropperTransformer():
    def __init__(self, categorical_features):
        self.columns=categorical_features
    def transform(self,X,y=None):
        return X.drop(self.columns,axis=1)
    def fit(self, X, y=None):
        return self 

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
    ]
)
dropper = ColumnDropperTransformer(categorical_features)
# Now we have a full prediction pipeline.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('dropper', dropper),
    ('logregression', LogisticRegression())
])

