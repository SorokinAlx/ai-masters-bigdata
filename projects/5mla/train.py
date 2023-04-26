#!/opt/conda/envs/dsenv/bin/python

import os, sys
import logging
import mlflow


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV


#
# Import model definition
#

if __name__ == "__main__":
    #
    # Logging initialization
    #
    logging.basicConfig(level=logging.DEBUG)
    logging.info("CURRENT_DIR {}".format(os.getcwd()))
    logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
    logging.info("ARGS {}".format(sys.argv[1:]))

    #
    # Read script arguments
    #
    try:
        train_path = sys.argv[1] 
        model_param1 = sys.argv[2]
    except:
        logging.critical("Need to pass both train path and l2-reg (C) parameter")
        sys.exit(1)

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
    ('dropper', dropper),
    ('preprocessor', preprocessor),
    ('logregression', LogisticRegression())
    ])


    logging.info(f"TRAIN_PATH {train_path}")
    logging.info(f"C {model_param1}")
    #
    # Read dataset
    #
    #fields = """doc_id,hotel_name,hotel_url,street,city,state,country,zip,class,price,
    #num_reviews,CLEANLINESS,ROOM,SERVICE,LOCATION,VALUE,COMFORT,overall_ratingsource""".replace("\n",'').split(",")

    read_table_opts = dict(sep="\t", names=fields, index_col=0)
    df = pd.read_table(train_path, **read_table_opts)

    #split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:,1:], df.iloc[:,0], test_size=0.33, random_state=42
    )

    #
    # Train the model
    #
    with mlflow.start_run():
        model.set_params(logregression__C=float(model_param1))
        #mlflow.log_param("model_param1", float(model_param1))
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, artifact_path="model")
        y_pred = model.predict(X_test)
        model_score = log_loss(y_test, y_pred)
        mlflow.log_metric("log_loss", model_score)


