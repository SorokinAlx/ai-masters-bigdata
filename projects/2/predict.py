#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load
import pandas as pd

sys.path.append('.')
from model import fields
fields.remove("label")
#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("DONEDOENDOENDOENEODNEOENEDODNEONO")
#logging.info("CURRENT_DIR {}".format(os.getcwd()))

#load the model
model = load("1.joblib")

#fields = """doc_id,hotel_name,hotel_url,street,city,state,country,zip,class,price,
#num_reviews,CLEANLINESS,ROOM,SERVICE,LOCATION,VALUE,COMFORT,overall_ratingsource""".replace("\n",'').split(",")

#read and infere
read_opts=dict(
        sep='\t', names=fields, index_col=0, header=None,
        iterator=True, chunksize=1000
)
for df in pd.read_csv(sys.stdin, **read_opts):
    df = df.replace('\\N', '0')
    pred = model.predict_proba(df.iloc[:,:13])[:,1]
    out = zip(df.index, pred)
    print("\n".join(["{0}\t{1}".format(*i) for i in out]))
