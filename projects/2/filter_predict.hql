add file projects/2/predict.py;
add file projects/2/model.py;
add file projects/2/1.joblib;
INSERT INTO TABLE hw2_pred SELECT TRANSFORM(*) USING "predict.py" AS (id string, predicted double)
from hw2_test WHERE if1>20 and if1<40;
