from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime
import os





base_dir = '{{ dag_run.conf["base_dir"] if dag_run else "" }}'

with DAG(
    dag_id='SorokinAlx_dag',
    schedule_interval=None,
    start_date = datetime(2023, 5, 1),
    catchup=False
    ) as dag:
    
    feature_eng_task = SparkSubmitOperator(
       task_id = "feature_eng_train_task",
       spark_binary = "/usr/bin/spark3-submit",
       application=f"{base_dir}/feature_eng.py",
       application_args = ["--path-in", '/datasets/amazon/all_reviews_5_core_train_extra_small_sentiment.json', "--path-out",  'SorokinAlx_train_out'],
       env_vars={"PYSPARK_PYTHON": '/opt/conda/envs/dsenv/bin/python'}
    )
    
    train_download_task = BashOperator(
       task_id='download_train_task',
       bash_command=f"hdfs dfs -getmerge SorokinAlx_train_out {os.path.join(base_dir, 'SorokinAlx_train_out_local')}"
    )
    
    train_task = BashOperator(
       task_id='train_task',
       bash_command=f'{"/opt/conda/envs/dsenv/bin/python"} {os.path.join(base_dir, "train.py")} --train-in {os.path.join(base_dir, "SorokinAlx_train_out_local")} --sklearn-model-out {os.path.join(base_dir, "6.joblib")}',
    )
    
    model_sensor = FileSensor(task_id= "model_sensor", filepath= f"{base_dir}/6.joblib")
    
    feature_eng_task_test = SparkSubmitOperator(
       task_id="feature_eng_test_task",
       spark_binary="/usr/bin/spark3-submit",
       application=f"{base_dir}/feature_eng.py",
       application_args = ['--path-in', '/datasets/amazon/all_reviews_5_core_test_extra_small_features.json', '--path-out', 'SorokinAlx_test_out'],
       env_vars={"PYSPARK_PYTHON": '/opt/conda/envs/dsenv/bin/python'}
    )

    predict_task = SparkSubmitOperator(
       task_id="predict_task",
       spark_binary="/usr/bin/spark3-submit",
       application=f"{base_dir}/predict.py",
       application_args = ['--test-in', 'SorokinAlx_test_out', '--pred-out', 'SorokinAlx_hw6_prediction', '--sklearn-model-in', f"{base_dir}/6.joblib"],
       env_vars={"PYSPARK_PYTHON": '/opt/conda/envs/dsenv/bin/python'}
    ) 
    

feature_eng_task >>  train_download_task >> train_task >> model_sensor >> feature_eng_task_test >> predict_task