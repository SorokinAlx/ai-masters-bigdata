INSERT OVERWRITE DIRECTORY 'SorokinAlx_hiveout' 
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
SELECT id, case when predicted is null then 0 else predicted end predicted from hw2_pred;
