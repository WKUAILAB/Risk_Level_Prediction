create table if not exists data_mining.rc_risklevel_result_v2 (
    ds string, 
    user_id string, 
    order_id string,
    pred_risk int,
    lstm float, 
    gbdt float) 
row format delimited fields terminated by '\x01';
load data local inpath '/data/luyining/result.csv'
overwrite into table data_mining.rc_risklevel_result_v2;

