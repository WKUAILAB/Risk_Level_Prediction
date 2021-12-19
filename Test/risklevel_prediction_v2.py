# -*- coding:utf-8 -*-
import sklearn 

import warnings
warnings.filterwarnings('ignore')
import os,sys
print('sys path:',sys.path)
import joblib

from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd

from utils import fetch_df, actions_to_indices, pretrained_embedding_layer, attention_3d_block, RiskLevelPredict, make_model, read_action_vecs, time_scalar, convert_to_one_hot, learning_rate_010_decay_power_0995, evaluate_recall
    

### check working envs
work_dir = '.'
# work_dir = 'risklevelV2'
# work_dir = '/home/jovyan/work/risklevel-v2'
print('current working directory:')
print(os.getcwd())
print(os.listdir())

### set other paths
emb_fn = 'action_page_fasttext.dict'
model_fn = 'bidirectional_attention_lstm'
emb_dir = os.path.join(work_dir,'data',emb_fn)
lstm_model_dir = os.path.join(work_dir,'model',model_fn)
lgb_model_dir = os.path.join(work_dir,'model','lgb.pkl')
print('emb_dir:',emb_dir)
print('lstm_model_dir:',lstm_model_dir)

cols = ['ds', 'user_id', 'order_id', 'reg_days', 'platform', 'usertype', 'mobil_prefix3', 'mobile_prefix5', 'len_sequence', 'cnt_pay', 'max_time_diff', 'min_time_diff', 'avg_time_diff', 'std_time_diff', 'cnt_src', 'device_ios', 'device_android', 'device_wap', 'device_web', 'device_app', 'device_mini', 'cnt_login', 'is_bk_log', 'is_wzp_log', 'is_dc_log', 'cnt_item', 'cnt_cheap_item', 'cnt_lyl_item', 'roi', 'avg_roi', 'is_gift_inclued', 'is_virtual_inclued', 'actions', 'times']
data = fetch_df('data_mining','rc_risklevel_features_v3', cols = cols)

action_sequences = pd.DataFrame.to_numpy(data['actions']) 
X = []
index = 0
for index in range(len(action_sequences)):
    temp_action_sequence = action_sequences[index]
    X.append(temp_action_sequence.strip().split(","))

time_sequences = pd.DataFrame.to_numpy(data['times']) 
T = []
index = 0
for index in range(len(time_sequences)):
    temp_time_sequence = time_sequences[index]
    T.append(list(map(np.int64, temp_time_sequence.strip().split(","))))
    
X = np.asarray(X)                                               # array of action_sequences
T = np.asarray(T)                                               # array of time_sequences
# Y = pd.DataFrame.to_numpy(data['has_risk'], dtype = 'int64')    # has_risk（categorical）

t_scalar = [list(map(time_scalar,i)) for i in T] # time scaling
maxLen = len(max(X, key=len))
# Y_indices = Y
# 读取单个动作的 embedding，作数值索引
action_to_index, index_to_action, action_to_vec_map = read_action_vecs(emb_dir)
# 把动作转为数值索引
X_indices = actions_to_indices(X, action_to_index, maxLen)
# 反过来，最后的动作放在最后面
X_indices = np.array([i[::-1] for i in X_indices])
T_indices = np.array([[-1]*(maxLen-len(i))+i[::-1] for i in t_scalar])
T_indices = T_indices.reshape(T_indices.shape[0], T_indices.shape[1], 1)

print('loading lstm model...')
lstm_model = load_model(lstm_model_dir)
lstm_pred = lstm_model.predict([X_indices,T_indices], batch_size = 64)

#########GBDT
print('loading lgb model...')
lgb_model = joblib.load(lgb_model_dir)

feature_columns = ['len_sequence', 'cnt_pay', 'max_time_diff', 'min_time_diff', 'avg_time_diff', 'std_time_diff', 'cnt_src', 'device_ios', 'device_android', 'device_wap', 'device_web', 'device_app', 'device_mini', 'cnt_login', 'is_bk_log', 'is_wzp_log', 'is_dc_log', 'cnt_item', 'cnt_cheap_item', 'cnt_lyl_item', 'roi', 'avg_roi', 'is_gift_inclued','is_virtual_inclued']
feature_columns.append('lstm')
data['lstm'] = lstm_pred
data[feature_columns] = data[feature_columns].astype(float)

data['lgb'] = lgb_model.predict_proba(data[feature_columns])[:,1]
data['pred_risk'] = np.where(data.lgb>=0.5,1,0)

cols = ['ds', 'user_id', 'order_id','pred_risk', 'lstm', 'lgb']

# result_dir = os.path.join(work_dir,'data','result.csv')
result_dir = os.path.join('/data/luyining/result.csv')#写到公共目录上检查数据
data[cols].to_csv(result_dir,index=False,header=False,sep='\x01',line_terminator='\n',float_format='%.2f')

os.system("hive -f {}".format(os.path.join(work_dir,'update_hivedb.sql')))
