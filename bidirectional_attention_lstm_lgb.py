import numpy as np
import pandas as pd
import os
from joblib import dump

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, recall_score, precision_recall_fscore_support
from sklearn.ensemble import GradientBoostingClassifier

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Activation, LSTM, concatenate, Reshape, Permute, Lambda, RepeatVector, Multiply
from tensorflow.keras.layers import Embedding, Bidirectional
from tensorflow.keras.initializers import Constant
import tensorflow.keras.backend as K
import lightgbm as lgb
from lightgbm import LGBMClassifier

from utils import fetch_df, actions_to_indices, pretrained_embedding_layer, attention_3d_block, RiskLevelPredict, make_model, read_action_vecs, time_scalar, convert_to_one_hot, learning_rate_010_decay_power_0995, evaluate_recall
    
cur_dir = '.'
print('current working directory:')
print(os.getcwd())
print(os.listdir())
emb_fn = 'action_page_fasttext.dict'
emb_dir = os.path.join(cur_dir,'data',emb_fn)
    
model_fn = 'attention_lstm_3'
model_dir = os.path.join('/data/luyining','models',model_fn)
    
cols = ['has_risk', 'ds', 'user_id', 'order_id', 'reg_days', 'platform', 'usertype', 'mobil_prefix3', 'mobile_prefix5', 'len_sequence', 'cnt_pay', 'max_time_diff', 'min_time_diff', 'avg_time_diff', 'std_time_diff', 'cnt_src', 'device_ios', 'device_android', 'device_wap', 'device_web', 'device_app', 'device_mini', 'cnt_login', 'is_bk_log', 'is_wzp_log', 'is_dc_log', 'cnt_item', 'cnt_cheap_item', 'cnt_lyl_item', 'roi', 'avg_roi', 'is_gift_inclued', 'is_virtual_inclued', 'actions', 'times']
data = fetch_df('temp','rc_risklevel_labels4train_fin4', cols = cols)


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
Y = pd.DataFrame.to_numpy(data['has_risk'], dtype = 'int64')    # has_risk（categorical）

X_train,X_test,T_train,T_test,y_train,y_test = train_test_split(X, T, Y, test_size=0.3, random_state=0)

## training set
t_scalar = [list(map(time_scalar,i)) for i in T_train] # time scaling
maxLen = len(max(X_train, key=len))
Y_indices = y_train
# 读取单个动作的 embedding，作数值索引
action_to_index, index_to_action, action_to_vec_map = read_action_vecs(emb_dir)
# 把动作转为数值索引
X_indices = actions_to_indices(X_train, action_to_index, maxLen)
# 反过来，最后的动作放在最后面
X_indices = np.array([i[::-1] for i in X_indices])
T_indices = np.array([[-1]*(maxLen-len(i))+i[::-1] for i in t_scalar])
T_indices = T_indices.reshape(T_indices.shape[0], T_indices.shape[1], 1)

## test set
t_scalar_test = [list(map(time_scalar, i)) for i in T_test] 
maxLen = len(max(X_train, key =len))
Y_indices_test = y_test
action_to_index, index_to_action, action_to_vec_map = read_action_vecs(emb_dir)
X_indices_test = actions_to_indices(X_test, action_to_index, maxLen)
X_indices_test = np.array([i[::-1] for i in X_indices_test])
T_indices_test = np.array([[-1]*(maxLen-len(i))+i[::-1] for i in t_scalar_test])
T_indices_test = T_indices_test.reshape(T_indices_test.shape[0], T_indices_test.shape[1], 1)


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]
initial_bias = np.log(sum(Y==1) / (Y.shape[0]-sum(Y==1)))

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_recall', 
    verbose=1,
    patience=5,
    mode='max',
    min_delta=0.003,
    restore_best_weights=True)

model = make_model(metrics=METRICS, output_bias = initial_bias, attention_share = False, bidirectional = True)
model.summary()
history = model.fit(
                    [X_indices,T_indices], 
                    Y_indices, 
                    epochs=50, 
                    batch_size=64, 
                    shuffle=True,
                    validation_data=([X_indices_test, T_indices_test], Y_indices_test),
                    validation_split = 0.2, #从测试集中划分80%给训练集
                    validation_freq = 1, #测试的间隔次数为1,
                    callbacks=[early_stopping]
                    )
model.save(model_dir)

feature_columns = ['len_sequence', 'cnt_pay', 'max_time_diff', 'min_time_diff', 'avg_time_diff', 'std_time_diff', 'cnt_src', 'device_ios', 'device_android', 'device_wap', 'device_web', 'device_app', 'device_mini', 'cnt_login', 'is_bk_log', 'is_wzp_log', 'is_dc_log', 'cnt_item', 'cnt_cheap_item', 'cnt_lyl_item', 'roi', 'avg_roi', 'is_gift_inclued','is_virtual_inclued']
feature_columns.append('lstm')
target_column = ['has_risk']
t_scalar_total = [list(map(time_scalar,i)) for i in T] # time scaling
Y_indices_total = Y
# 把动作转为数值索引
X_indices_total = actions_to_indices(X, action_to_index, maxLen)
# 反过来，最后的动作放在最后面
X_indices_total = np.array([i[::-1] for i in X_indices_total])
# T_indices = np.array([[-1]*(maxLen-len(i))+i[::-1] for i in t])
T_indices_total = np.array([[-1]*(maxLen-len(i))+i[::-1] for i in t_scalar_total])
T_indices_total = T_indices_total.reshape(T_indices_total.shape[0], T_indices_total.shape[1], 1)

data['lstm'] = model.predict([X_indices_total, T_indices_total], batch_size=64)
data[feature_columns] = data[feature_columns].astype(float)
data[target_column] = data[target_column].astype(int)

train_x, test_x, train_y, test_y = train_test_split(data[feature_columns], data[target_column], test_size = 0.2, random_state = 0)
train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y, test_size = 0.2, random_state = 0)

fit_params={"early_stopping_rounds":30, 
            "eval_metric" : evaluate_recall,
            "eval_set" : [(validation_x,validation_y)],
            'eval_names': ['valid'],
            'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_0995)],
            'verbose': 100
           }

param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

n_HP_points_to_test = 500

clf = lgb.LGBMClassifier(objective = 'binary',
                         boosting = 'gbdt',
                         seed = 0,
                         max_depth=-1, 
                         learning_rate = 0.05,
                         random_state=314, 
                         silent=True, 
                         metric=None, 
                         n_jobs=4, 
                         n_estimators=5000)
gs = RandomizedSearchCV(
    estimator=clf, 
    param_distributions=param_test, 
    n_iter=n_HP_points_to_test,
    scoring='recall',
    cv=5,
    refit=True,
    random_state=314,
    verbose=True)
gs.fit(train_x, train_y, **fit_params)

opt_parameters = gs.best_params_

clf_sw = lgb.LGBMClassifier(**clf.get_params())
#set optimal parameters
clf_sw.set_params(**opt_parameters)

gs_sample_weight = GridSearchCV(estimator=clf_sw, 
                                param_grid={'scale_pos_weight':[1,2,6,7,8,12]},
                                scoring='recall',
                                cv=5,
                                refit=True,
                                verbose=True)
gs_sample_weight.fit(train_x, train_y, **fit_params)
opt_parameters["scale_pos_weight"] =  gs_sample_weight.best_params_['scale_pos_weight']
#Configure locally from hardcoded values
clf_final = lgb.LGBMClassifier(**clf.get_params())
#set optimal parameters
clf_final.set_params(**opt_parameters)

# #Train the final model with learning rate decay
clf_final.fit(train_x, train_y, 
              **fit_params
             )
train_prob_cv = clf_final.predict_proba(train_x)[:,1]
validation_prob_cv = clf_final.predict_proba(validation_x)[:,1]
test_prob_cv = clf_final.predict_proba(test_x)[:,1]

print(classification_report(train_y,train_prob_cv>0.5))
print('--------------------------------------------------')
print(classification_report(validation_y,validation_prob_cv>0.5))
print('--------------------------------------------------')
print(classification_report(test_y,test_prob_cv>0.5))
dump(clf_final, '/data/luyining/models/lgb_3.pkl')




