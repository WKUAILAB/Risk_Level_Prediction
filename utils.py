import re
import numpy as np
import pandas as pd
import hdfs
from hdfs.ext.kerberos import KerberosClient
import requests
import os

import lightgbm as lgb
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, precision_recall_fscore_support
from sklearn.ensemble import GradientBoostingClassifier

from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Activation, LSTM, concatenate, Reshape, Permute, Lambda, RepeatVector, Multiply
from tensorflow.keras.layers import Embedding, Bidirectional
from tensorflow.keras.initializers import Constant
import tensorflow.keras.backend as K



def fetch_df(db,table,cols,sep='\x01',nline='\n'):
    '''
    Fetch DataFrame from Database
    '''
    session = requests.Session()
    session.verify = False

    client = KerberosClient(url='http://holmes-mm-master-2.hz.infra.mail:50070', session=session, mutual_auth='REQUIRED')
    
    hdfs_dir = '/user/holmes/hive_db/'+db+'.db/'+table
    hdfs_file_ls = [os.path.join(hdfs_dir,f) for f in client.list(hdfs_dir)]
    print('files in hdfs:',hdfs_dir)
    print(client.list(hdfs_dir))
    # 读取hdfs数据，不可以做 sql 操作
    content = ''
    for f in hdfs_file_ls:
        with client.read(f) as reader:
            content += reader.read().decode('utf-8')

    data = pd.DataFrame([i.split(sep) for i in content.split(nline) if i.strip()!=''],columns=cols)
    return data

# def fetch_str(db, table):
#     session = requests.Session()
#     session.verify = False
#     content = ''
#     hdfs_url = 'http://holmes-mm-master-1.hz.infra.mail:50070'
#     client = KerberosClient(hdfs_url)
#     hdfs_data_dir = '/user/holmes/hive_db/'+ db + '.db/' + table
#     try:
#         hdfs_file_ls = [os.path.join(hdfs_data_dir, f) for f in client.list(hdfs_data_dir)]
#     except hdfs.util.HdfsError as e:
#         client = KerberosClient('http://holmes-mm-master-2.hz.infra.mail:50070')
#         hdfs_file_ls = [os.path.join(hdfs_data_dir, f) for f in client.list(hdfs_data_dir)]
#     for f in hdfs_file_ls:
#         with client.read(f) as reader:
#             content += reader.read().decode("utf-8")
#     return content



# def clean_features(content):
#     actions_sequence = []
#     time_sequence = []
#     label = []
#     id_ls = []
#     ft_ls = []
    
#     for line in content.split('\n'):
#         ls = line.split('\x01')
#         if len(ls)!=35:
#             break
#         has_risk,ds,user_id,order_id,reg_days,platform,usertype,mobile_prefix3,mobile_prefix5,*ft_cols,action,time = ls
        
#         id_ls.append([ds,user_id,order_id])
#         ft_ls.append(['0' if i=='' or re.search('[^\d.]',i) else i for i in ft_cols])
        
#         label.append(has_risk)
        
#         # 缺少当天最后一分钟购买的记录
#         if action=='\\N':action='0'
#         if time=='\\N':time='0'
        
#         actions_sequence.append(action.split(','))
#         time_sequence.append(list(map(int,time.split(','))))
    
#     X = np.asarray(actions_sequence)
#     T = np.asarray(time_sequence)
#     Y = np.asarray(label, dtype=int)
#     F = np.asarray(ft_ls, dtype=float)

#     return X,T,Y,id_ls,F

def actions_to_indices(X, action_to_index, max_len,ref="click"):
    m = X.shape[0]  # number of training examples
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))
    oov_set = set()
    for i in range(m):  # loop over training examples
        # Convert the ith training actions in lower case and split is into actions. You should get a list of actions.
        actions = X[i]
        # Initialize j to 0
        j = 0
        # Loop over the action
        for a in actions:
            # print(a)
            # Set the (i,j)th entry of X_indices to the index of the correct action.
            # 如果不是常见动作，则最后一维补 1
            # a = a.lower().strip()
            a = a.strip()
            if a in action_to_index.keys():
                X_indices[i, j] = action_to_index[a]
            # elif hasattr(ref, '__class__'):
            # elif ref=='model':
            #     # 如果传递了一个语言模型,不行，需要同步更行 emb_fn
            #     X_indices[i, j] = fasttext_model.wv[a]
            elif a.split('_')[0]in action_to_index.keys():
                # print(f"transfoming new words:{a}")
                # 如果首字母在辞典里
                oov_set.add(a)
                X_indices[i, j] = action_to_index[a.split('_')[0]]
            else:
                # 使用默认 oov
                X_indices[i, j] = action_to_index[ref]
            # Increment j to j + 1
            j = j + 1
    print('\n'.join(list(oov_set)))
    return X_indices

def pretrained_embedding_layer(action_to_vec_map, action_to_index):
    '''
    Pre-train an embedding layer
    Input: action_to_vec_map：dict{action: vectors}
           action_to_index: dict{action: index}
    Output: An embedding layer
    '''
    vocab_len = len(action_to_index) + 1  # adding 1 to fit Keras embedding (requirement)
    emb_dim = action_to_vec_map["login"].shape[0]  # define dimensionality
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of action vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim)) 
    
    # Set each row "index" of the embedding matrix to be the action vector representation of the "index"th action of the vocabulary
    try:
        for action, index in action_to_index.items():
            emb_matrix[index, :] = action_to_vec_map[action]
    except:
        pass
    # Define Keras embedding layer with the correct input/outpuit sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = Embedding(vocab_len, emb_dim) 

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def attention_3d_block(inputs, time_steps, attention_share):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, time_steps))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_steps, activation='softmax')(a)
    # SINGLE_ATTENTION_VECTOR: 
    # FALSE: 每维特征会单独有一个权重 / TRUE: 则共享一个注意力权重
    SINGLE_ATTENTION_VECTOR = attention_share
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


    
def RiskLevelPredict(input_shape, action_to_vec_map, action_to_index, output_bias, attention_share, bidirectional = False):
    '''
    Predict Risk level
    Input:  input_shape
            action_to_vec_map: dict{action: vectors}
            action_to_index: dict{action: index}
    Output: model
    '''
    
    if bidirectional:
        
        actions_indices = Input(shape=input_shape, dtype = "int32")
        embedding_layer = pretrained_embedding_layer(action_to_vec_map, action_to_index)
        embeddings = embedding_layer(actions_indices) 
        
        attention = attention_3d_block(embeddings,input_shape[0], attention_share = attention_share)
        X = Bidirectional(LSTM(64, return_sequences = True), name="bi_lstm", merge_mode='concat')(attention)
        delta_time_indices = Input(shape=[input_shape[0],1])
        X2 = LSTM(4, return_sequences=True)(delta_time_indices)
        X = concatenate([X, X2])
        X = Dropout(0.5)(X)
        X = LSTM(128)(X)
        X = Dropout(0.5)(X)        
        X = Dense(1, activation='sigmoid', bias_initializer=output_bias)(X)
        
        model = Model([actions_indices,delta_time_indices], X)
        return model
        
        
        
    else:
#         # Define actions_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
#         actions_indices = Input(shape=input_shape, dtype=np.int32)
#         # Create the embedding layer(≈1 line)
#         embedding_layer = pretrained_embedding_layer(action_to_vec_map, action_to_index)
    
#         # Propagate actions_indices through your embedding layer, you get back the embeddings
#         # pre-trained embedding_layer has input with dimension (99,) and has output with dimension (99,20) 
#         embeddings = embedding_layer(actions_indices)   
    
#         # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
#         X = LSTM(128, return_sequences=True)(embeddings)
#         attention = attention_3d_block(X,input_shape[0], attention_share = attention_share)

#         delta_time_indices = Input(shape=[input_shape[0],1])
#         X2 = LSTM(4, return_sequences=True)(delta_time_indices)
#         X = concatenate([attention, X2])

#         X = Dropout(0.5)(X)
#         X = LSTM(128)(X)
#         X = Dropout(0.5)(X)

#         X = Dense(1, activation='sigmoid', bias_initializer=output_bias)(X) # Output the probability of having risk or not

#         # Create Model instance which converts actions_indices into X.
#         model = Model([actions_indices,delta_time_indices], X)
#         return model

        actions_indices = Input(shape=input_shape, dtype = "int32")
        embedding_layer = pretrained_embedding_layer(action_to_vec_map, action_to_index)
        embeddings = embedding_layer(actions_indices) 
        
        attention = attention_3d_block(embeddings,input_shape[0], attention_share = attention_share)
        X = LSTM(128, return_sequences = True)(attention)
        delta_time_indices = Input(shape=[input_shape[0],1])
        X2 = LSTM(4, return_sequences=True)(delta_time_indices)
        X = concatenate([X, X2])
        X = Dropout(0.5)(X)
        X = LSTM(128)(X)
        X = Dropout(0.5)(X)        
        X = Dense(1, activation='sigmoid', bias_initializer=output_bias)(X)
        
        model = Model([actions_indices,delta_time_indices], X)
        return model

def read_action_vecs(fn):
    '''
    Input:a file name
    '''
    with open(fn, 'r', encoding="utf8") as f:
        actions = set()
        action_to_vec_map = {}
        for line in f:
            line = line.strip().split(" ") # should explicit use " "
            curr_action = line[0]
            actions.add(curr_action)
            action_to_vec_map[curr_action] = np.array(line[1:], dtype=np.float32)
        i = 1
        actions_to_index = {}
        index_to_actions = {}
        for w in sorted(actions):   # dict{action: array 1x20}
            actions_to_index[w] = i # dict{action: index}
            index_to_actions[i] = w # dict{index: action}
            i = i + 1
    return actions_to_index, index_to_actions, action_to_vec_map


def make_model(metrics,output_bias=None, attention_share = False, bidirectional = False):
    cur_dir = '.'
    emb_fn = 'action_page_fasttext.dict'
    emb_dir = emb_dir = os.path.join(cur_dir,'data',emb_fn)
    action_to_index, index_to_action, action_to_vec_map = read_action_vecs(emb_dir)
    maxLen = 100
    if output_bias is not None:
        output_bias = Constant(output_bias)
    model = RiskLevelPredict((maxLen,), action_to_vec_map, action_to_index,output_bias, attention_share = attention_share, bidirectional = bidirectional)
    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=metrics)
    return model

def time_scalar(x):
    if x >=10800:
        return 1
    return x/10800

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

def evaluate_recall(truth, predictions):  
    recall = recall_score(truth, predictions.round())
    return ('recall', recall, True)
