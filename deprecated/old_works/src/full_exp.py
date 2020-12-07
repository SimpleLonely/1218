# import packages
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, confusion_matrix, f1_score
from bt_classes import my_backtest, test_indicator
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
#importing required libraries
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, LSTM, CuDNNLSTM
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from tqdm import tqdm
# from sa import *
# from utils import *
# follow the literature
# we don't use min-max scaling here, use partial mean-std scaling instead
from sklearn.preprocessing import StandardScaler
from itertools import chain
rcParams['figure.figsize'] = 20,10
# df = pd.read_csv('../res/input0130.csv')

orig_df = pd.read_csv('../xau_1d_20y.csv')
orig_df['datetime'] = pd.to_datetime(orig_df['date'])
orig_df = orig_df.set_index('datetime')

df = orig_df.copy()
df['log_r'] = np.log(df['close']) - np.log(df['open'])
df['label'] = np.sign(df['log_r'].shift(-1))
df['label'][df['label']==-1] = 0
df['label'] = df['label'].fillna(0)


# Please select the last activation layer.
layer_names = ['lstm_2']

default_upper_bound = 2000
default_n_bucket = 1000
default_n_classes = 2
class Args(): #创建一个类
    def __init__(self): #定义初始化信息。
        self.is_classification = True
        self.save_path = ''
        self.d = 'lstm_r'
        self.num_classes = 2
        self.lsa = True
        self.dsa = True
        self.target = 'none'
        self.batch_size = 128
        self.var_threshold = 1e-5
        self.upper_bound = 2000
        self.n_bucket = 1000
        self.is_classification = True
args = Args()

def lstm_model(sample_len=240,para_a=42, para_b=17,drop1=0.05,drop2=0.02):
    model = Sequential()
    # model.add(LSTM(units=para_a, dropout=0.1, return_sequences=True, input_shape=(sample_len,1),activation='tanh'))# (25,15)-57, (42,17)-58
    # model.add(LSTM(units=para_b, dropout=0.08, activation='tanh'))
    model.add(CuDNNLSTM(units=para_a, return_sequences=True, input_shape=(sample_len,1)))# (25,15)-57, (42,17)-58
    model.add(Dropout(drop1))
    model.add(Activation('tanh'))
    model.add(CuDNNLSTM(units=para_b))
    model.add(Dropout(drop2))
    model.add(Activation('tanh'))
    # model.add(Dropout(0.08))# 加了之后同原先效果差不多，（应该一定程度上）可以防止过拟合
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# this experiment is intended for trying to calculate the transition probability matrix rollingly.
# firstly let's define some useful functions
def get_transtitions(y_true, y_pred):
    '''
    To generate transition probability matrix with y_true, y_pred of an any period.
    '''
    y_output = y_pred
    y = y_true
    continues_suc = 0
    continues_fail = 0
    result = []
    maxx = 0
    for i in range(0,len(y_output)):
        if y_output[i] == y[i]:
            continues_suc+=1
            if continues_fail!=0:
                result.append(-continues_fail)
                if continues_fail > maxx:
                    maxx = continues_fail
                continues_fail = 0
        else:
            continues_fail+=1
            if continues_suc != 0:
                result.append(continues_suc)
                if continues_suc > maxx:
                    maxx = continues_suc
                continues_suc = 0

    length = maxx+1
    suc_result = [[0] * length for i in range(length)]
    fail_result = [[0]*length for i in range(length)]

    for i in range(len(result)-1):
        if result[i]>0:
            suc_result[result[i]][-result[i+1]]+=1
        else:
            fail_result[-result[i]][result[i+1]]+=1
    return suc_result, fail_result

def get_trans_prob(suc_result, fail_result, weighted=False):
    status_porb = {}
    if weighted:
        for i in range(len(suc_result)):
            fail = np.sum([j*suc_result[i][j] for j in range(len(suc_result[i]))])
            if i+1 < len(suc_result):
                success = np.sum([(j-i)*np.sum(suc_result[j]) for j in range(i+1,len(suc_result))])
                # success = np.sum(suc_result[i+1:])
            else:
                success = 0
            status_porb[i] = success / (success + fail)

        for i in range(len(fail_result)):
            success = np.sum([j*fail_result[i][j] for j in range(len(fail_result[i]))])
            if i+1 < len(fail_result):
                fail = np.sum([(j-i)*np.sum(fail_result[j]) for j in range(i+1,len(fail_result))])
                # fail = np.sum(fail_result[i+1:])
            else:
                fail = 0
            status_porb[-i] = success / (success + fail)
    else:
        for i in range(len(suc_result)):
            fail = np.sum(suc_result[i])
            if i+1 < len(suc_result):
                success = np.sum(suc_result[i+1:])
            else:
                success = 0
            status_porb[i] = success / (success + fail)

        for i in range(len(fail_result)):
            success = np.sum(fail_result[i])
            if i+1 < len(fail_result):
                fail = np.sum(fail_result[i+1:])
            else:
                fail = 0
            status_porb[-i] = success / (success + fail)
    return status_porb

def trans_prob(y_true, y_pred, weighted=False):
    suc_result, fail_result = get_transtitions(y_true, y_pred)
    return get_trans_prob(suc_result, fail_result, weighted)

def get_suc_num(test_df):
    test_df['win'] = -1
    test_df['win'].loc[test_df['y_true']==test_df['y_pred']] = 1
    test_df['suc_num'] = np.nan
    test_df['suc_num'].loc[test_df['win']!=test_df['win'].shift(1)] = 1
    test_df['suc_num'] = test_df['suc_num'].cumsum().fillna(method='ffill')
    test_df['suc_num'] = test_df.groupby('suc_num')['suc_num'].cumsum() / test_df['suc_num'] * test_df['win']
    return test_df['suc_num']

def get_adj_metrics(test_df):
    pre_acc = accuracy_score(test_df['y_true'],test_df['y_pred'])
    pre_pre = precision_score(test_df['y_true'],test_df['y_pred'],labels=[0,1])
    pre_rec = recall_score(test_df['y_true'],test_df['y_pred'],labels=[0,1])
    pre_f1 = f1_score(test_df['y_true'],test_df['y_pred'],labels=[0,1])
    pre_cm = confusion_matrix(test_df['y_true'],test_df['y_pred'],labels=[0,1])
    pre_cm00,pre_cm01,pre_cm10,pre_cm11 = pre_cm[0][0],pre_cm[0][1],pre_cm[1][0],pre_cm[1][1]
    after_acc = accuracy_score(test_df['y_true'],test_df['adjusted_pred'])
    after_pre = precision_score(test_df['y_true'],test_df['adjusted_pred'],labels=[0,1])
    after_rec = recall_score(test_df['y_true'],test_df['adjusted_pred'],labels=[0,1])
    after_f1 = f1_score(test_df['y_true'],test_df['adjusted_pred'],labels=[0,1])
    after_cm = confusion_matrix(test_df['y_true'],test_df['adjusted_pred'],labels=[0,1])
    after_cm00,after_cm01,after_cm10,after_cm11 = after_cm[0][0],after_cm[0][1],after_cm[1][0],after_cm[1][1]

    test_df['label'] = test_df['y_pred'].shift(-1).fillna(0)
    pre_adj,pre_dd,pre_ar = my_backtest(test_df.iloc[fit_window:])
    test_df['label'] = test_df['adjusted_pred'].shift(-1).fillna(0)
    after_adj,after_dd,after_ar = my_backtest(test_df.iloc[fit_window:])

    test_df['adj_true'] = 1
    test_df['adj_true'].loc[test_df['y_true']==test_df['y_pred']] = 0
    test_df['adj_pred'] = 1
    test_df['adj_pred'].loc[test_df['adjusted_pred']==test_df['y_pred']] = 0
    adj_acc = accuracy_score(test_df['adj_true'],test_df['adj_pred'])
    adj_pre = precision_score(test_df['adj_true'],test_df['adj_pred'],labels=[0,1])
    adj_rec = recall_score(test_df['adj_true'],test_df['adj_pred'],labels=[0,1])
    adj_f1 = f1_score(test_df['adj_true'],test_df['adj_pred'],labels=[0,1])
    adj_cm = confusion_matrix(test_df['adj_true'],test_df['adj_pred'],labels=[0,1])
    adj_cm00,adj_cm01,adj_cm10,adj_cm11 = adj_cm[0][0],adj_cm[0][1],adj_cm[1][0],adj_cm[1][1]

    test_df['log_profit'] = 2*(test_df['y_pred']-0.5)*test_df['log_r']
    win_profit = test_df['log_profit'].loc[test_df['y_true']==test_df['y_pred']].mean()
    lose_profit = test_df['log_profit'].loc[test_df['y_true']!=test_df['y_pred']].mean()
    pre_wtl = abs(win_profit / lose_profit)
    adj_win_profit = test_df['log_profit'].loc[test_df['y_true']==test_df['adjusted_pred']].mean()
    adj_lose_profit = test_df['log_profit'].loc[test_df['y_true']!=test_df['adjusted_pred']].mean()
    adj_wtl = abs(adj_win_profit / adj_lose_profit)

    return [pre_acc,pre_pre,pre_rec,pre_f1,pre_cm00,pre_cm01,pre_cm10,pre_cm11,after_acc,after_pre,after_rec,after_f1,after_cm00,after_cm01,after_cm10,after_cm11,pre_adj,pre_dd,pre_ar,after_adj,after_dd,after_ar,adj_acc,adj_pre,adj_rec,adj_f1,adj_cm00,adj_cm01,adj_cm10,adj_cm11,win_profit,lose_profit,pre_wtl,adj_win_profit,adj_lose_profit,adj_wtl]


# reproduce training set
# sample_len = 9
# p1 = 192
# p2 = 192
# epochs = 30
# batch_size = 200
train_len = 1500
test_len = 500
weighted = False
fit_window = 100

rs_list = []
for samlen in [6,9,30,120]:
    rs_list.append(pd.read_csv(f'grid_result_{samlen}.csv',index_col=0))
grid_result = pd.concat(rs_list).reset_index(drop=True)
all_result = []
for index,row in grid_result.sort_values('test_acc',ascending=False)[['sample_len','p1','p2','epoch','batch_size']].iloc[20:30].iterrows():
    sample_len,p1,p2,epochs,batch_size = row.astype(int).to_list()

            # train_begin = sample_len
    for train_begin in np.arange(sample_len,len(df)-train_len-test_len-sample_len*2,test_len):
        for fit_window in [100, 150, 200]:
            for weighted in [True, False]:
                if weighted:
                    prefix = [sample_len, p1, p2, epochs, batch_size, train_len, test_len, weighted, fit_window, train_begin]
                    this_result = []
                    train_end = train_begin + train_len
                    test_begin = train_end + sample_len
                    test_end = test_begin + test_len

                    scaler = StandardScaler()
                    train_set = df[['log_r','label']][train_begin-sample_len:train_end].reset_index()
                    x_train, y_train = [], []
                    x_train_set = list(chain.from_iterable(scaler.fit_transform(train_set['log_r'].values.reshape(-1,1))))
                    for i in range(sample_len,len(x_train_set)):
                        x_train.append(x_train_set[i-sample_len:i])
                        y_train.append(train_set['label'][i])
                    x_train, y_train = np.array(x_train), np.array(y_train)
                    y_train = to_categorical(y_train,2)
                    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

                    model_name = f'd{sample_len}-{p1}_{p2}_{epochs}_{batch_size}_{train_begin}.h5'
                    if os.path.exists(model_name):
                        model = load_model(model_name)
                    else:
                        model = lstm_model(sample_len=sample_len,para_a=p1,para_b=p2)
                        model.fit(x_train,y_train,epochs=epochs, batch_size=batch_size, callbacks=[EarlyStopping(monitor='loss',patience=10)])
                        model.save(model_name)


                    x_test, y_test = [], []
                    test_set = df[['log_r','label']][test_begin-sample_len:test_end].reset_index()
                    test_df = df[test_begin:test_end].copy()
                    x_test_set = list(chain.from_iterable(scaler.transform(test_set['log_r'].values.reshape(-1,1))))
                    for i in range(sample_len,len(x_test_set)):
                        x_test.append(x_test_set[i-sample_len:i])
                        y_test.append(test_set['label'][i-1])
                    test_df['y_true'] = y_test
                    x_test, y_test = np.array(x_test), np.array(y_test)
                    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
                    y_test = to_categorical(y_test,2)
                    y_pred = model.predict_classes(x_test)
                    test_df['y_pred'] = y_pred
                    test_loss,test_acc = model.evaluate(x_test, y_test,verbose=0)
                    precision = precision_score(test_df['y_true'],test_df['y_pred'],labels=[0,1])
                    recall = recall_score(test_df['y_true'],test_df['y_pred'],labels=[0,1])
                    f1 = f1_score(test_df['y_true'],test_df['y_pred'],labels=[0,1])
                    cm = confusion_matrix(test_df['y_true'],test_df['y_pred'],labels=[0,1])
                    cm00,cm01,cm10,cm11 = cm[0][0],cm[0][1],cm[1][0],cm[1][1]
                    test_df['log_profit'] = 2*(test_df['y_pred']-0.5)*test_df['log_r']
                    win_profit = test_df['log_profit'].loc[test_df['y_true']==test_df['y_pred']].sum()
                    lose_profit = test_df['log_profit'].loc[test_df['y_true']!=test_df['y_pred']].sum()
                    wtl = abs(win_profit / lose_profit)
                    test_df['label'] = test_df['y_pred'].shift(-1)
                    sharpe, dd, ar = my_backtest(test_df)
                    result1 = [test_loss,test_acc,precision,recall,f1,cm00,cm01,cm10,cm11,win_profit,lose_profit,wtl,sharpe,dd,ar]

                    ## 滚动的测试：先把全部预测、连续对错状态都算出来，再遍历判断修改,使用短期历史对错法则

                    x_test, y_test = [], []
                    test_set = df[['log_r','label']][test_begin-sample_len:test_end].reset_index()
                    test_df = df[test_begin:test_end].copy()
                    x_test_set = list(chain.from_iterable(scaler.transform(test_set['log_r'].values.reshape(-1,1))))
                    for i in range(sample_len,len(x_test_set)):
                        x_test.append(x_test_set[i-sample_len:i])
                        y_test.append(test_set['label'][i-1])
                    test_df['y_true'] = y_test
                    x_test, y_test = np.array(x_test), np.array(y_test)
                    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
                    y_test = to_categorical(y_test,2)

                    y_pred = model.predict_classes(x_test)
                    test_df['y_pred'] = y_pred

                    test_df['win'] = 0
                    test_df['win'].loc[test_df['y_true']==test_df['y_pred']] = 1

                    win_rate = [1] * fit_window
                    for i in range(fit_window, len(test_df)):
                        true_prob = test_df['win'].iloc[i-fit_window:i].sum() / fit_window
                        win_rate.append(true_prob)
                    test_df['win_rate'] = win_rate
                    test_df['adjusted_pred'] = test_df['y_pred']
                    test_df['adjusted_pred'].loc[test_df['win_rate']<0.5] = 1 - test_df['adjusted_pred'].loc[test_df['win_rate']<0.5]
                    result2 = get_adj_metrics(test_df.iloc[fit_window:])

                ## 滚动的测试：先把全部预测、连续对错状态都算出来，再遍历判断修改

                x_test, y_test = [], []
                test_set = df[['log_r','label']][test_begin-sample_len:test_end].reset_index()
                test_df = df[test_begin:test_end].copy()
                x_test_set = list(chain.from_iterable(scaler.transform(test_set['log_r'].values.reshape(-1,1))))
                for i in range(sample_len,len(x_test_set)):
                    x_test.append(x_test_set[i-sample_len:i])
                    y_test.append(test_set['label'][i-1])
                test_df['y_true'] = y_test
                x_test, y_test = np.array(x_test), np.array(y_test)
                x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
                y_test = to_categorical(y_test,2)

                y_pred = model.predict_classes(x_test)
                test_df['y_pred'] = y_pred

                test_df['suc_num'] = get_suc_num(test_df)

                win_rate = [1] * fit_window
                for i in range(fit_window, len(test_df)):
                    this_true = test_df['y_true'].iloc[i-fit_window:i]
                    this_pred = test_df['y_pred'].iloc[i-fit_window:i]
                    this_prob = trans_prob(this_true,this_pred,weighted)
                    last_suc = test_df['suc_num'].iloc[i-1]
                    if last_suc not in this_prob.keys():
                        if last_suc > 0:
                            this_win = 0
                        else:
                            this_win = 1
                    else:
                        this_win = this_prob[last_suc]
                    win_rate.append(this_win)
                test_df['win_rate'] = win_rate
                test_df['adjusted_pred'] = test_df['y_pred']
                test_df['adjusted_pred'].loc[test_df['win_rate']<0.5] = 1 - test_df['adjusted_pred'].loc[test_df['win_rate']<0.5]

                result3 = get_adj_metrics(test_df.iloc[fit_window:])

                ## 浮动阈值法
                ## 先计算出全部输出概率值，然后挑取大者观察
                if weighted:
                    x_test, y_test = [], []
                    test_set = df[['log_r','label']][test_begin-sample_len:test_end].reset_index()
                    test_set = df[['log_r','label']][test_begin-sample_len:test_end].reset_index()
                    test_df = df[test_begin:test_end].copy()
                    x_test_set = list(chain.from_iterable(scaler.transform(test_set['log_r'].values.reshape(-1,1))))
                    for i in range(sample_len,len(x_test_set)):
                        x_test.append(x_test_set[i-sample_len:i])
                        y_test.append(test_set['label'][i-1])
                    test_df['y_true'] = y_test
                    x_test, y_test = np.array(x_test), np.array(y_test)
                    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
                    y_test = to_categorical(y_test,2)

                    y_pred = model.predict_classes(x_test)
                    test_df['y_pred'] = y_pred
                    y_pred_prob = model.predict(x_test)

                    test_df['win'] = -1
                    test_df['win'].loc[test_df['y_true']==test_df['y_pred']] = 1
                    test_df['max_conf'] = y_pred_prob.max(axis=1)

                    trust_thres = [0.5] * fit_window
                    for i in range(fit_window, len(test_df)):
                        this_df = test_df.iloc[i-fit_window:i]
                        this_win = this_df['max_conf'].loc[this_df['win']==1].mean()
                        this_lose = this_df['max_conf'].loc[this_df['win']==-1].mean()
                        if this_win > this_lose:
                            threshold = (this_win + this_lose) * 0.5
                            trust_thres.append(threshold)
                        else:
                            trust_thres.append(1)

                    test_df['trust_thres'] = threshold
                    test_df['adjusted_pred'] = y_pred
                    test_df['adjusted_pred'].loc[test_df['max_conf'] < test_df['trust_thres']] = 1 - test_df['adjusted_pred'].loc[test_df['max_conf'] < test_df['trust_thres']]

                    result4 = get_adj_metrics(test_df.iloc[fit_window:])
                all_result.append(prefix+result1+result2+result3+result4)
rs = pd.DataFrame(all_result,columns=['sample_len', 'p1', 'p2', 'epochs', 'batch_size', 'train_len', 'test_len', 'weighted', 'fit_window', 'train_begin','test_loss','test_acc','precision','recall','f1','cm00','cm01','cm10','cm11','win_profit','lose_profit','wtl','sharpe','dd','ar','pre_acc_1','pre_pre_1','pre_rec_1','pre_f1_1','pre_cm00_1','pre_cm01_1','pre_cm10_1','pre_cm11_1','after_acc_1','after_pre_1','after_rec_1','after_f1_1','after_cm00_1','after_cm01_1','after_cm10_1','after_cm11_1','pre_adj_1','pre_dd_1','pre_ar_1','after_adj_1','after_dd_1','after_ar_1','adj_acc_1','adj_pre_1','adj_rec_1','adj_f1_1','adj_cm00_1','adj_cm01_1','adj_cm10_1','adj_cm11_1','win_profit_1','lose_profit_1','pre_wtl_1','adj_win_profit_1','adj_lose_profit_1','adj_wtl','pre_acc_2','pre_pre_2','pre_rec_2','pre_f1_2','pre_cm00_2','pre_cm01_2','pre_cm10_2','pre_cm11_2','after_acc_2','after_pre_2','after_rec_2','after_f1_2','after_cm00_2','after_cm01_2','after_cm10_2','after_cm11_2','pre_adj_2','pre_dd_2','pre_ar_2','after_adj_2','after_dd_2','after_ar_2','adj_acc_2','adj_pre_2','adj_rec_2','adj_f1_2','adj_cm00_2','adj_cm01_2','adj_cm10_2','adj_cm11_2','win_profit_2','lose_profit_2','pre_wtl_2','adj_win_profit_2','adj_lose_profit_2','adj_wtl_2','pre_acc_3','pre_pre_3','pre_rec_3','pre_f1_3','pre_cm00_3','pre_cm01_3','pre_cm10_3','pre_cm11_3','after_acc_3','after_pre_3','after_rec_3','after_f1_3','after_cm00_3','after_cm01_3','after_cm10_3','after_cm11_3','pre_adj_3','pre_dd_3','pre_ar_3','after_adj_3','after_dd_3','after_ar_3','adj_acc_3','adj_pre_3','adj_rec_3','adj_f1_3','adj_cm00_3','adj_cm01_3','adj_cm10_3','adj_cm11_3','win_profit_3','lose_profit_3','pre_wtl_3','adj_win_profit_3','adj_lose_profit_3','adj_wtl_3'])
rs.to_csv('exp_result3.csv')