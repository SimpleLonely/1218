from __future__ import (absolute_import, division, print_function, unicode_literals)
import backtrader as bt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
import seaborn as sns

class MyPandasData(bt.feeds.PandasData):
    lines = ('label',)

    params = (
	('datetime', None),
    ('label', -1),
    )


class GoldStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.label = self.datas[0].label
        # To keep track of pending orders
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f, label %d' % (self.dataclose[0],self.label[0]))
        # self.log('Close, %.2f' % self.dataclose[0])
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            if self.label[0] == 1:
                self.log('LONG CREATE, %.2f' % self.dataclose[0])
                self.order = self.buy()
            elif self.label[0] == 0:
                self.log('SHORT CREATE, %.2f' % self.dataclose[0])
                self.order = self.sell()

            # # Not yet ... we MIGHT BUY if ...
            # if self.dataclose[0] < self.dataclose[-1]:
            #         # current close less than previous close

            #         if self.dataclose[-1] < self.dataclose[-2]:
            #             # previous close less than the previous close

            #             # BUY, BUY, BUY!!! (with default parameters)
            #             self.log('BUY CREATE, %.2f' % self.dataclose[0])

            #             # Keep track of the created order to avoid a 2nd order
            #             self.order = self.buy()

        else:
            cur_pos = self.position.size
            # Already in the market ... we might sell
            # if len(self) >= (self.bar_executed + 5):
            if cur_pos > 0 and self.label[0] == 0:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()
            if cur_pos < 0 and self.label[0] == 1:
                self.log('BUY CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()


class GoldStrategy_nolog(bt.Strategy):
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.label = self.datas[0].label

    def next(self):
        if not self.position:
            if self.label[0] == 1:
                self.buy()
            elif self.label[0] == 0:
                self.sell()
        else:
            cur_pos = self.position.size
            if cur_pos > 0 and self.label[0] == 0:
                self.sell()
            if cur_pos < 0 and self.label[0] == 1:
                self.buy()


class OptInvest(bt.Sizer):
    params = (('stake', 1),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        self.params.stake = max(np.floor(cash/(data.close[0]*100)),1)
        return self.params.stake


def my_backtest(mydata, log=False, drawpic=False, iplot=False):
    plt.rcParams['figure.figsize'] = 12,8
    cerebro = bt.Cerebro()
    data = MyPandasData(dataname=mydata)
    cerebro.adddata(data)
    if log:
        cerebro.addstrategy(GoldStrategy)
    else:
        cerebro.addstrategy(GoldStrategy_nolog)
    cerebro.addsizer(OptInvest)
    init_value= 100000.0
    cerebro.broker.setcash(init_value)
    cerebro.broker.setcommission(commission=50,margin=1000,mult=100)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='ar')
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    thestrats = cerebro.run()
    final_value = cerebro.broker.getvalue()
    print('Final Portfolio Value: %.2f' % final_value)
    if drawpic:
        cerebro.plot(iplot=iplot,volume=False)
    sharpe = thestrats[0].analyzers.sharpe.get_analysis()['sharperatio']
    dd = thestrats[0].analyzers.dd.get_analysis()['max']['drawdown']
    ar = thestrats[0].analyzers.returns.get_analysis()['rnorm100']
    print('Sharpe: {:.2f}'.format(sharpe))
    print('Max drawdown: {:.2f}%'.format(dd))
    print('Annual rate: {:.2f}%'.format(ar))
    return sharpe,dd,ar

# 失效的衡量：单点不算，而是去度量一个时间窗内的数据，比如说一个时间窗里的收益率、回撤、精确率准确率召回率、最大连续错误、交叉熵、
class test_indicator():
    def __init__(self,test_df):
        assert 'y_pred' in test_df.columns
        assert 'y_true' in test_df.columns
        self.y_pred = test_df['y_pred'].values
        self.y_true = test_df['y_true'].values
        self.suc_fail, self.fail_suc = np.array(self.successive_distribution())
        # 算这个比例是否要加权？
        self.status_prob = self.get_status_win_rate()
        self.test_df = test_df
        if test_df is not None:
            assert 'label' in test_df.columns, 'label must be in columns'
            test_df['label'] = test_df['y_pred'].shift(-1)
            # self.sharpe, self.drawdown, self.annual = my_backtest(test_df)

    def adjust_label(self):
        test_df = self.test_df
        test_df['win'] = -1
        test_df['win'].loc[test_df['y_true']==test_df['y_pred']] = 1
        test_df['suc_num'] = np.nan
        test_df['suc_num'].loc[test_df['win']!=test_df['win'].shift(1)] = 1
        test_df['suc_num'] = test_df['suc_num'].cumsum().fillna(method='ffill')
        test_df['suc_num'] = test_df.groupby('suc_num')['suc_num'].cumsum() / test_df['suc_num'] * test_df['win']
        test_df['suc_rate'] = test_df['suc_num'].map(self.status_prob)
        test_df['label'] = test_df['y_pred'].shift(-1)
        test_df['label'].loc[test_df['suc_rate']<0.5] = 1 - test_df['label'].loc[test_df['suc_rate']<0.5]
        return test_df['label']

    def successive_distribution(self):
        y_output = self.y_pred
        y = self.y_true
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

    def plot_suc_fail(self):
        plt.figure(figsize=(16,10))
        suc_result = pd.DataFrame(self.suc_fail).sort_index(ascending=False)
        sns.heatmap(suc_result.iloc[:-1,1:],cmap='Blues',annot=True, fmt='.0f')
        plt.yticks(rotation=0)
        plt.xlabel('Successive Wrong',fontsize=16)
        plt.ylabel('Successive Correct',fontsize=16)
        plt.show()
        plt.close()

    def plot_fail_suc(self):
        plt.figure(figsize=(16,10))
        fail_result = pd.DataFrame(self.fail_suc).sort_index(ascending=False)
        sns.heatmap(fail_result.iloc[:-1,1:],cmap='Blues',annot=True, fmt='.0f')
        plt.yticks(rotation=0)
        plt.xlabel('Successive Correct',fontsize=16)
        plt.ylabel('Successive Wrong',fontsize=16)
        plt.show()
        plt.close()

    def get_accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def get_precision(self):
        return precision_score(self.y_true, self.y_pred)

    def get_recall(self):
        return recall_score(self.y_true, self.y_pred)

    def get_successive_n_failure_rate(self,n=5):
        if n > len(self.suc_fail):
            return 0
        return self.suc_fail[:,n:].sum() / self.suc_fail.sum()

    def get_maximum_successive_failure(self):
        # TODO: calculate max failure number
        diff = [1 if self.y_pred[i]==self.y_true[i] else -1 for i in range(len(self.y_pred))]
        curStats = diff[0]
        curNum = 1
        continue01 = []
        for i in range(1,len(diff)):
            if diff[i] == curStats:
                curNum+=1
            else:
                continue01.append(curNum*curStats)
                curStats=diff[i]
                curNum=1
        continue01.append(curNum*curStats)
        return abs(np.min(continue01))

    def get_entropy(self):
        # TODO: calculate entropy
        return log_loss(self.y_true, self.y_pred,labels=[0,1])


    def backtest(self,prob_adjusted=False,log=False,drawpic=False,iplot=False):
        assert self.test_df is not None, 'test_df is not allocated.'
        test_df = self.test_df
        if prob_adjusted:
            test_df['label'] = self.adjust_label()
            win_rate = (test_df['label'].shift(1)==test_df['y_true']).value_counts(True)[True]
            print(f'Adjusted accuracy: {win_rate:.4f}')
        else:
            win_rate = self.get_accuracy()
            print(f'Accuracy: {win_rate:.4f}')
        return [win_rate] + list(my_backtest(test_df,log=log,drawpic=drawpic,iplot=iplot))

    def get_status_win_rate(self, weighted=False):
        suc_result = self.suc_fail
        fail_result = self.fail_suc
        status_porb = {}
        if weighted:
            for i in range(len(suc_result)):
                fail = np.sum([j*suc_result[i][j] for j in range(len(suc_result[i]))])
                if i+1 < len(suc_result):
                    success = np.sum([j*np.sum(suc_result[j]) for j in range(i+1,len(suc_result))])
                else: 
                    success = 0
                status_porb[i] = success / (success + fail)

            for i in range(len(fail_result)):
                success = np.sum([j*fail_result[i][j] for j in range(len(fail_result[i]))])
                if i+1 < len(fail_result):
                    fail = np.sum([j*np.sum(fail_result[j]) for j in range(i+1,len(fail_result))])
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

    def get_drawback(self):
        # TODO: calculate maximum drawback of this phrase.
        return 0

    def get_volatility(self):
        # TODO
        return 0