from __future__ import (absolute_import, division, print_function, unicode_literals)
import backtrader as bt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss

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
    def __init__(self, y_true, y_pred):
        assert len(y_pred) == len(y_true), f'Length of y is {len(y_pred)} while y_pred {len(y_true)}'
        self.y_pred = y_pred
        self.y_true = y_true
        self.suc_fail, fail_suc = np.array(self.successive_distribution())
        # 算这个比例是否要加权？
        self.successive_n_failure_rate = None
        self.period_entropy = None
        self.period_drawback = None
        self.volatility = None

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
        ax = plt.gca()
        plt.xlabel('Successive Wrong',fontsize=16)
        plt.ylabel('Successive Correct',fontsize=16)
        plt.show()
        plt.close()

    def plot_fail_suc(self):
        plt.figure(figsize=(16,10))
        fail_result = pd.DataFrame(self.suc_fail).sort_index(ascending=False)
        sns.heatmap(fail_result.iloc[:-1,1:],cmap='Blues',annot=True, fmt='.0f')
        plt.yticks(rotation=0)
        ax = plt.gca()
        plt.xlabel('Successive Wrong',fontsize=16)
        plt.ylabel('Successive Correct',fontsize=16)
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

    def get_drawback(self):
        # TODO: calculate maximum drawback of this phrase.
        return 0

    def get_volatility(self):
        # TODO
        return 0