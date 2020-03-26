from __future__ import (absolute_import, division, print_function, unicode_literals)
import backtrader as bt
import numpy as np
import matplotlib.pyplot as plt

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
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe')
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
    print('Sharpe: {:.2f}'.format(thestrats[0].analyzers.sharpe.get_analysis()['sharperatio']))
    print('Max drawdown: {:.2f}%'.format(thestrats[0].analyzers.dd.get_analysis()['max']['drawdown']))
    print('Annual rate: {:.2f}%'.format(thestrats[0].analyzers.returns.get_analysis()['rnorm100']))