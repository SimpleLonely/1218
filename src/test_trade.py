import pandas as pd


class Trader:
    default_balance = 1000000
    margin_each = 1000
    default_fee = 50
    force_close_level = 0.2

    def __init__(self,name='defaultUser', balance=default_balance, margin = 0, call_position=0, put_positon=0):
        self.balance = balance
        self.name = name
        self.call_position = call_position
        self.put_position = put_positon
        self.margin = margin
        self.records = pd.DataFrame(columns=['time', 'trade_price', 'amt', 'call_pos', 'put_pos', 'balance', 'margin', 'trade_type'])

    def buy_open(self, amt, price, time):
        new_balance = self.balance - Trader.default_fee
        new_margin = self.margin + amt * Trader.margin_each
        if new_balance >= new_margin:
            if new_balance > 0:
                self.balance = new_balance
                self.margin = new_margin
                self.call_position = self.call_position + amt
                new = pd.DataFrame({
                    'time':time,
                    'trade_price':price,
                    'amt':amt,
                    'call_pos':self.call_position,
                    'put_pos':self.put_position,
                    'balance':self.balance,
                    'margin':self.margin,
                    'trade_type':'buy_open'
                })
                self.records.append(new)
                print('%s在价格%lf开多仓%lf手，现仓位为多%lf，空%lf，保证金账户余额为%lf，所需保证金总额为%lf') % self.name, price, amt, new_balance, new_margin
            else:
                print('%s保证金余额已小于0，可能已经爆仓。') % self.name
        else:
            print('%s保证金账户不足，交易失败，保证金账户余额为%lf，所需保证金总额为%lf') % self.name, self.balance, new_margin

    def sell_open(self, amt, price, time):
        new_balance = self.balance - Trader.default_fee
        new_margin = self.margin + amt * Trader.margin_each
        if new_balance >= new_margin:
            if new_balance > 0:
                self.balance = new_balance
                self.margin = new_margin
                self.put_position = self.put_position + amt
                new = pd.DataFrame({
                    'time':time,
                    'trade_price':price,
                    'amt':amt,
                    'call_pos':self.call_position,
                    'put_pos':self.put_position,
                    'balance':self.balance,
                    'margin':self.margin,
                    'trade_type':'sell_open'
                })
                self.records.append(new)
                print('%s在价格%lf开空仓%lf手，现仓位为多%lf，空%lf，保证金账户余额为%lf，所需保证金总额为%lf') % self.name, price, amt, new_balance, new_margin
            else:
                print('%s保证金余额已小于0，可能已经爆仓。') % self.name
        else:
            print('%s保证金账户不足，交易失败，保证金账户余额为%lf，所需保证金总额为%lf') % self.name, self.balance, new_margin

    def buy_close(self, amt, price, time):
        if self.put_position >= amt :
            self.margin = self.margin - amt * Trader.margin_each
            self.put_position = self.put_position - amt
            new = pd.DataFrame({
                'time': time,
                'trade_price': price,
                'amt': amt,
                'call_pos': self.call_position,
                'put_pos': self.put_position,
                'balance': self.balance,
                'margin': self.margin,
                'trade_type': 'buy_close'
            })
            self.records.append(new)
            print('%s在价格%lf平空仓%lf手，现仓位为多%lf，空%lf，保证金账户余额为%lf，所需保证金总额为%lf') % self.name, price, amt, self.call_position, self.put_position, self.balance, self.margin
        else:
            print('%s剩余空仓手数为%lf，却要求操作%lf手，仓位不足，操作失败') % self.name, self.put_position, amt

    def sell_close(self,amt, price, time):
        if self.call_position >= amt:
            self.margin = self.margin - amt * Trader.margin_each
            self.call_position = self.call_position - amt
            new = pd.DataFrame({
                'time': time,
                'trade_price': price,
                'amt': amt,
                'call_pos': self.call_position,
                'put_pos': self.put_position,
                'balance': self.balance,
                'margin': self.margin,
                'trade_type': 'sell_close'
            })
            self.records.append(new)
            print(
                '%s在价格%lf平多仓%lf手，现仓位为多%lf，空%lf，保证金账户余额为%lf，所需保证金总额为%lf') % self.name, price, amt, self.call_position, self.put_position, self.balance, self.margin
        else:
            print('%s剩余多仓手数为%lf，却要求操作%lf手，仓位不足，操作失败') % self.name, self.put_position, amt

    def settle(self, buy_price, sell_price, time):




