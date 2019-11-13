import pandas as pd

class Trader:
    default_balance = 1000000 # 初始余额
    margin_each = 1000 # 每手保证金
    default_fee = 50 # 单笔交易费用
    force_close_level = 0.2 # 强制平仓保证金水平
    # 初始化用户
    def __init__(self,name='defaultUser', balance=default_balance, margin = 0, call_position=0, put_positon=0, present_price=0):
        self.balance = balance # 现余额
        self.name = name # 用户名
        self.call_position = call_position # 总多仓
        self.put_position = put_positon # 总空仓
        self.calls_value = call_position * present_price * 100 # 多仓价值
        self.puts_value = (-1) * put_positon * present_price * 100 # 空仓价值
        self.margin = margin # 需要的保证金
        # 该用户的交易记录本，时间、价格、成交量、多仓位、空仓位、余额、所需保证金、交易类型。 
        self.records = pd.DataFrame(columns=['time', 'trade_price', 'amt', 'call_pos', 'put_pos', 'balance', 'margin', 'trade_type', 'call_val', 'put_val'])
        self.id = 1

    # 买开仓，输入成交量、当前价格、时间（指所用日期，可以用index代替）
    def buy_open(self, amt, price, time):
        new_balance = self.balance - Trader.default_fee # 拟变动余额
        new_margin = self.margin + amt * Trader.margin_each # 拟变动保证金
        if new_balance >= new_margin: # 如果变动后余额足够支撑保证金
            if new_balance > 0: # 且余额大于0
                self.balance = new_balance # 设置为新余额
                self.margin = new_margin # 设置为新保证金
                self.call_position = self.call_position + amt # 修改仓位
                self.calls_value = self.calls_value + amt * price * 100 # 多仓价值增加
                new = pd.DataFrame({
                    'time':time,
                    'trade_price':price,
                    'amt':amt,
                    'call_pos':self.call_position,
                    'put_pos':self.put_position,
                    'balance':self.balance,
                    'margin':self.margin,
                    'trade_type':'buy_open',
                    'call_val':self.calls_value,
                    'put_val':self.puts_value
                },index=[self.id]) # 新建记录
                self.id = self.id +1
                self.records = self.records.append(new) # 插入记录
                print('{}在价格{}开多仓{}手，现仓位为多{}，空{}，保证金账户余额为{}，所需保证金总额为{}'.format(self.name, price, amt, self.call_position, self.put_position, new_balance, new_margin))
            else:
                print('{}保证金余额已小于0，可能已经爆仓。'.format(self.name))
        else:
            print('{}保证金账户不足，交易失败，保证金账户余额为{}，所需保证金总额为{}'.format(self.name, self.balance, new_margin))

    def sell_open(self, amt, price, time):
        new_balance = self.balance - Trader.default_fee
        new_margin = self.margin + amt * Trader.margin_each
        if new_balance >= new_margin:
            if new_balance > 0:
                self.balance = new_balance
                self.margin = new_margin
                self.put_position = self.put_position + amt
                self.puts_value = self.puts_value - amt * price * 100 # 空仓价值减少
                new = pd.DataFrame({
                    'time':time,
                    'trade_price':price,
                    'amt':amt,
                    'call_pos':self.call_position,
                    'put_pos':self.put_position,
                    'balance':self.balance,
                    'margin':self.margin,
                    'trade_type':'sell_open',
                    'call_val':self.calls_value,
                    'put_val':self.puts_value
                },index=[self.id])
                self.id = self.id +1
                self.records = self.records.append(new) # 插入记录
                print('{}在价格{}开空仓{}手，现仓位为多{}，空{}，保证金账户余额为{}，所需保证金总额为{}'.format(self.name, price, amt, self.call_position, self.put_position, new_balance, new_margin))
            else:
                print('{}保证金余额已小于0，可能已经爆仓。'.format(self.name))
        else:
            print('{}保证金账户不足，交易失败，保证金账户余额为{}，所需保证金总额为{}'.format(self.name, self.balance, new_margin))

    def buy_close(self, amt, price, time):
        if self.put_position >= amt :
            self.margin = self.margin - amt * Trader.margin_each
            self.put_position = self.put_position - amt
            self.puts_value = self.puts_value + amt * price * 100
            new = pd.DataFrame({
                'time': time,
                'trade_price': price,
                'amt': amt,
                'call_pos': self.call_position,
                'put_pos': self.put_position,
                'balance': self.balance,
                'margin': self.margin,
                'trade_type': 'buy_close',
                'call_val':self.calls_value,
                'put_val':self.puts_value
            },index=[self.id])
            self.id = self.id +1
            self.records = self.records.append(new) # 插入记录
            print('{}在价格{}平空仓{}手，现仓位为多{}，空{}，保证金账户余额为{}，所需保证金总额为{}'.format(self.name, price, amt, self.call_position, self.put_position, self.balance, self.margin))
        else:
            print('{}剩余空仓手数为{}，却要求操作{}手，仓位不足，操作失败'.format(self.name, self.put_position, amt))

    def sell_close(self,amt, price, time):
        if self.call_position >= amt:
            self.margin = self.margin - amt * Trader.margin_each
            self.call_position = self.call_position - amt
            self.calls_value = self.calls_value - amt * price * 100
            new = pd.DataFrame({
                'time': time,
                'trade_price': price,
                'amt': amt,
                'call_pos': self.call_position,
                'put_pos': self.put_position,
                'balance': self.balance,
                'margin': self.margin,
                'trade_type': 'sell_close',
                'call_val':self.calls_value,
                'put_val':self.puts_value
            },index=[self.id])
            self.id = self.id +1
            self.records = self.records.append(new) # 插入记录
            print('{}在价格{}平多仓{}手，现仓位为多{}，空{}，保证金账户余额为{}，所需保证金总额为{}'.format(self.name, price, amt, self.call_position, self.put_position, self.balance, self.margin))
        else:
            print('{}剩余多仓手数为{}，却要求操作{}手，仓位不足，操作失败'.format(self.name, self.put_position, amt))
 
 
    def settle(self, price, time):
        modified_calls_value = price * self.call_position * 100
        modified_puts_value = (-1) * price * self.put_position * 100
        balance_chg = (modified_calls_value - self.calls_value) - (modified_puts_value - self.puts_value)
        print('Balance changed: ' + str(balance_chg))
        self.puts_value = modified_puts_value
        self.calls_value = modified_calls_value
        new_balance = self.balance + balance_chg
        if new_balance < (self.force_close_level * self.margin):
            self.balance = new_balance
            self.call_position = 0
            self.put_position = 0
            self.puts_value = 0
            self.calls_value = 0
            self.margin = 0

        elif new_balance < self.margin:
            self.balance = new_balance
            print('Warning: The user\'s balance is lower than lowest margin needed.')

        else:
            self.balance = new_balance
        
        new = pd.DataFrame({
                'time': time,
                'trade_price': price,
                'amt': 0,
                'call_pos': self.call_position,
                'put_pos': self.put_position,
                'balance': self.balance,
                'margin': self.margin,
                'trade_type': 'settle',
                'call_val':self.calls_value,
                'put_val':self.puts_value
        },index=[self.id])
        self.id = self.id +1
        self.records = self.records.append(new) # 
        print("This day is over.")
    # def settle(self, buy_price, sell_price, time):
    #     modified_calls_value = sell_price * self.call_position
    #     modified_puts_value = (-1) * buy_price * self.put_position
    #     self.balance = 
