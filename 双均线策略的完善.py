import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文显示
plt.rcParams["axes.unicode_minus"] = False

# ---------------------- 1.读取数据 + 清洗 ----------------------
df = pd.read_csv(r"C:\Users\31118\Desktop\量化练习\stock_data.csv")
# 按交易日期升序排序（量化回测最重要！防止乱序）
df = df.sort_values("日期").reset_index(drop=True)

# ---------------------- 2.计算均线、收益率、金叉死叉信号 ----------------------
df['ret'] = df['收盘'].pct_change()
df['ma5'] = df['收盘'].rolling(window=5).mean()
df['ma20'] = df['收盘'].rolling(window=20).mean()

# 信号：ma5>ma20=持仓1，否则空仓0
df['signal'] = np.where(df['ma5'] > df['ma20'], 1, 0)

# 交易标记：signal发生变化，代表产生一次交易
df['trade'] = df['signal'].diff()

# ---------------------- 3.策略收益，扣除交易成本 ----------------------
trade_cost = 0.001
# shift(1)：避免未来函数，当天信号，次日才能交易
df['strategy_ret'] = df['signal'].shift(1) * df['ret']
# 发生交易时扣手续费
df['strategy_ret'] = np.where(df['trade'] != 0, df['strategy_ret'] - trade_cost, df['strategy_ret'])

# 策略净值、基准净值（买入持有）
df['strategy_nav'] = (1 + df['strategy_ret']).cumprod()
df['benchmark_nav'] = (1 + df['ret']).cumprod()

# ---------------------- 4.最大回撤函数 ----------------------
def max_drawdown(series):
    rolling_max = series.cummax()
    drawdown = series / rolling_max - 1
    return drawdown.min()

# 全周期指标
benchmark_dd = max_drawdown(df['benchmark_nav'])
strategy_dd = max_drawdown(df['strategy_nav'])
total_trade = (df['trade'] != 0).sum()

print("==== 全周期回测结果 ====")
print(f"基准(买入持有)最大回撤：{benchmark_dd:.2%}")
print(f"双均线策略最大回撤：{strategy_dd:.2%}")
print(f"总交易次数：{total_trade}")

# ---------------------- 5.样本内、样本外划分（重点！面试高频考点） ----------------------
split_ratio = 0.7
split_idx = int(len(df)*split_ratio)
df_in = df.iloc[:split_idx].copy()
df_out = df.iloc[split_idx:].copy()

in_dd = max_drawdown(df_in['strategy_nav'])
out_dd = max_drawdown(df_out['strategy_nav'])
print("\n==== 样本内/样本外最大回撤 ====")
print(f"样本内最大回撤：{in_dd:.2%}")
print(f"样本外最大回撤：{out_dd:.2%}")

# ---------------------- 6.净值绘图，保存图片放到GitHub README ----------------------
plt.figure(figsize=(12,6))
plt.plot(df['日期'], df['strategy_nav'], label="双均线策略净值")
plt.plot(df['日期'], df['benchmark_nav'], label="买入持有基准净值")
plt.title("双均线策略回测净值对比")
plt.xlabel("日期")
plt.ylabel("净值")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(r"C:\Users\31118\Desktop\量化练习\backtest_plot.png", dpi=200)
plt.show()
print("\n✅ 绘图完成！图片已保存到项目文件夹，可以放进GitHub README")