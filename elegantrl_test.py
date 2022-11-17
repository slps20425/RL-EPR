from elegantrl.tutorial_run import *
from elegantrl.tutorial_agent import AgentPPO, AgentDDPG,AgentDiscretePPO
from elegantrl.envs.FinRL.StockTrading import StockTradingEnv,check_stock_trading_env
import yfinance as yf
# from stockstats import StockDataFrame as Sdf
# from StockTradingPortfolio import StockPortfolioEnv
# from portfolio_cash import StockPortfolioCashEnv
# from tradingEnvCash import StockTradingEnvCash
from StockPortfolioCashEnv_AE_30d import StockPortfolioCashEnv_AE_30d


# Agent
args = Arguments(if_on_policy=True)
args.agent = AgentPPO() #AgentPPO() # AgentSAC(), AgentTD3(), AgentDDPG()
args.agent.if_use_gae = True
args.agent.lambda_entropy = 0.00

# Environment
# tickers = [
#   'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'ALGN', 'ALXN', 'AMAT', 'AMD', 'AMGN',
#   'AMZN', 'ASML']  # finrl.config.NAS_74_TICKER

#tickers = [x + '.tw' for x in config.TW_50_TICKER]
tickers = ['2330.tw','2303.tw','2603.tw']#,'USDTWD=X'] #last element is currency USDTWD
tech_indicator_list = ['macd','boll_ub', 'boll_lb', 'rsi_30','close_30_sma', 'close_60_sma']#,'FI','MarginPurchaseTodayBalance','ShortSaleTodayBalance','TWII_close','TWII_volume','turbulence']  # finrl.config.TECHNICAL_INDICATORS_LIST

gamma = 0.99
max_stock = 100
initial_capital = 5e5# 1e6
initial_stocks = np.zeros(len(tickers), dtype=np.float32)
buy_cost_pct = 0.1425*1e-2
sell_cost_pct = 0.1425*1e-2
start_date = '2012-01-01' #'2010-01-01'
start_eval_date = '2018-01-01'
end_eval_date = '2021-10-31'
model_type = 720 # 90 day, 180 day, 360 day
if_PCA = False
if_AE_Trend = True
if_additional_data = True

# args.env = StockTradingEnv('./evan_stock', gamma, max_stock, initial_capital, buy_cost_pct, 
#                            sell_cost_pct, start_date, start_eval_date, 
#                            end_eval_date, tickers, tech_indicator_list, 
#                            initial_stocks, if_eval=False)
# args.eval_env = StockTradingEnv('./evan_stock', gamma, max_stock, initial_capital, buy_cost_pct, 
#                            sell_cost_pct, start_date, start_eval_date, 
#                            end_eval_date, tickers, tech_indicator_list, 
#                            initial_stocks, if_eval=True)

# args.env = StockTradingEnvCash('./evan_trading_cash', gamma, max_stock, initial_capital, buy_cost_pct, 
#                            sell_cost_pct, start_date, start_eval_date, 
#                            end_eval_date, tickers, tech_indicator_list, 
#                            initial_stocks, if_eval=False)
# args.eval_env = StockTradingEnvCash('./evan_trading_cash', gamma, max_stock, initial_capital, buy_cost_pct, 
#                            sell_cost_pct, start_date, start_eval_date, 
#                            end_eval_date, tickers, tech_indicator_list, 
#                            initial_stocks, if_eval=True)

# args.env = StockPortfolioEnv('./evan_portfolio', gamma, max_stock, initial_capital, buy_cost_pct, 
#                            sell_cost_pct, start_date, start_eval_date, 
#                            end_eval_date, tickers, tech_indicator_list, 
#                            initial_stocks, if_eval=False,if_PCA=if_PCA)
# args.eval_env = StockPortfolioEnv('./evan_portfolio', gamma, max_stock, initial_capital, buy_cost_pct, 
#                            sell_cost_pct, start_date, start_eval_date, 
#                            end_eval_date, tickers, tech_indicator_list, 
#                            initial_stocks, if_eval=True,if_PCA=if_PCA)

# args.env = StockPortfolioCashEnv('./evan_portfolio_cash', gamma, max_stock, initial_capital, buy_cost_pct, 
#                            sell_cost_pct, start_date, start_eval_date, 
#                            end_eval_date, tickers, tech_indicator_list, 
#                            initial_stocks, if_eval=False)
# args.eval_env = StockPortfolioCashEnv('./evan_portfolio_cash', gamma, max_stock, initial_capital, buy_cost_pct, 
#                            sell_cost_pct, start_date, start_eval_date, 
#                            end_eval_date, tickers, tech_indicator_list, 
#                            initial_stocks, if_eval=True)
                           
args.env = StockPortfolioCashEnv_AE_30d('./evan_portfolio_cash_AE_30d', gamma, max_stock, initial_capital, buy_cost_pct, 
                           sell_cost_pct, start_date, start_eval_date, 
                           end_eval_date, tickers, tech_indicator_list, 
                           initial_stocks, if_eval=False, if_PCA=if_PCA, if_AE_Trend= if_AE_Trend,if_additional_data=if_additional_data,model_type=model_type)
args.eval_env = StockPortfolioCashEnv_AE_30d('./evan_portfolio_cash_AE_30d', gamma, max_stock, initial_capital, buy_cost_pct, 
                           sell_cost_pct, start_date, start_eval_date, 
                           end_eval_date, tickers, tech_indicator_list, 
                           initial_stocks, if_eval=True,if_PCA = if_PCA, if_AE_Trend = if_AE_Trend,if_additional_data=if_additional_data,model_type=model_type)

args.env.target_reward = 6
args.eval_env.target_reward = 6

# Hyperparameters
args.gamma = gamma
args.break_step = int(1e5)
args.net_dim = 2 ** 11
args.max_step = args.env.max_step
args.max_memo = args.max_step * 3
args.batch_size = 2 ** 11
args.repeat_times = 2 ** 5
args.eval_gap = 2 ** 4
args.eval_times1 = 2 ** 3
args.eval_times2 = 2 ** 6 
args.if_allow_break = False
args.rollout_num = 2 # the number of rollout workers (larger is not always faster)
args.cwd= './dashboard/RLWeight_720d_new_newCritic_bad'
if __name__ == "__main__": 
  train_and_evaluate(args) # the training process will terminate once it reaches the target reward.

