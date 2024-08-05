from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
from alpaca_trade_api import REST
from pandas import Timedelta  # Correct import for Timedelta
from finbert_utils import estimate_sentiment

API_KEY = "PKIGR33QK4FUJAGUQHBC"
API_SECRET = "NKgHxzHDXUa2hCKCPVmN5ocdcdCESiGF3AarDsh4"
BASE_URL = "https://paper-api.alpaca.markets"  # Ensure this is correct

# Dictionary Import
ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}

# All trading logic goes in here
class MLTrader(Strategy):
    def initialize(self, symbol: str="SPY", cash_at_risk: float = 0.5):
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None  # Undoing buys and other features
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def position_sizing(self):
        cash = self.get_cash()  # Ensure this method exists in your parent Strategy class
        last_price = self.get_last_price(self.symbol)  # Ensure this method exists in your parent Strategy class
        quantity = round(cash * self.cash_at_risk / last_price, 0)  # How many units can get per risk amount
        return cash, last_price, quantity
    
    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')
    
    def get_sentiment(self):
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)  # Get dynamic dates based on backtesting
        news = [ev._raw["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment
        
    def on_trading_iteration(self):  # Runs every time there is a tic after the initialization
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()

        if cash > last_price:
            if sentiment == "positive" and probability > .999:
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,  # Change to make dynamic input for shares to purchase
                    "buy",
                    type="bracket",  # Market, limit or bracket order
                    take_profit_price=last_price * 1.20,
                    stop_loss_price=last_price * .95  # If you lose more than that much money
                )
                self.submit_order(order)
                self.last_trade = "buy"
            elif sentiment == "negative" and probability > .999:
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,  # Change to make dynamic input for shares to purchase
                    "sell",
                    type="bracket",  # Market, limit or bracket order
                    take_profit_price=last_price * .8,
                    stop_loss_price=last_price * 1.05  # If you lose more than that much money
                )
                self.submit_order(order)
                self.last_trade = "sell"

start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

broker = Alpaca(ALPACA_CREDS)

# Instantiate Strategy
strategy = MLTrader(name='mlstrat', broker=broker, parameters={"symbol": "SPY", "cash_at_risk": 0.5})

# Perform backtesting and generate the tearsheet
backtest_result = strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": "SPY", "cash_at_risk": 0.5}
)

# Check if backtest_result contains tearsheet data
if backtest_result and hasattr(backtest_result, 'tearsheet'):
    # Save the tearsheet to a file
    tearsheet_path = 'tearsheet.html'
    backtest_result.tearsheet(save_path=tearsheet_path)
    print(f"Tearsheet saved to {tearsheet_path}")
else:
    print("No tearsheet available or backtesting did not complete successfully.")
