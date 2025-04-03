# Core Libraries
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import yfinance.shared as shared
from IPython.display import display
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Scheduling
import schedule

market_open = pd.Timestamp("09:30:00")
market_close = pd.Timestamp("16:00:00")
calendar = USFederalHolidayCalendar()
days_market_open = CustomBusinessDay(calendar=calendar)

intraday_intervals = ["1m", "2m", "5m", "15m", "30m", "60m"]
cols_price = ["Date", "Time", "Close", "Suggestion"]
cols_portfolio = ["Date", "Time", "Close", "Suggestion", "Action", "Shares", "Equity", "Cash", "P&L"]
cols_signals = ["Date", "Time", "Signal_Score", "MACD_Crossover", "RSI_Signal", "BB_Signal", "MACD_Signal"]
cols_ta_indicators = [
    "Date",
    "Time",
    "RSI_14",
    "MACD_5_20_10",
    "MACDs_5_20_10",
    "MACDh_5_20_10",
    "BBL_20_2.0",
    "BBM_20_2.0",
    "BBU_20_2.0",
    "BBB_20_2.0",
    "BBP_20_2.0",
    "Vol_%_Change",
]


def backtest_stock(ticker, interval, backtest_days, cash, buying_increment):
    if validate_input(interval, backtest_days) is False:
        return

    current_time = pd.Timestamp.now().floor("min")

    # Calculate appropriate lookback window (in order to calculate our TA indicators)
    lookback_window, end_date = calculate_backtesting_window(current_time, interval, backtest_days)

    # Create dataframe from Yahoo Finance API
    df = fetch_yfinance(ticker, lookback_window, end_date, interval)
    if df is not False:
        # Clean up dataframe + initialize portfolio
        df.columns.name = ticker
        df["Date"] = df.index.date
        df["Time"] = df.index.strftime("%I:%M %p")
        df["Equity"] = 0.0
        df["Cash"] = float(cash)
        df[["Action", "Shares"]] = 0

        # Calculate TA indicators + slice dataframe from desired start date
        start_date = current_time.normalize() - (backtest_days * days_market_open)
        start_date = start_date.tz_localize("US/Eastern")
        df = calculate_ta_indicators(df)
        df = df.loc[start_date:, :]

        # Suggest trades based on TA indicators + calculate P&L
        df = suggest_trades(df)
        df = calculate_portfolio(df, buying_increment)
        df = calculate_profit_loss(df, cash)
        df = df.set_index("Datetime")
        return df[cols_portfolio + cols_signals[2:] + cols_ta_indicators[2:]]


def validate_input(interval, backtest_days):
    if backtest_days < 0:
        print("Timeframe too short to calculate TA indicators.")
        return False

    if interval == "1m":
        if backtest_days > 5:
            print(
                "Error: Timeframe too long: (Yahoo Finance only provides '1m' interval data up to to 5 business days)."
            )
            return False
    elif interval == "60m":
        if backtest_days > 499:
            print(
                "Error: Timeframe too long (Yahoo Finance only provides '60m' interval data up to 499 business days)."
            )
            return False
    elif interval in intraday_intervals:
        if backtest_days > 40:
            print(
                "Error: Timeframe too long: (Yahoo Finance only provides '"
                + interval
                + "' interval data up to 40 business days)."
            )
            return False

    elif interval != "1d":
        print("Error: Invalid interval")
        return False

    return True


# Calculate how far back we need data from in order to calculate our TA indicators
def calculate_backtesting_window(current_time, interval, backtest_days):
    if interval in intraday_intervals:
        if interval == "1m":
            get_days_before = 5
        elif interval == "60m":
            get_days_before = backtest_days + 5
        else:
            get_days_before = 40

        # Calculate end date (depending if market is still open or not)
        if (current_time >= market_open) & (current_time <= market_close):
            minute_interval = interval + "in"
            end_date = current_time.floor(minute_interval)
        else:
            end_date = current_time.normalize() + pd.Timedelta(days=1)

    elif interval == "1d":
        get_days_before = backtest_days + 32
        end_date = current_time.normalize() + pd.Timedelta(days=1)

    lookback_window = current_time.normalize() - (get_days_before * days_market_open)
    return lookback_window, end_date


def fetch_yfinance(ticker, start, end, interval):
    temp_df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
    )

    # Error Handling
    if len(shared._ERRORS) > 0:
        return False

    temp_df.columns = temp_df.columns.droplevel(1)
    temp_df.index.name = "Datetime"
    if interval in intraday_intervals:
        temp_df.index = temp_df.index.tz_convert("US/Eastern")
    elif interval == "1d":
        temp_df.index = temp_df.index.tz_localize("US/Eastern")

    return temp_df


# Calculate Momentum Indicators (RSI, Bollinger Bands, MACD)
def calculate_ta_indicators(df_input):
    df = df_input.copy()
    df.ta.macd(fast=5, slow=20, signal=10, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.rsi(append=True)

    # Calculate change in Volume (%)
    df["Vol_%_Change"] = round(df["Volume"].diff(1) / df["Volume"], 2)

    #  Calculate RSI, Bollinger Band, and MACD Crossover Signals
    if ("RSI_14" in df.columns) & ("BBP_20_2.0" in df.columns) & ("MACD_5_20_10" in df.columns):
        # Trigger when below 30 (oversold) or above 70 (overbought)
        df["RSI_Signal"] = np.where(df["RSI_14"] < 30, 1, 0)
        df["RSI_Signal"] = np.where(df["RSI_14"] > 70, -1, df["RSI_Signal"])

        # Trigger when price is 10% from lower/upper band
        df["BB_Signal"] = np.where(df["BBP_20_2.0"] < 0.1, 1, 0)
        df["BB_Signal"] = np.where(df["BBP_20_2.0"] > 0.9, -1, df["BB_Signal"])

        # Trigger when MACD crosses signal line
        df["MACD_Signal"] = np.where(df["MACD_5_20_10"] < df["MACDs_5_20_10"], -1, 0)
        df["MACD_Signal"] = np.where(df["MACD_5_20_10"] > df["MACDs_5_20_10"], 1, df["MACD_Signal"])
        df["MACD_Crossover"] = df["MACD_Signal"].diff()

    df = df.apply(lambda x: np.round(x, 2) if pd.api.types.is_numeric_dtype(x) else x)
    return df


# Suggest trade when there is a MACD Crossover AND a signal from either RSI or BBs
def suggest_trades(df):
    signals = df["RSI_Signal"] + df["BB_Signal"] + df["MACD_Crossover"]
    df["Suggestion"] = np.where((signals == 3) | (signals == 4), "Buy", "None")
    df["Suggestion"] = np.where((signals == -3) | (signals == -4), "Sell", df["Suggestion"])
    df["Signal_Score"] = signals
    return df


def calculate_portfolio(df, num_shares_to_buy):
    df["Action"] = np.where(df["Suggestion"] == "Buy", 1, df["Action"])
    df["Action"] = np.where(df["Suggestion"] == "Sell", -1, df["Action"])

    df = df.reset_index()
    for i in range(1, len(df)):
        action = df.at[i, "Action"]
        curr_cash = df.at[i - 1, "Cash"]
        share_price = df.at[i, "Close"]
        current_equity = df.at[i - 1, "Equity"]
        num_shares = df.at[i - 1, "Shares"]
        cost_of_new_shares = num_shares_to_buy * share_price

        if action == 1:
            if curr_cash >= cost_of_new_shares:  # Buy new shares by increment
                df.at[i, "Equity"] = current_equity + cost_of_new_shares
                df.at[i, "Shares"] = num_shares + num_shares_to_buy
                df.at[i, "Cash"] = curr_cash - cost_of_new_shares
            else:
                df.at[i, "Action"] = 0
                df.at[i, "Equity"] = current_equity
                df.at[i, "Shares"] = num_shares
                df.at[i, "Cash"] = df.at[i - 1, "Cash"]

        elif action == -1:  # Sell shares (ALL)
            df.at[i, "Shares"] = 0
            df.at[i, "Equity"] = 0
            df.at[i, "Cash"] = curr_cash + (num_shares * share_price)
        else:
            df.at[i, "Shares"] = num_shares
            df.at[i, "Equity"] = current_equity
            df.at[i, "Cash"] = df.at[i - 1, "Cash"]
    return df


def calculate_profit_loss(df, cash):
    df["P&L"] = np.where(
        df["Action"] == -1,
        (df.shift(1)["Shares"] * df["Close"]) - df["Equity"].shift(1),
        0,
    )
    total_profit = round(df["P&L"].sum(), 2)
    total_profit_percent = round(100 * (total_profit / cash), 2)

    # Display when at least one buy action was triggered
    trades = df[df["Action"] != 0]
    if len(trades) > 0:
        if len(trades[trades["Action"] == 1]) >= 1:  # Print if shares were bought
            display(trades[cols_portfolio])
            print("[Total P&L: $" + str(total_profit) + ", ROI: " + str(total_profit_percent) + "%]")
        else:
            print("No shares were bought during this timeframe..")
    else:
        print("No trades were made during this timeframe..")

    return df


def visualize_indicators(df):
    df = df.reset_index()
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 15))

    # Price action + Bollinger Bands
    ax1.set_title("Momentum Indicators: " + df.columns.name)
    ax1.plot(df["Close"], label=f"{df.columns.name}", color="red")
    ax1.plot(df["BBL_20_2.0"], color="lightgray", linestyle="dashed", alpha=0.9)
    ax1.plot(df["BBM_20_2.0"], label="Bollinger Band", color="lightgray", linestyle="dashed", alpha=0.9)
    ax1.plot(df["BBU_20_2.0"], color="lightgray", linestyle="dashed", alpha=0.9)

    # MACD
    ax2.plot(df["MACD_5_20_10"], label="MACD", color="purple")
    ax2.plot(df["MACDs_5_20_10"], label="Signal", color="orange", alpha=0.8)

    # RSI
    ax3.plot(df["RSI_14"], label="RSI", color="lightgray")
    ax3.set_ylim(0, 100)
    ax3.set_yticks([0, 30, 70, 100])
    ax3.axhline(y=30, color="g", linestyle="dashed", label="Oversold")
    ax3.axhline(y=70, color="r", linestyle="dashed", label="Overbought")

    #  Add buy / sell markers on MACD
    ax2.plot(
        df[df["MACD_Crossover"] == 2].index,
        df[df["MACD_Crossover"] == 2]["MACD_5_20_10"],
        "^",
        color="green",
        markersize=10,
    )
    ax2.plot(
        df[df["MACD_Crossover"] == -2].index,
        df[df["MACD_Crossover"] == -2]["MACD_5_20_10"],
        "v",
        color="red",
        markersize=7.5,
    )

    # Change bin sizing depending on if timeframe is intraday or not
    if df["Date"].value_counts().count() == 1:
        xticklabels = df["Time"]
        xbins = 6.5
    else:
        xticklabels = df["Date"]
        xbins = 7
    fig.tight_layout(pad=2.5)

    # Customize plot
    for ax in fig.axes:
        ax.grid(True, linestyle="--", color="gray", alpha=0.5)
        ax.legend(loc="upper left")
        ax.set_xticks(df.index)
        ax.set_xticklabels(list(xticklabels))
        ax.locator_params(axis="x", nbins=xbins)
        ax.margins(0.01)


# Fetches price data for current day and suggests trades (on a 1 minute interval)
def run_live_daytrading_algo(ticker):
    global live_df
    live_df = suggest_trading_signal(ticker)
    schedule.every(60).seconds.do(job, ticker)

    while True:
        # Continues running algorithm until market close
        if (pd.Timestamp.now().floor("min") >= market_open) & (pd.Timestamp.now().floor("min") <= market_close):
            schedule.run_pending()
        else:
            print("\n<-- Market is now closed -->")
            break


def job(ticker):
    global live_df
    live_df = suggest_trading_signal(ticker)


# Append minute data to live dataframe (create new one if doesn't yet exist)
def suggest_trading_signal(ticker):
    global live_df
    current_time = pd.Timestamp.now().floor("min")
    start_window = pd.Timestamp.now().normalize()

    # Create initial dataframe
    if "live_df" not in globals():
        print("Creating initial DataFrame.. \n")
        live_df = fetch_yfinance(ticker, start_window, current_time, "1m")
        live_df.columns.name = ticker
        live_df["Date"] = live_df.index.date
        live_df["Time"] = live_df.index.strftime("%I:%M %p")

        #  Calculate indicators + suggest trades + pretty print
        live_df = calculate_ta_indicators(live_df)
        live_df = suggest_trades(live_df)
        min = live_df.loc[:, cols_price + cols_signals[2:]]
        print(min.tail().to_string(index=False))

    else:
        prev_minute = current_time - pd.Timedelta(minutes=1)
        prev_minute = prev_minute.tz_localize("US/Eastern")
        prev_minute_str = prev_minute.strftime("%Y-%m-%d %X")

        temp_df = fetch_yfinance(ticker, current_time - pd.Timedelta(minutes=1), current_time, "1m")

        # Append to existing dataframe
        live_df.loc[prev_minute] = {
            "Date": prev_minute.strftime("%Y-%m-%d"),
            "Time": prev_minute.strftime("%I:%M %p"),
            "Open": temp_df.loc[prev_minute_str]["Open"],
            "Close": temp_df.loc[prev_minute_str]["Close"],
            "Low": temp_df.loc[prev_minute_str]["Low"],
            "High": temp_df.loc[prev_minute_str]["High"],
            "Volume": round(temp_df.loc[prev_minute_str]["Volume"]),
        }
        live_df = calculate_ta_indicators(live_df)
        live_df = suggest_trades(live_df)
        min = live_df.loc[:, cols_price + cols_signals[2:]]
        print(min.iloc[-1:].to_string(index=False))

    return live_df
