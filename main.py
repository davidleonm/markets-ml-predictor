import argparse
import logging
import sys
from datetime import datetime

import colorlog
import mplfinance as mpf
import pandas
import yfinance as yf
from pandas import DataFrame

# Charts configuration
SMA200_LINE_COLOR: str = "blue"
EMA4_LINE_COLOR: str = "black"
EMA18_LINE_COLOR: str = "black"
EMA40_LINE_COLOR: str = "orange"
RSI_LINE_COLOR: str = "orange"
RSI_OVERSOLD_LINE_COLOR: str = "green"
RSI_OVERBOUGHT_LINE_COLOR: str = "red"
MACD_LINE_COLOR: str = "blue"
MACD_SIGNAL_LINE_COLOR: str = "orange"
MACD_HISTOGRAM_LINE_COLOR: str = "dimgray"

SMA200_LINE_WIDTH: float = 2
EMA4_LINE_WIDTH: float = 0.5
EMA18_LINE_WIDTH: float = 0.5
EMA40_LINE_WIDTH: float = 1
RSI_LINE_WIDTH: float = 1
RSI_OVERSOLD_LINE_WIDTH: float = 0.5
RSI_OVERBOUGHT_LINE_WIDTH: float = 0.5
MACD_LINE_WIDTH: float = 0.5
MACD_SIGNAL_LINE_WIDTH: float = 0.5

RSI_PERIOD: int = 14
MACD_FAST_PERIOD: int = 14
MACD_SLOW_PERIOD: int = 28
MACD_SIGNAL_PERIOD: int = 9

TRIPLE_CROSS_THRESHOLD: float = 0.5

# Columns
CLOSE: str = "Close"

# Moving average columns
SMA200: str = "SMA200"
EMA4: str = "EMA4"
EMA18: str = "EMA18"
EMA40: str = "EMA40"

# Indicator columns
RSI: str = "RSI"
MACD: str = "MACD"
MACD_SIGNAL: str = "MACD_SIGNAL"
MACD_HISTOGRAM: str = "MACD_HISTOGRAM"
CROSS: str = "CROSS"
ASCENT_CROSS: str = "ASCENT_CROSS"
DESCENT_CROSS: str = "DESCENT_CROSS"


# Helper methods
def get_timestamp_seconds(start_time: datetime) -> float:
    return round(number=(datetime.now() - start_time).total_seconds(), ndigits=2)


def get_logger(name: str) -> logging.Logger:
    custom_logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    color_formatter = colorlog.ColoredFormatter(
        fmt="%(log_color)s%(asctime)s - [%(levelname)s] [%(module)s] - %(message)s",
        log_colors={
            "DEBUG": "white",
            "INFO": "blue",
            "WARNING": "bold_yellow",
            "ERROR": "bold_red",
            "CRITICAL": "bold_red,bg_white",
        },
    )
    handler.setFormatter(color_formatter)

    custom_logger.handlers = [handler]
    custom_logger.setLevel(logging.getLevelName('DEBUG'))

    return custom_logger


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML stock market price predictor")
    parser.add_argument(
        "--ticker", help="Ticker or stock symbol", type=str, required=True
    )

    return parser.parse_args()


# Methods to add indicators and moving averages
def set_rsi(stock_data: DataFrame, period: int) -> None:
    delta = stock_data[CLOSE].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    stock_data[RSI] = 100 - (100 / (1 + rs))


def set_macd(
        stock_data: DataFrame, fast_period: int, slow_period: int, signal_period: int
) -> None:
    short_ema = stock_data[CLOSE].ewm(span=fast_period, adjust=False).mean()
    long_ema = stock_data[CLOSE].ewm(span=slow_period, adjust=False).mean()

    stock_data[MACD] = short_ema - long_ema
    stock_data[MACD_SIGNAL] = (
        stock_data[MACD].ewm(span=signal_period, adjust=False).mean()
    )
    stock_data[MACD_HISTOGRAM] = stock_data[MACD] - stock_data[MACD_SIGNAL]


def set_triple_cross(stock_data: DataFrame, threshold: float) -> None:
    stock_data[CROSS] = (abs(stock_data[EMA4] - stock_data[EMA18]) < threshold) & (
            abs(stock_data[EMA18] - stock_data[EMA40]) < threshold
    )

    stock_data[ASCENT_CROSS] = pandas.NA
    stock_data[DESCENT_CROSS] = pandas.NA
    stock_data.loc[
        (stock_data[CROSS])
        & (stock_data[EMA4] > stock_data[EMA18])
        & (stock_data[EMA18] > stock_data[EMA40]),
        ASCENT_CROSS,
    ] = True
    stock_data.loc[
        (stock_data[CROSS])
        & (stock_data[EMA4] < stock_data[EMA18])
        & (stock_data[EMA18] < stock_data[EMA40]),
        DESCENT_CROSS,
    ] = False


logger = get_logger(name=__name__)


# Main method
def main():
    args = parse_arguments()

    # Disable yahoo finance logging
    ylogger = logging.getLogger("yfinance")
    ylogger.disabled = True
    ylogger.propagate = False

    try:
        # Getting data from yahoo finance
        logger.info(msg=f"Downloading data for {args.ticker.upper()}...")
        start_time = datetime.now()
        stock_data: DataFrame = yf.download(
            tickers=args.ticker.upper(), period="max", interval="1d", progress=False
        )
        stock_data.drop_duplicates(inplace=True)
        stock_data.dropna(inplace=True)
        if stock_data.empty:
            raise ValueError(f"No data found for {args.ticker.upper()}")
        logger.debug(
            msg=f"Data downloaded in {get_timestamp_seconds(start_time=start_time)} seconds"
        )

        # Enrich data with moving averages and indicators
        logger.info(msg="Enriching data...")
        start_time = datetime.now()
        stock_data[SMA200] = stock_data[CLOSE].rolling(window=200).mean()
        stock_data[EMA4] = stock_data[CLOSE].ewm(span=4, adjust=False).mean()
        stock_data[EMA18] = stock_data[CLOSE].ewm(span=18, adjust=False).mean()
        stock_data[EMA40] = stock_data[CLOSE].ewm(span=40, adjust=False).mean()
        set_rsi(stock_data=stock_data, period=RSI_PERIOD)
        set_macd(
            stock_data=stock_data,
            fast_period=MACD_FAST_PERIOD,
            slow_period=MACD_SLOW_PERIOD,
            signal_period=MACD_SIGNAL_PERIOD,
        )
        set_triple_cross(stock_data=stock_data, threshold=TRIPLE_CROSS_THRESHOLD)
        logger.debug(
            msg=f"Data enriched in {get_timestamp_seconds(start_time=start_time)} seconds"
        )

        more_plots = [
            mpf.make_addplot(
                data=stock_data[SMA200],
                color=SMA200_LINE_COLOR,
                width=SMA200_LINE_WIDTH,
                label=SMA200,
            ),
            mpf.make_addplot(
                data=stock_data[EMA4],
                color=EMA4_LINE_COLOR,
                width=EMA4_LINE_WIDTH,
                label=EMA4,
                linestyle="dotted",
            ),
            mpf.make_addplot(
                data=stock_data[EMA18],
                color=EMA18_LINE_COLOR,
                width=EMA18_LINE_WIDTH,
                label=EMA18,
                linestyle="dotted",
            ),
            mpf.make_addplot(
                data=stock_data[EMA40],
                color=EMA40_LINE_COLOR,
                width=EMA40_LINE_WIDTH,
                label=EMA40,
            ),
            mpf.make_addplot(
                stock_data[ASCENT_CROSS], type="scatter", markersize=200, marker="^"
            ),
            mpf.make_addplot(
                stock_data[DESCENT_CROSS], type="scatter", markersize=200, marker="v"
            ),
            mpf.make_addplot(
                data=stock_data[RSI],
                panel=1,
                color=RSI_LINE_COLOR,
                width=RSI_LINE_WIDTH,
                ylabel=RSI,
            ),
            mpf.make_addplot(
                data=[70] * len(stock_data),
                panel=1,
                color=RSI_OVERBOUGHT_LINE_COLOR,
                width=RSI_OVERBOUGHT_LINE_WIDTH,
                linestyle="--",
                secondary_y=False,
            ),
            mpf.make_addplot(
                data=[30] * len(stock_data),
                panel=1,
                color=RSI_OVERSOLD_LINE_COLOR,
                width=RSI_OVERSOLD_LINE_WIDTH,
                linestyle="--",
                secondary_y=False,
            ),
            mpf.make_addplot(
                data=stock_data[MACD],
                panel=2,
                color=MACD_LINE_COLOR,
                ylabel=MACD,
                width=MACD_LINE_WIDTH,
            ),
            mpf.make_addplot(
                data=stock_data[MACD_SIGNAL],
                panel=2,
                color=MACD_SIGNAL_LINE_COLOR,
                width=MACD_SIGNAL_LINE_WIDTH,
            ),
            mpf.make_addplot(
                data=stock_data[MACD_HISTOGRAM],
                panel=2,
                type="bar",
                color=MACD_HISTOGRAM_LINE_COLOR,
                alpha=0.5,
            ),
        ]
        mpf.plot(
            data=stock_data,
            type="candle",
            style="yahoo",
            title=f"Daily chart for {args.ticker.upper()}",
            xlabel="Date",
            ylabel="Price",
            volume=False,
            warn_too_much_data=sys.maxsize,
            addplot=more_plots,
            scale_padding={"left": 0.1, "top": 0.3, "right": 0.5, "bottom": 0.75},
            panel_ratios=(3, 1, 2),
            datetime_format="%d/%m/%Y",
            figscale=2,
        )

        return 0
    except Exception as e:
        logger.critical(e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
