import argparse
import logging
import sys
from datetime import datetime

import colorlog
import mplfinance as mpf
import numpy
import pandas
import yfinance as yf
from keras import Sequential
from keras.src.layers import LSTM, Dense
from numpy import ndarray
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

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
TRIPLE_CROSS_MARKER_COLOR: str = "blue"

SMA200_LINE_WIDTH: float = 2
EMA4_LINE_WIDTH: float = 0.5
EMA18_LINE_WIDTH: float = 0.5
EMA40_LINE_WIDTH: float = 1
RSI_LINE_WIDTH: float = 1
RSI_OVERSOLD_LINE_WIDTH: float = 0.5
RSI_OVERBOUGHT_LINE_WIDTH: float = 0.5
MACD_LINE_WIDTH: float = 0.5
MACD_SIGNAL_LINE_WIDTH: float = 0.5
TRIPLE_CROSS_MARKER_SIZE: int = 100

RSI_PERIOD: int = 14
MACD_FAST_PERIOD: int = 14
MACD_SLOW_PERIOD: int = 28
MACD_SIGNAL_PERIOD: int = 9

TRIPLE_CROSS_THRESHOLD: float = 20

# Base columns
CLOSE: str = "Close"
VOLUME: str = "Volume"

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

# Columns for predictions
FEATURES: list[str] = [CLOSE, VOLUME, SMA200, EMA4, EMA18, EMA40, RSI, MACD, MACD_SIGNAL, MACD_HISTOGRAM]
CLOSE_PREDICTED_RF: str = "ClosePredictedRF"
CLOSE_PREDICTED_XGB: str = "ClosePredictedXGB"

# Constants
NUMBER_OF_PREDICTIONS_TO_COMPARE: int = 365


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
        "--ticker", help="Ticker or stock symbol taken from https://finance.yahoo.com", type=str, required=True
    )

    return parser.parse_args()


def create_lstm_sequence(dataset: ndarray, time_step: int) -> tuple:
    sequences, labels = [], []

    for i in range(len(dataset) - time_step):
        sequences.append(dataset[i:i + time_step])
        labels.append(dataset[i + time_step])

    return numpy.array(sequences), numpy.array(labels)


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
    cross_condition = ((numpy.abs(stock_data[EMA4] - stock_data[EMA18]) < threshold) &
                       (numpy.abs(stock_data[EMA4] - stock_data[EMA40]) < threshold) &
                       (numpy.abs(stock_data[EMA18] - stock_data[EMA40]) < threshold))
    stock_data[CROSS] = stock_data[EMA4].where(cross_condition)

    ascent_cross_condition = (stock_data[EMA4] < stock_data[EMA40]) & (stock_data[EMA18] < stock_data[EMA40])
    stock_data[ASCENT_CROSS] = stock_data[CROSS].where(ascent_cross_condition)

    descent_cross_condition = (stock_data[EMA4] > stock_data[EMA40]) & (stock_data[EMA18] > stock_data[EMA40])
    stock_data[DESCENT_CROSS] = stock_data[CROSS].where(descent_cross_condition)


def set_moving_averages(stock_data: DataFrame) -> None:
    stock_data[SMA200] = stock_data[CLOSE].rolling(window=200).mean()
    stock_data[EMA4] = stock_data[CLOSE].ewm(span=4, adjust=False).mean()
    stock_data[EMA18] = stock_data[CLOSE].ewm(span=18, adjust=False).mean()
    stock_data[EMA40] = stock_data[CLOSE].ewm(span=40, adjust=False).mean()


def enrich_data(stock_data: DataFrame) -> None:
    set_moving_averages(stock_data=stock_data)
    set_rsi(stock_data=stock_data, period=RSI_PERIOD)
    set_macd(
        stock_data=stock_data,
        fast_period=MACD_FAST_PERIOD,
        slow_period=MACD_SLOW_PERIOD,
        signal_period=MACD_SIGNAL_PERIOD,
    )
    set_triple_cross(stock_data=stock_data, threshold=TRIPLE_CROSS_THRESHOLD)


logger = get_logger(name=__name__)


# Main method
def main():
    args = parse_arguments()

    # Disable yahoo finance logging
    ylogger = logging.getLogger("yfinance")
    ylogger.disabled = True
    ylogger.propagate = False

    try:
        ## Getting data from yahoo finance
        logger.info(msg=f"Downloading data for {args.ticker.upper()}...")
        start_time = datetime.now()
        stock_data: DataFrame = yf.download(tickers=args.ticker.upper(), period="max", interval="1d", progress=False)
        stock_data.drop_duplicates(inplace=True)
        stock_data.dropna(inplace=True)
        if stock_data.empty:
            raise ValueError(f"No data found for {args.ticker.upper()}")
        logger.debug(
            msg=f"Data downloaded in {get_timestamp_seconds(start_time=start_time)} seconds"
        )

        ## Enrich data with moving averages and indicators
        logger.info(msg="Enriching data...")
        start_time = datetime.now()
        enrich_data(stock_data=stock_data)
        logger.debug(
            msg=f"Data enriched in {get_timestamp_seconds(start_time=start_time)} seconds"
        )

        ## Print data info
        stock_data.info()

        ## Simulate prediction for the last year
        logger.info(msg="Simulating prediction for the last year...")
        start_time = datetime.now()

        # Create dataframe skipping the last year to let the ML model predict it
        stock_data_until_minus_days: DataFrame = stock_data.iloc[:-NUMBER_OF_PREDICTIONS_TO_COMPARE].dropna(subset=[SMA200, RSI])
        features: DataFrame = stock_data_until_minus_days[FEATURES]
        target: DataFrame = stock_data_until_minus_days[CLOSE]

        # Scaler as ML languages work better using lower values
        scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(X=features)
        scaled_target = scaler.fit_transform(X=target.values.reshape(-1, 1))

        # Split data into training and testing
        train_features, test_features, train_target, test_targets = train_test_split(scaled_features, scaled_target, train_size=0.8, shuffle=False)
        training_size = int(len(scaled_features) * 0.8)
        train_data = scaled_features[:training_size]
        test_data = scaled_features[training_size:]
        train_seq, train_label = create_lstm_sequence(dataset=train_data, time_step=60)
        test_seq, test_label = create_lstm_sequence(dataset=test_data, time_step=60)

        # Train the models
        random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
        xgb_regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=60, return_sequences=True, input_shape=(train_seq.shape[1], train_seq.shape[2])))
        lstm_model.add(LSTM(units=60, return_sequences=False))
        lstm_model.add(Dense(units=1))
        lstm_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
        lstm_model.summary()

        random_forest.fit(X=train_features, y=train_target.ravel())
        xgb_regressor.fit(X=train_features, y=train_target.ravel())
        lstm_model.fit(train_seq, train_label, epochs=80, validation_data=(test_seq, test_label), verbose=1)

        # Predict the values and reverse the scaling
        rf_predictions = scaler.inverse_transform(X=random_forest.predict(X=test_features[-NUMBER_OF_PREDICTIONS_TO_COMPARE:]).reshape(-1, 1))
        xgb_predictions = scaler.inverse_transform(X=xgb_regressor.predict(test_features[-NUMBER_OF_PREDICTIONS_TO_COMPARE:]).reshape(-1, 1))
        y_test_descaled = scaler.inverse_transform(X=test_targets[-NUMBER_OF_PREDICTIONS_TO_COMPARE:])

        lstm_predictions = scaler.inverse_transform(X=lstm_model.predict(test_seq).reshape(-1, 1))

        # Calculate the metrics and evaluate the models
        logger.debug(f"Random Forest MAE: {mean_absolute_error(y_true=y_test_descaled, y_pred=rf_predictions)}, "
                     f"R²: {r2_score(y_true=y_test_descaled, y_pred=rf_predictions)}")
        logger.debug(f"XGBoost MAE: {mean_absolute_error(y_true=y_test_descaled, y_pred=xgb_predictions)}, "
                     f"R²: {r2_score(y_true=y_test_descaled, y_pred=xgb_predictions)}")
        # logger.debug(f"LSTM MAE: {mean_absolute_error(y_true=y_test_descaled, y_pred=lstm_predictions)}, "
        #              f"R²: {r2_score(y_true=y_test_descaled, y_pred=lstm_predictions)}")
        logger.debug(msg=f"Predicting done in {get_timestamp_seconds(start_time=start_time)} seconds")

        # Assigning the predicted data to the original one to compare it
        stock_data_last_year = stock_data.iloc[-NUMBER_OF_PREDICTIONS_TO_COMPARE:]
        stock_data.loc[stock_data_last_year.index, CLOSE_PREDICTED_RF] = rf_predictions
        stock_data.loc[stock_data_last_year.index, CLOSE_PREDICTED_XGB] = xgb_predictions
        stock_data.loc[stock_data_last_year.index, 'ClosePredictedLSTM'] = lstm_predictions[-NUMBER_OF_PREDICTIONS_TO_COMPARE:]

        # Configuring plot charts
        # https://github.com/matplotlib/mplfinance
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
                data=stock_data[ASCENT_CROSS][stock_data[ASCENT_CROSS] != pandas.NA],
                type="scatter",
                markersize=TRIPLE_CROSS_MARKER_SIZE,
                marker="^",
                color=TRIPLE_CROSS_MARKER_COLOR),
            mpf.make_addplot(
                data=stock_data[DESCENT_CROSS][stock_data[DESCENT_CROSS] != pandas.NA],
                type="scatter",
                markersize=TRIPLE_CROSS_MARKER_SIZE,
                marker="v",
                color=TRIPLE_CROSS_MARKER_COLOR,
            ),
            mpf.make_addplot(
                data=stock_data[CLOSE_PREDICTED_RF],
                color='green',
                width=1,
                label=CLOSE_PREDICTED_RF,
            ),
            mpf.make_addplot(
                data=stock_data[CLOSE_PREDICTED_XGB],
                color='magenta',
                width=1,
                label=CLOSE_PREDICTED_XGB,
            ),
            mpf.make_addplot(
                data=stock_data['ClosePredictedLSTM'],
                color='cyan',
                width=1,
                label='ClosePredictedLSTM',
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
