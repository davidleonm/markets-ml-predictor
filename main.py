import argparse
import logging
import sys

import colorlog
import mplfinance as mpf
import numpy
import pandas
import yfinance as yf
from keras import Sequential, Input
from keras.src.layers import LSTM, Dense
from keras.src.legacy.preprocessing.sequence import TimeseriesGenerator
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# Charts configuration
EMA200_LINE_COLOR: str = "blue"
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
CLOSE_RF_XGB_COLOR: str = "magenta"
CLOSE_LSTM_COLOR: str = "cyan"

EMA200_LINE_WIDTH: float = 2
EMA4_LINE_WIDTH: float = 0.5
EMA18_LINE_WIDTH: float = 0.5
EMA40_LINE_WIDTH: float = 1
RSI_LINE_WIDTH: float = 1
RSI_OVERSOLD_LINE_WIDTH: float = 0.5
RSI_OVERBOUGHT_LINE_WIDTH: float = 0.5
MACD_LINE_WIDTH: float = 0.5
MACD_SIGNAL_LINE_WIDTH: float = 0.5
TRIPLE_CROSS_MARKER_SIZE: int = 100
CLOSE_RF_XGB_LINE_WIDTH: float = 1
CLOSE_LSTM_LINE_WIDTH: float = 1

RSI_PERIOD: int = 14
MACD_FAST_PERIOD: int = 14
MACD_SLOW_PERIOD: int = 28
MACD_SIGNAL_PERIOD: int = 9

TRIPLE_CROSS_THRESHOLD: float = 20

# Base columns
COLUMN_OPEN: str = "Open"
COLUMN_HIGH: str = "High"
COLUMN_LOW: str = "Low"
COLUMN_CLOSE: str = "Close"
COLUMN_VOLUME: str = "Volume"

# Moving average columns
COLUMN_EMA200: str = "EMA200"
COLUMN_EMA4: str = "EMA4"
COLUMN_EMA18: str = "EMA18"
COLUMN_EMA40: str = "EMA40"

# Indicator columns
COLUMN_RSI: str = "RSI"
COLUMN_MACD: str = "MACD"
COLUMN_MACD_SIGNAL: str = "MACD_SIGNAL"
COLUMN_MACD_HISTOGRAM: str = "MACD_HISTOGRAM"
COLUMN_CROSS: str = "CROSS"
COLUMN_ASCENT_CROSS: str = "ASCENT_CROSS"
COLUMN_DESCENT_CROSS: str = "DESCENT_CROSS"

# Columns for predictions
COLUMNS_FEATURES: list[str] = [COLUMN_OPEN, COLUMN_HIGH, COLUMN_LOW, COLUMN_VOLUME, COLUMN_EMA200, COLUMN_EMA4, COLUMN_EMA18,
                               COLUMN_EMA40, COLUMN_RSI, COLUMN_MACD,
                               COLUMN_MACD_SIGNAL, COLUMN_MACD_HISTOGRAM]
CLOSE_PREDICTED_RF_XGB: str = "ClosePredicted_RF_XGB"
CLOSE_PREDICTED_LSTM: str = "ClosePredicted_LSTM"

# Constants
N_ESTIMATORS: int = 100
RANDOM_STATE: int = 42
LEARNING_RATE: float = 0.1
TRAIN_SIZE: float = 0.8
DEFAULT_EPOCHS: int = 1
DROPOUT: float = 0.2
UNITS_1: int = 100
UNITS_2: int = 50
UNITS_3: int = 10
DENSE_UNITS: int = 1


# Helper methods
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


def check_simulation_days(value):
    inf_limit: int = 90
    sup_limit: int = 2000
    result = int(value)

    if result < inf_limit or result > sup_limit:
        raise argparse.ArgumentTypeError(f"simulation_days must be between {inf_limit} and {sup_limit}, got {result}")

    return result


def check_future_days(value):
    inf_limit: int = 7
    sup_limit: int = 365
    result = int(value)

    if result < inf_limit or result > sup_limit:
        raise argparse.ArgumentTypeError(f"future_days must be between {inf_limit} and {sup_limit}, got {result}")

    return result


def check_epoch_days(value):
    inf_limit: int = 1
    sup_limit: int = 100
    result = int(value)

    if result < inf_limit or result > sup_limit:
        raise argparse.ArgumentTypeError(f"epochs must be between {inf_limit} and {sup_limit}, got {result}")

    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML stock market price predictor")
    parser.add_argument(
        "--ticker", help="Ticker or stock symbol taken from https://finance.yahoo.com",
        type=str,
        required=True
    )
    parser.add_argument(
        "--simulation_days", help="If a simulation of a prediction for those days must be done",
        type=check_simulation_days,
        required=False
    )
    parser.add_argument(
        "--future_days", help="If a future prediction for those days must be done",
        type=check_future_days,
        required=False
    )

    parser.add_argument(
        "--epochs", help="Number of EPOCHS for the LSTM model",
        type=check_epoch_days,
        required=False,
        default=DEFAULT_EPOCHS
    )

    return parser.parse_args()


# Methods to add indicators and moving averages
def set_rsi(stock_data: DataFrame, period: int) -> None:
    delta = stock_data[COLUMN_CLOSE].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    stock_data[COLUMN_RSI] = 100 - (100 / (1 + rs))


def set_macd(
        stock_data: DataFrame, fast_period: int, slow_period: int, signal_period: int
) -> None:
    short_ema = stock_data[COLUMN_CLOSE].ewm(span=fast_period, adjust=False).mean()
    long_ema = stock_data[COLUMN_CLOSE].ewm(span=slow_period, adjust=False).mean()

    stock_data[COLUMN_MACD] = short_ema - long_ema
    stock_data[COLUMN_MACD_SIGNAL] = (
        stock_data[COLUMN_MACD].ewm(span=signal_period, adjust=False).mean()
    )
    stock_data[COLUMN_MACD_HISTOGRAM] = stock_data[COLUMN_MACD] - stock_data[COLUMN_MACD_SIGNAL]


def set_triple_cross(stock_data: DataFrame, threshold: float) -> None:
    cross_condition = ((numpy.abs(stock_data[COLUMN_EMA4] - stock_data[COLUMN_EMA18]) < threshold) &
                       (numpy.abs(stock_data[COLUMN_EMA4] - stock_data[COLUMN_EMA40]) < threshold) &
                       (numpy.abs(stock_data[COLUMN_EMA18] - stock_data[COLUMN_EMA40]) < threshold))
    stock_data[COLUMN_CROSS] = stock_data[COLUMN_EMA4].where(cross_condition)

    ascent_cross_condition = (stock_data[COLUMN_EMA4] < stock_data[COLUMN_EMA40]) & (stock_data[COLUMN_EMA18] < stock_data[COLUMN_EMA40])
    stock_data[COLUMN_ASCENT_CROSS] = stock_data[COLUMN_CROSS].where(ascent_cross_condition)

    descent_cross_condition = (stock_data[COLUMN_EMA4] > stock_data[COLUMN_EMA40]) & (stock_data[COLUMN_EMA18] > stock_data[COLUMN_EMA40])
    stock_data[COLUMN_DESCENT_CROSS] = stock_data[COLUMN_CROSS].where(descent_cross_condition)


def set_moving_averages(stock_data: DataFrame) -> None:
    stock_data[COLUMN_EMA4] = stock_data[COLUMN_CLOSE].ewm(span=4, adjust=False).mean()
    stock_data[COLUMN_EMA18] = stock_data[COLUMN_CLOSE].ewm(span=18, adjust=False).mean()
    stock_data[COLUMN_EMA40] = stock_data[COLUMN_CLOSE].ewm(span=40, adjust=False).mean()
    stock_data[COLUMN_EMA200] = stock_data[COLUMN_CLOSE].ewm(span=200, adjust=False).mean()


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
    stock_data.dropna(subset=[COLUMN_EMA200, COLUMN_RSI], inplace=True)


logger = get_logger(name=__name__)


def get_lstm_model(input_shape: tuple) -> Sequential:
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=UNITS_1, activation="relu", return_sequences=True))
    model.add(LSTM(units=UNITS_2, activation="relu", return_sequences=True))
    model.add(LSTM(units=UNITS_3, activation="relu", return_sequences=False))
    model.add(Dense(units=DENSE_UNITS))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_error"])
    model.summary()

    return model


def get_scaler_and_scaled_values(stock_data: DataFrame) -> tuple[MinMaxScaler, DataFrame, DataFrame]:
    # Scaler as ML languages work better using lower values
    scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(X=stock_data[COLUMNS_FEATURES])
    scaled_target = scaler.fit_transform(X=stock_data[COLUMN_CLOSE].values.reshape(-1, 1))

    return scaler, scaled_features, scaled_target


def get_generator(num_days: int, scaled_features: DataFrame, scaled_target: DataFrame) -> TimeseriesGenerator:
    # Create a generator for the LSTM model
    return TimeseriesGenerator(data=scaled_features,
                               targets=scaled_target,
                               length=num_days,
                               batch_size=1)


# Main method
def simulate_prediction(stock_data: DataFrame, num_days: int, epochs: int) -> None:
    logger.info(msg="Simulating prediction")

    # Scaler as ML languages work better using lower values
    scaler, scaled_features, scaled_target = get_scaler_and_scaled_values(stock_data=stock_data)

    # Split data into training and testing
    train_features, test_features, train_target, test_targets = train_test_split(scaled_features, scaled_target,
                                                                                 train_size=TRAIN_SIZE,
                                                                                 shuffle=False)
    # Create a generator for the LSTM model
    generator: TimeseriesGenerator = get_generator(num_days=num_days, scaled_features=scaled_features, scaled_target=scaled_target)

    # Train the models
    random_forest = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    xgb_regressor = XGBRegressor(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, random_state=RANDOM_STATE)
    lstm_model = get_lstm_model(input_shape=(num_days, scaled_features.shape[1]))

    random_forest.fit(X=train_features, y=train_target.ravel())
    xgb_regressor.fit(X=train_features, y=train_target.ravel())
    lstm_model.fit(generator, epochs=epochs)

    # Predict the values and reverse the scaling
    rf_predictions = scaler.inverse_transform(X=random_forest.predict(X=test_features[-num_days:]).reshape(-1, 1))
    xgb_predictions = scaler.inverse_transform(X=xgb_regressor.predict(test_features[-num_days:]).reshape(-1, 1))
    lstm_predictions = scaler.inverse_transform(X=lstm_model.predict(x=generator).reshape(-1, 1))

    # Assigning the predicted data to the original one to compare it
    stock_data_to_date = stock_data.iloc[-num_days:]
    stock_data.loc[stock_data_to_date.index, CLOSE_PREDICTED_RF_XGB] = numpy.mean([rf_predictions, xgb_predictions], axis=0)
    stock_data.loc[stock_data_to_date.index, CLOSE_PREDICTED_LSTM] = lstm_predictions[-num_days:]


def get_future_predictions(stock_data: DataFrame, num_days: int, epochs: int) -> DataFrame:
    # Calculate future prediction
    logger.info(msg="Predicting future values to make you rich $$$ :)")

    # Scaler as ML languages work better using lower values
    scaler, scaled_features, scaled_target = get_scaler_and_scaled_values(stock_data=stock_data)

    # Create a generator for the LSTM model
    generator: TimeseriesGenerator = get_generator(num_days=num_days, scaled_features=scaled_features, scaled_target=scaled_target)

    # Train the LSTM model
    lstm_model = get_lstm_model(input_shape=(num_days, scaled_features.shape[1]))
    lstm_model.fit(generator, epochs=epochs)

    # Prepare data for future predictions
    future_predictions = []
    current_batch = scaled_features[-num_days:].reshape((1, num_days, scaled_features.shape[1]))
    for _ in range(num_days):
        pred = lstm_model.predict(current_batch)[0]
        future_predictions.append(pred)
        current_batch = numpy.append(current_batch[:, 1:, :], scaled_features[-num_days + 1 + _].reshape((1, 1, scaled_features.shape[1])),
                                     axis=1)

    # Assigning the predicted data to the original one to review it
    future_dates = pandas.date_range(start=stock_data.index[-1], periods=num_days, freq="B")
    return pandas.DataFrame(data=scaler.inverse_transform(numpy.array(future_predictions).reshape(-1, 1)),
                            index=future_dates,
                            columns=[CLOSE_PREDICTED_LSTM])


def main():
    args = parse_arguments()

    # Disable yahoo finance logging
    ylogger = logging.getLogger("yfinance")
    ylogger.disabled = True
    ylogger.propagate = False

    try:
        # Getting data from yahoo finance
        logger.info(msg=f"Downloading data for {args.ticker.upper()}...")
        stock_data: DataFrame = yf.download(tickers=args.ticker.upper(), period="max", interval="1d", progress=False)
        stock_data.drop_duplicates(inplace=True)
        stock_data.dropna(inplace=True)
        if stock_data.empty:
            raise ValueError(f"No data found for {args.ticker.upper()}")

        # Enrich data with moving averages and indicators
        logger.info(msg="Enriching data...")
        enrich_data(stock_data=stock_data)

        # Print data info
        stock_data.info()

        # Simulation of a prediction for the last X days to check the algorithm
        if args.simulation_days is not None:
            simulate_prediction(stock_data=stock_data, num_days=args.simulation_days, epochs=args.epochs)

        # Future prediction for the next X days to make you rich $$$
        if args.future_days is not None:
            future_predictions = get_future_predictions(stock_data=stock_data, num_days=args.future_days, epochs=args.epochs)
            stock_data = pandas.concat([stock_data, future_predictions], axis=0)

        # Configuring plot charts
        # https://github.com/matplotlib/mplfinance
        more_plots = [
            mpf.make_addplot(
                data=stock_data[COLUMN_EMA200],
                color=EMA200_LINE_COLOR,
                width=EMA200_LINE_WIDTH,
                label=COLUMN_EMA200,
            ),
            mpf.make_addplot(
                data=stock_data[COLUMN_EMA4],
                color=EMA4_LINE_COLOR,
                width=EMA4_LINE_WIDTH,
                label=COLUMN_EMA4,
                linestyle="dotted",
            ),
            mpf.make_addplot(
                data=stock_data[COLUMN_EMA18],
                color=EMA18_LINE_COLOR,
                width=EMA18_LINE_WIDTH,
                label=COLUMN_EMA18,
                linestyle="dotted",
            ),
            mpf.make_addplot(
                data=stock_data[COLUMN_EMA40],
                color=EMA40_LINE_COLOR,
                width=EMA40_LINE_WIDTH,
                label=COLUMN_EMA40,
            ),
            mpf.make_addplot(
                data=stock_data[COLUMN_ASCENT_CROSS][stock_data[COLUMN_ASCENT_CROSS] != pandas.NA],
                type="scatter",
                markersize=TRIPLE_CROSS_MARKER_SIZE,
                marker="^",
                color=TRIPLE_CROSS_MARKER_COLOR),
            mpf.make_addplot(
                data=stock_data[COLUMN_DESCENT_CROSS][stock_data[COLUMN_DESCENT_CROSS] != pandas.NA],
                type="scatter",
                markersize=TRIPLE_CROSS_MARKER_SIZE,
                marker="v",
                color=TRIPLE_CROSS_MARKER_COLOR,
            ),
            mpf.make_addplot(
                data=stock_data[COLUMN_RSI],
                panel=1,
                color=RSI_LINE_COLOR,
                width=RSI_LINE_WIDTH,
                ylabel=COLUMN_RSI,
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
                data=stock_data[COLUMN_MACD],
                panel=2,
                color=MACD_LINE_COLOR,
                ylabel=COLUMN_MACD,
                width=MACD_LINE_WIDTH,
            ),
            mpf.make_addplot(
                data=stock_data[COLUMN_MACD_SIGNAL],
                panel=2,
                color=MACD_SIGNAL_LINE_COLOR,
                width=MACD_SIGNAL_LINE_WIDTH,
            ),
            mpf.make_addplot(
                data=stock_data[COLUMN_MACD_HISTOGRAM],
                panel=2,
                type="bar",
                color=MACD_HISTOGRAM_LINE_COLOR,
                alpha=0.5,
            ),
        ]

        # Add plots for predictions simulation and future predictions
        if args.simulation_days is not None:
            more_plots.append(
                mpf.make_addplot(
                    data=stock_data[CLOSE_PREDICTED_RF_XGB],
                    color=CLOSE_RF_XGB_COLOR,
                    width=CLOSE_RF_XGB_LINE_WIDTH,
                    ylabel=CLOSE_PREDICTED_RF_XGB,
                ),
            )
            more_plots.append(
                mpf.make_addplot(
                    data=stock_data[CLOSE_PREDICTED_LSTM],
                    color=CLOSE_LSTM_COLOR,
                    width=CLOSE_LSTM_LINE_WIDTH,
                    ylabel=CLOSE_PREDICTED_LSTM,
                ),
            )

        if args.simulation_days is None and args.future_days is not None:
            more_plots.append(
                mpf.make_addplot(
                    data=stock_data[CLOSE_PREDICTED_LSTM],
                    color=CLOSE_LSTM_COLOR,
                    width=CLOSE_LSTM_LINE_WIDTH,
                    ylabel=CLOSE_PREDICTED_LSTM,
                ),
            )

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
