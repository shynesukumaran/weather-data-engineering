import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# We'll need a way to load historical data for training; assuming DatabaseManager is available
# (loaded via pipeline_orchestrator, or pass in dataframe directly)


class WeatherForecaster:
    """
    Manages training and prediction of weather features using ML models.
    """

    def __init__(self, features_to_predict: list[str], lags: int = 24):
        self.features_to_predict = features_to_predict
        self.lags = lags
        self.trained_models = {}  # To store a model for each feature

    def _create_time_series_features(
        self, df: pd.DataFrame, feature: str
    ) -> pd.DataFrame:
        """
        Creates time series features for a given feature DataFrame.
        """
        if df.empty or feature not in df.columns:
            print(
                f"Forecaster Warning: Empty or invalid DataFrame for feature '{feature}'."
            )
            return pd.DataFrame()

        df_features = df[[feature]].copy()

        # Add lagged features
        for lag in range(1, self.lags + 1):
            df_features[f"{feature}_lag_{lag}"] = df_features[feature].shift(lag)

        # Add time-based features
        df_features["hour_of_day"] = df_features.index.hour
        df_features["day_of_week"] = df_features.index.dayofweek
        df_features["day_of_year"] = df_features.index.dayofyear

        # Add rolling statistics (handle constant series)
        if not df_features[feature].isna().all():
            df_features[f"{feature}_roll_mean"] = (
                df_features[feature].rolling(window=6, min_periods=1).mean()
            )
            df_features[f"{feature}_roll_std"] = (
                df_features[feature]
                .rolling(window=6, min_periods=1)
                .std()
                .fillna(0)  # Fill NaN std for constant series
            )
        else:
            df_features[f"{feature}_roll_mean"] = df_features[feature]
            df_features[f"{feature}_roll_std"] = 0

        # Debug: Check for NaN or constant values
        if df_features[feature].nunique() == 1:
            print(
                f"Forecaster Warning: Feature '{feature}' is constant: {df_features[feature].iloc[0]}"
            )

        # Drop rows with NaN values
        df_features = df_features.dropna()

        if df_features.empty:
            print(
                f"Forecaster Warning: All features for '{feature}' are NaN after processing."
            )
            return pd.DataFrame()

        return df_features

    def train_models(self, historical_df: pd.DataFrame):
        """
        Trains an ML model for each specified weather feature.
        """
        if historical_df.empty:
            print("Forecaster: No historical data provided for training.")
            return

        # Ensure historical_df has 'date' as index for time series features
        if "date" not in historical_df.index.names:
            historical_df = historical_df.set_index("date").sort_index()

        for feature in self.features_to_predict:
            print(f"Forecaster: Training model for '{feature}'...")
            if feature not in historical_df.columns:
                print(
                    f"Forecaster Warning: Feature '{feature}' not found in historical data. Skipping training."
                )
                continue

            df_with_features = self._create_time_series_features(
                historical_df[[feature]], feature
            )

            if (
                len(df_with_features) < 2 * self.lags
            ):  # Need enough data for lags + train/test
                print(
                    f"Forecaster Warning: Not enough historical data for '{feature}' after creating lags. Skipping training."
                )
                continue

            X = df_with_features.drop(columns=[feature])
            y = df_with_features[feature]

            # Use a time-based split for robustness
            split_point = int(len(X) * 0.8)  # Train on 80% of data
            X_train, y_train = X[:split_point], y[:split_point]

            # Using RandomForestRegressor as the chosen model for demonstration
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            self.trained_models[feature] = model
            print(f"Forecaster: Model trained for '{feature}'.")

            # Optional: Evaluate on a small test set for demo
            if split_point < len(X):
                X_test, y_test = X[split_point:], y[split_point:]
                y_pred_test = model.predict(X_test)
                print(
                    f"Forecaster: Test MAE for {feature}: {mean_absolute_error(y_test, y_pred_test):.2f}"
                )
                print(
                    f"Forecaster: Test R2 for {feature}: {r2_score(y_test, y_pred_test):.2f}"
                )

    def predict_future(
        self, historical_df: pd.DataFrame, hours_to_predict: int = 24
    ) -> pd.DataFrame:
        """
        Generates future predictions for the next X hours using trained models.
        """
        if not self.trained_models:
            print("Forecaster: No models trained. Call .train_models() first.")
            return pd.DataFrame()
        if historical_df.empty:
            print("Forecaster: No historical data provided for prediction seeding.")
            return pd.DataFrame()

        # Ensure historical_df has 'date' as index
        if "date" not in historical_df.index.names:
            historical_df = historical_df.set_index("date").sort_index()

        last_historical_timestamp = historical_df.index.max()
        if pd.isna(last_historical_timestamp):
            print(
                "Forecaster: No valid timestamp found in historical data for prediction."
            )
            return pd.DataFrame()

        future_timestamps = pd.date_range(
            start=last_historical_timestamp + pd.Timedelta(hours=1),
            periods=hours_to_predict,
            freq="H",
            tz="UTC",
        )
        predictions_df = pd.DataFrame(index=future_timestamps)
        predictions_df.index.name = "date"

        for feature in self.features_to_predict:
            model = self.trained_models.get(feature)
            if not model:
                print(
                    f"Forecaster Warning: No trained model found for '{feature}'. Skipping prediction."
                )
                predictions_df[f"{feature}_ml_pred"] = np.nan
                continue

            current_feature_series = historical_df[feature].iloc[-self.lags :].copy()
            if len(current_feature_series) < self.lags:
                print(
                    f"Forecaster Warning: Only {len(current_feature_series)} rows available for '{feature}', need {self.lags}."
                )
                predictions_df[f"{feature}_ml_pred"] = np.nan
                continue
            if current_feature_series.isna().all():
                print(
                    f"Forecaster Warning: All values for '{feature}' in last {self.lags} rows are NaN."
                )
                predictions_df[f"{feature}_ml_pred"] = np.nan
                continue

            # Debug: Print the series to inspect
            print(f"Last {self.lags} rows for {feature}:\n{current_feature_series}")

            initial_features_df = self._create_time_series_features(
                current_feature_series.to_frame(name=feature), feature
            )
            if initial_features_df.empty:
                print(
                    f"Forecaster Warning: Could not create initial features for '{feature}' after processing."
                )
                predictions_df[f"{feature}_ml_pred"] = np.nan
                continue

            next_features_input = (
                initial_features_df.drop(columns=[feature]).iloc[[-1]].copy()
            )

            predicted_values = []
            for i in range(hours_to_predict):
                next_pred = model.predict(next_features_input)[0]
                predicted_values.append(next_pred)

                for j in range(self.lags, 1, -1):
                    next_features_input.iloc[
                        0, next_features_input.columns.get_loc(f"{feature}_lag_{j}")
                    ] = next_features_input.iloc[
                        0, next_features_input.columns.get_loc(f"{feature}_lag_{j-1}")
                    ]
                next_features_input.iloc[
                    0, next_features_input.columns.get_loc(f"{feature}_lag_1")
                ] = next_pred

                next_features_input["hour_of_day"] = (
                    next_features_input["hour_of_day"] + 1
                ) % 24
                if next_features_input["hour_of_day"].iloc[0] == 0:
                    next_features_input["day_of_week"] = (
                        next_features_input["day_of_week"] + 1
                    ) % 7
                    next_features_input["day_of_year"] = (
                        next_features_input["day_of_year"] + 1
                    )

            predictions_df[f"{feature}_ml_pred"] = predicted_values

        return predictions_df
