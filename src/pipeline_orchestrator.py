# src/pipeline_orchestrator.py (Adjusted Date Logic)

import datetime
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))  # src/
project_root = os.path.dirname(project_root)  # data-engineering-project/
sys.path.insert(0, project_root)

from src.data_extract.weather_api_client import WeatherAPIClient
from src.db.database_manager import DatabaseManager
from src.ml.weather_forecaster import WeatherForecaster
from src.processing.data_processing import process_weather_data


def run_daily_pipeline():
    """
    Orchestrates the daily data engineering pipeline:
    1. Fetches historical weather data.
    2. Processes raw data.
    3. Stores processed data to DB.
    4. Trains ML models and generates predictions.
    5. Fetches API forecast.
    6. Stores ML predictions and API forecast to DB.
    """
    current_time = datetime.datetime.now(datetime.timezone.utc)
    print(f"[{current_time}] Starting daily weather data pipeline...")

    # Initialize Managers/Clients
    db_manager = DatabaseManager()
    # Assuming Magdeburg coordinates for now (52.1132, 11.6081)
    api_client = WeatherAPIClient(latitude=52.1132, longitude=11.6081)
    forecaster = WeatherForecaster(
        features_to_predict=[
            "temperature_2m",
            "relative_humidity_2m",
            "cloud_cover",
            "wind_speed_10m",
            "precipitation",
        ],
        lags=24,  # Example: use last 24 hours for features
    )

    try:
        # --- Dynamic Date Range for Historical Data ---
        today = datetime.date.today()
        # Fetch historical data up to yesterday
        historical_end_date = today - datetime.timedelta(days=1)
        # Fetch data for, say, the last 3-5 years. Adjust as needed for your model training.
        historical_start_date = historical_end_date - datetime.timedelta(
            days=365 * 3 + 180
        )  # Approx 3.5 years back

        historical_start_date_str = historical_start_date.strftime("%Y-%m-%d")
        historical_end_date_str = historical_end_date.strftime("%Y-%m-%d")

        # Step 1: Fetch historical weather data
        print(
            f"[{current_time}] Fetching historical weather data from {historical_start_date_str} to {historical_end_date_str}..."
        )
        historical_data_df = api_client.fetch_historical_data(
            start_date=historical_start_date_str, end_date=historical_end_date_str
        )

        if historical_data_df is not None and not historical_data_df.empty:
            # Step 2: Process the fetched data
            print(f"[{current_time}] Processing historical weather data...")
            processed_data_df = process_weather_data(historical_data_df)

            if processed_data_df is not None and not processed_data_df.empty:
                # Step 3: Store processed historical data into the database
                print(
                    f"[{current_time}] Storing processed historical data to DB (table: weather_data)..."
                )
                db_manager.store_dataframe(
                    processed_data_df, "weather_data", if_exists="replace"
                )

                # Step 4: Train ML models and generate predictions
                print(
                    f"[{current_time}] Training ML models and generating predictions for next 24 hours..."
                )
                forecaster.train_models(processed_data_df)
                ml_predictions_df = forecaster.predict_future(
                    processed_data_df, hours_to_predict=24
                )

                if ml_predictions_df is not None and not ml_predictions_df.empty:
                    print(
                        f"[{current_time}] Storing ML predictions to DB (table: weather_predictions)..."
                    )
                    db_manager.store_dataframe(
                        ml_predictions_df,
                        "weather_predictions",
                        if_exists="replace",
                        add_timestamp_col="prediction_generated_at",
                    )
                else:
                    print(
                        f"[{current_time}] No ML predictions generated or DataFrame is empty."
                    )

                # Step 5: Fetch next 24-hour API forecast
                print(f"[{current_time}] Fetching API forecast for next 24 hours...")
                api_forecast_df = api_client.fetch_forecast_data(hours=24)

                if api_forecast_df is not None and not api_forecast_df.empty:
                    print(
                        f"[{current_time}] Storing API forecast to DB (table: weather_api_forecasts)..."
                    )
                    db_manager.store_dataframe(
                        api_forecast_df,
                        "weather_api_forecasts",
                        if_exists="replace",
                        add_timestamp_col="forecast_fetched_at",
                    )
                else:
                    print(
                        f"[{current_time}] No API forecast data fetched or DataFrame is empty."
                    )

                print(f"[{current_time}] Daily pipeline completed successfully.")
            else:
                print(
                    f"[{current_time}] Processed data is empty or None. Skipping ML and DB storage."
                )
        else:
            print(
                f"[{current_time}] Historical data fetching failed or returned empty. Skipping processing and further steps."
            )

    except Exception as e:
        print(
            f"[{current_time}] An error occurred during the daily pipeline: {e}",
            file=sys.stderr,
        )

    finally:
        # Dispose the database engine once the pipeline is finished
        if db_manager._engine:  # Access the private attribute for disposal
            db_manager._dispose_engine()
            print(f"[{current_time}] Database engine disposed.")


if __name__ == "__main__":
    run_daily_pipeline()
