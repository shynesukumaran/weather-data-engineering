import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


class WeatherAPIClient:
    """
    Client for fetching weather data from Open-Meteo APIs (historical and forecast)
    """

    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude
        self._setup_client()

    def _setup_client(self):
        cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        print("API_Client: Open-Meteo client initialized with cache and retry.")

    def _process_hourly_response(
        self, response, expected_hourly_vars: list[str]
    ) -> pd.DataFrame:
        hourly = response.Hourly()
        if not hourly:
            print("API_Client Warning: No hourly data in response.")
            return pd.DataFrame()

        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
        }

        for i, var_name in enumerate(expected_hourly_vars):
            hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()

        df = pd.DataFrame(data=hourly_data)
        return df

    def fetch_historical_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical weather data from Open-Meteo Archive API.
        """
        url = "https://archive-api.open-meteo.com/v1/archive"
        hourly_vars = [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "cloud_cover",
            "precipitation",
        ]
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": hourly_vars,
        }
        try:
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]
            print(
                f"API_Client: Fetched Historical Data from {response.Latitude()}°N {response.Longitude()}°E"
            )
            df = self._process_hourly_response(response, hourly_vars)
            return df
        except Exception as e:
            print(f"API_Client Error: Fetching historical data: {e}")
            return pd.DataFrame()

    def fetch_forecast_data(self, hours: int = 24) -> pd.DataFrame:
        """
        Fetches the next X hours of weather forecast data from Open-Meteo Forecast API.
        """
        url = "https://api.open-meteo.com/v1/forecast"
        hourly_vars = [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "cloud_cover",
            "precipitation",
        ]
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": hourly_vars,
            "forecast_days": int(hours / 24) + (1 if hours % 24 > 0 else 0),
            "timezone": "auto",
        }
        try:
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]
            print(
                f"API_Client: Fetched API Forecast Data from {response.Latitude()}°N {response.Longitude()}°E"
            )
            forecast_dataframe = self._process_hourly_response(response, hourly_vars)

            if len(forecast_dataframe) > hours:
                forecast_dataframe = forecast_dataframe.head(hours)

            forecast_dataframe = forecast_dataframe.rename(
                columns={
                    col: f"{col}_forecast"
                    for col in forecast_dataframe.columns
                    if col != "date"
                }
            )
            print(
                f"API_Client: Fetched {len(forecast_dataframe)} hours of API forecast data."
            )
            return forecast_dataframe

        except Exception as e:
            print(f"API_Client Error: Fetching API forecast: {e}")
            return pd.DataFrame()
