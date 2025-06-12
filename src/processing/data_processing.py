import pandas as pd


def process_weather_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process the fetched weather data.

    Args:
        data (pd.DataFrame): The raw weather data.

    Returns:
        pd.DataFrame: Processed weather data.
    """
    if data.empty:
        print("DataProcessor: Input raw weather data is empty.")
        return pd.DataFrame()

    if "date" not in data.columns:
        print("DataProcessor Error: 'date' column not found in raw weather data.")
        return pd.DataFrame()

    print(f"Raw weather data shape: {data.shape}")
    print(f"Raw weather data head:\n{data.head()}")
    print(f"Raw weather data tail:\n{data.tail()}")

    processed_data = data.copy()
    processed_data["date"] = pd.to_datetime(processed_data["date"], utc=True)
    processed_data.set_index("date", inplace=True)
    processed_data.sort_index(inplace=True)

    recent_data = processed_data.last("24H")
    for col in processed_data.columns:
        if recent_data[col].nunique() <= 1:
            print(
                f"DataProcessor Warning: Column '{col}' has constant value {recent_data[col].unique()} in last 24 hours."
            )

    valid_data = processed_data.loc[processed_data.index < processed_data.index[-24]]
    if not valid_data.empty:
        valid_data = valid_data.ffill()
        processed_data.update(valid_data)
    else:
        processed_data = processed_data.ffill()

    print(f"Processed weather data shape: {processed_data.shape}")
    print(f"Processed weather data head:\n{processed_data.head()}")
    print(f"Processed weather data tail:\n{processed_data.tail()}")

    return processed_data
