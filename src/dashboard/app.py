import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

project_root = os.path.dirname(os.path.abspath(__file__))  # src/dashboard/
project_root = os.path.dirname(project_root)  # src/
project_root = os.path.dirname(project_root)  # data-engineering-project/
sys.path.insert(0, project_root)
from src.db.database_manager import DatabaseManager

# --- Dash Application ---
app = Dash(__name__)

# Initialize DatabaseManager
db_manager = DatabaseManager()

# Load all necessary data when the app starts
historical_df = db_manager.load_dataframe(table_name="weather_data")
ml_predictions_df = db_manager.load_dataframe(table_name="weather_predictions")
api_forecast_df = db_manager.load_dataframe(table_name="weather_api_forecasts")

# Ensure historical_df["date"] is timezone-aware UTC
if not historical_df.empty:
    historical_df["date"] = pd.to_datetime(historical_df["date"], utc=True)

# Set default date range (last 7 days)
last_historical_date = historical_df["date"].max() if not historical_df.empty else None
default_end_date = last_historical_date
default_start_date = (
    last_historical_date - pd.Timedelta(days=7) if last_historical_date else None
)


def prepare_combined_data(start_date, end_date):
    """Prepare combined DataFrame for plotting based on date range."""
    # Convert inputs to timezone-aware UTC
    start_date = pd.to_datetime(start_date, utc=True)
    end_date = pd.to_datetime(end_date, utc=True)

    if last_historical_date:
        recent_historical_df = historical_df[
            (historical_df["date"] >= start_date) & (historical_df["date"] <= end_date)
        ].copy()
        print(f"Dashboard: Showing historical data from {start_date} to {end_date}")
    else:
        recent_historical_df = pd.DataFrame()
        print("Dashboard: No historical data to display.")

    # Prepare ML predictions
    if not ml_predictions_df.empty:
        ml_predictions_df_renamed = ml_predictions_df.rename(
            columns={
                col: col.replace("_ml_pred", "")
                for col in ml_predictions_df.columns
                if "_ml_pred" in col
            }
        )
        ml_predictions_df_renamed["date"] = pd.to_datetime(
            ml_predictions_df_renamed["date"], utc=True
        )
        ml_predictions_df_renamed["data_source"] = "ML Model Forecast"
        ml_predictions_forecast = ml_predictions_df_renamed[
            (ml_predictions_df_renamed["date"] > last_historical_date)
            & (ml_predictions_df_renamed["date"] <= end_date)
        ].copy()
    else:
        ml_predictions_forecast = pd.DataFrame()
        print("Dashboard: No ML predictions to display.")

    # Prepare API forecasts
    if not api_forecast_df.empty:
        api_forecast_df_renamed = api_forecast_df.rename(
            columns={
                col: col.replace("_forecast", "")
                for col in api_forecast_df.columns
                if "_forecast" in col
            }
        )
        api_forecast_df_renamed["date"] = pd.to_datetime(
            api_forecast_df_renamed["date"], utc=True
        )
        api_forecast_df_renamed["data_source"] = "API Forecast"
        api_forecast_only = api_forecast_df_renamed[
            (api_forecast_df_renamed["date"] > last_historical_date)
            & (api_forecast_df_renamed["date"] <= end_date)
        ].copy()
    else:
        api_forecast_only = pd.DataFrame()
        print("Dashboard: No API forecasts to display.")

    # Prepare historical data
    if not recent_historical_df.empty:
        recent_historical_df["data_source"] = "Historical Actuals"
        recent_historical_df = recent_historical_df[
            [
                "date",
                "temperature_2m",
                "relative_humidity_2m",
                "cloud_cover",
                "wind_speed_10m",
                "precipitation",
                "data_source",
            ]
        ]

    # Combine data
    combined_df = pd.concat(
        [
            recent_historical_df,
            ml_predictions_forecast[
                [
                    "date",
                    "temperature_2m",
                    "relative_humidity_2m",
                    "cloud_cover",
                    "wind_speed_10m",
                    "precipitation",
                    "data_source",
                ]
            ],
            api_forecast_only[
                [
                    "date",
                    "temperature_2m",
                    "relative_humidity_2m",
                    "cloud_cover",
                    "wind_speed_10m",
                    "precipitation",
                    "data_source",
                ]
            ],
        ]
    ).sort_values("date")

    return combined_df


def create_weather_plot(df: pd.DataFrame, y_col: str, title: str, y_label: str):
    if df.empty or y_col not in df.columns:
        return go.Figure()

    fig = px.line(
        df,
        x="date",
        y=y_col,
        color="data_source",
        title=title,
        labels={y_col: y_label, "date": "Date", "data_source": "Data Type"},
        template="plotly_white",
    )

    if last_historical_date:
        x_value = int(last_historical_date.timestamp() * 1000)
        fig.add_vline(
            x=x_value,
            line_width=1,
            line_dash="dash",
            line_color="gray",
            annotation_text="Historical/Forecast Split",
            annotation_position="top right",
        )

    return fig


def create_contour_plot(df: pd.DataFrame):
    if df.empty or "temperature_2m" not in df.columns:
        print("No temperature data available for contour plot.")
        return go.Figure()

    # Ensure date is datetime
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # Extract day and hour
    df["day"] = df["date"].dt.date
    df["hour"] = df["date"].dt.hour

    # Pivot to create 2D grid: rows = hours, columns = days
    pivot_df = df.pivot_table(
        values="temperature_2m", index="hour", columns="day", aggfunc="mean"
    )

    # Handle missing values
    pivot_df = pivot_df.ffill(axis=1).bfill(axis=1)

    # Get x, y, z for contour
    x = pivot_df.columns  # Days
    y = pivot_df.index  # Hours
    z = pivot_df.values  # Temperatures

    # Create contour plot
    fig = go.Figure(
        data=go.Contour(
            z=z,
            x=x,
            y=y,
            colorscale="Viridis",
            contours=dict(
                coloring="heatmap",
                showlabels=True,
                labelfont=dict(size=10, color="white"),
                start=np.nanmin(z) if not np.isnan(z).all() else 0,
                end=np.nanmax(z) if not np.isnan(z).all() else 10,
                size=(np.nanmax(z) - np.nanmin(z)) / 10 if not np.isnan(z).all() else 1,
            ),
            colorbar=dict(
                title="Temperature (°C)",
                titleside="right",
            ),
            line_smoothing=0.85,
            line_width=0.5,
        )
    )

    fig.update_layout(
        title="Temperature Contour Plot",
        xaxis_title="Date",
        yaxis_title="Hour of Day",
        xaxis=dict(
            tickangle=45,
            tickformat="%Y-%m-%d",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(0, 24, 3)),
        ),
        height=500,
        template="plotly_white",
    )

    return fig


# Initial combined data
combined_df = prepare_combined_data(default_start_date, default_end_date)

# Initial plots
temp_fig = create_weather_plot(
    combined_df,
    "temperature_2m",
    "Temperature: Actuals vs. Forecasts",
    "Temperature (°C)",
)
humidity_fig = create_weather_plot(
    combined_df,
    "relative_humidity_2m",
    "Relative Humidity: Actuals vs. Forecasts",
    "Relative Humidity (%)",
)
wind_fig = create_weather_plot(
    combined_df,
    "wind_speed_10m",
    "Wind Speed: Actuals vs. Forecasts",
    "Wind Speed (m/s)",
)
precipitation_fig = create_weather_plot(
    combined_df,
    "precipitation",
    "Precipitation: Actuals vs. Forecasts",
    "Precipitation (mm)",
)
cloud_fig = create_weather_plot(
    combined_df, "cloud_cover", "Cloud Cover: Actuals vs. Forecasts", "Cloud Cover (%)"
)
contour_fig = create_contour_plot(
    historical_df[historical_df["date"] >= default_start_date]
)

# App layout
app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "padding": "20px",
        "backgroundColor": "#f8f9fa",
    },
    children=[
        html.H1(
            "Weather Data Dashboard (Actuals & Forecasts)",
            style={"textAlign": "center", "color": "#343a40", "marginBottom": "30px"},
        ),
        # Date picker
        html.Div(
            style={
                "backgroundColor": "white",
                "padding": "15px",
                "borderRadius": "8px",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
                "marginBottom": "20px",
                "textAlign": "center",
            },
            children=[
                html.Label(
                    "Select Date Range:",
                    style={
                        "fontWeight": "bold",
                        "color": "#495057",
                        "marginRight": "10px",
                    },
                ),
                dcc.DatePickerRange(
                    id="date-picker-range",
                    min_date_allowed=(
                        historical_df["date"].min() if not historical_df.empty else None
                    ),
                    max_date_allowed=(
                        ml_predictions_df["date"].max()
                        if not ml_predictions_df.empty
                        else None
                    ),
                    initial_visible_month=default_end_date,
                    start_date=default_start_date,
                    end_date=default_end_date,
                    display_format="YYYY-MM-DD",
                    style={"margin": "10px"},
                ),
            ],
        ),
        # Contour plot
        html.Div(
            style={
                "backgroundColor": "white",
                "padding": "25px",
                "borderRadius": "8px",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
                "marginBottom": "20px",
            },
            children=[
                html.H2(
                    "Temperature Contour Plot",
                    style={
                        "textAlign": "left",
                        "color": "#495057",
                        "marginBottom": "15px",
                    },
                ),
                dcc.Graph(
                    id="contour-plot", figure=contour_fig, style={"height": "50vh"}
                ),
            ],
        ),
        # Existing plots
        html.Div(
            style={
                "backgroundColor": "white",
                "padding": "25px",
                "borderRadius": "8px",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
                "marginBottom": "20px",
            },
            children=[
                html.H2(
                    "Temperature Trends",
                    style={
                        "textAlign": "left",
                        "color": "#495057",
                        "marginBottom": "15px",
                    },
                ),
                dcc.Graph(
                    id="temperature-plot", figure=temp_fig, style={"height": "50vh"}
                ),
            ],
        ),
        html.Div(
            style={
                "backgroundColor": "white",
                "padding": "25px",
                "borderRadius": "8px",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
                "marginBottom": "20px",
            },
            children=[
                html.H2(
                    "Relative Humidity Trends",
                    style={
                        "textAlign": "left",
                        "color": "#495057",
                        "marginBottom": "15px",
                    },
                ),
                dcc.Graph(
                    id="humidity-plot", figure=humidity_fig, style={"height": "50vh"}
                ),
            ],
        ),
        html.Div(
            style={
                "backgroundColor": "white",
                "padding": "25px",
                "borderRadius": "8px",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
                "marginBottom": "20px",
            },
            children=[
                html.H2(
                    "Wind Speed Trends",
                    style={
                        "textAlign": "left",
                        "color": "#495057",
                        "marginBottom": "15px",
                    },
                ),
                dcc.Graph(
                    id="wind-speed-plot", figure=wind_fig, style={"height": "50vh"}
                ),
            ],
        ),
        html.Div(
            style={
                "backgroundColor": "white",
                "padding": "25px",
                "borderRadius": "8px",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
                "marginBottom": "20px",
            },
            children=[
                html.H2(
                    "Precipitation Trends",
                    style={
                        "textAlign": "left",
                        "color": "#495057",
                        "marginBottom": "15px",
                    },
                ),
                dcc.Graph(
                    id="precipitation-plot",
                    figure=precipitation_fig,
                    style={"height": "50vh"},
                ),
            ],
        ),
        html.Div(
            style={
                "backgroundColor": "white",
                "padding": "25px",
                "borderRadius": "8px",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
            },
            children=[
                html.H2(
                    "Cloud Cover Trends",
                    style={
                        "textAlign": "left",
                        "color": "#495057",
                        "marginBottom": "15px",
                    },
                ),
                dcc.Graph(
                    id="cloud-cover-plot", figure=cloud_fig, style={"height": "50vh"}
                ),
            ],
        ),
        html.P(
            "Data: Historical Actuals from Open-Meteo Archive API, ML Model Forecast (Random Forest Regression), and Official Open-Meteo API Forecast.",
            style={
                "textAlign": "center",
                "marginTop": "30px",
                "color": "#6c757d",
                "fontSize": "0.9em",
            },
        ),
    ],
)


# Callback to update plots based on date range
@app.callback(
    [
        Output("temperature-plot", "figure"),
        Output("humidity-plot", "figure"),
        Output("wind-speed-plot", "figure"),
        Output("precipitation-plot", "figure"),
        Output("cloud-cover-plot", "figure"),
        Output("contour-plot", "figure"),
    ],
    [
        Input("date-picker-range", "start_date"),
        Input("date-picker-range", "end_date"),
    ],
)
def update_plots(start_date, end_date):
    # Handle None or invalid dates
    start_date = (
        pd.to_datetime(start_date, utc=True) if start_date else default_start_date
    )
    end_date = pd.to_datetime(end_date, utc=True) if end_date else default_end_date

    # Ensure timezone-aware UTC
    if start_date and not start_date.tzinfo:
        start_date = start_date.tz_localize("UTC")
    if end_date and not end_date.tzinfo:
        end_date = end_date.tz_localize("UTC")

    # Prepare data
    combined_df = prepare_combined_data(start_date, end_date)
    historical_filtered = historical_df[
        (historical_df["date"] >= start_date) & (historical_df["date"] <= end_date)
    ].copy()

    # Update plots
    temp_fig = create_weather_plot(
        combined_df,
        "temperature_2m",
        "Temperature: Actuals vs. Forecasts",
        "Temperature (°C)",
    )
    humidity_fig = create_weather_plot(
        combined_df,
        "relative_humidity_2m",
        "Relative Humidity: Actuals vs. Forecasts",
        "Relative Humidity (%)",
    )
    wind_fig = create_weather_plot(
        combined_df,
        "wind_speed_10m",
        "Wind Speed: Actuals vs. Forecasts",
        "Wind Speed (m/s)",
    )
    precipitation_fig = create_weather_plot(
        combined_df,
        "precipitation",
        "Precipitation: Actuals vs. Forecasts",
        "Precipitation (mm)",
    )
    cloud_fig = create_weather_plot(
        combined_df,
        "cloud_cover",
        "Cloud Cover: Actuals vs. Forecasts",
        "Cloud Cover (%)",
    )
    contour_fig = create_contour_plot(historical_filtered)

    return temp_fig, humidity_fig, wind_fig, precipitation_fig, cloud_fig, contour_fig


# Run the app
if __name__ == "__main__":
    print("Starting Dash application...")
    print(
        "Ensure PostgreSQL is running and 'weather_data', 'weather_predictions', 'weather_api_forecasts' tables are populated."
    )
    app.run(host="0.0.0.0", port=8050, debug=True)
