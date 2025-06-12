# Weather Data Engineering Project

This project fetches historical weather data, processes it, trains ML models to predict key weather metrics, and visualizes everything in an interactive dashboard. It uses Python, Docker, PostgreSQL, and Dash.

## Features

- Collects weather data using the Open-Meteo API  
- Trains a Random Forest model to predict temperature, humidity, wind speed, precipitation, and cloud cover  
- Shows forecasts from both the ML model and the API  
- Displays everything in a web dashboard  
- Fully Dockerized for easy setup  

## Prerequisites

You need Docker, Docker Compose, and Git installed. Python 3.11 is optional if you're developing outside Docker. At least 4GB RAM and 10GB free disk space is recommended.

## Quick Setup Guide

1. Clone this repository and move into the project folder  
2. Copy the `.env.example` file to `.env` and review the values (default values work for local use)  
3. Run `make all` to build and start everything  
4. Once logs show that the Dash app has started, go to `http://localhost:8050` in your browser  
5. Use the date picker to explore data  

## Common Tasks

- To stop everything, run `make stop-docker`  
- To refresh the data, run `make run-pipeline-docker`  
- To clean everything, including volumes and images, run `make clean`  

## Notes

- Data is for Magdeburg, Germany (lat: 52.1132, lon: 11.6081)  
- Models use 24-hour lagged features for prediction  
- Dashboard includes contour plots and is date-range selectable  
- Everything runs in containers for portability  


## Contact

For questions, reach out to Shyne Sukumaran @ shynesukumaran92@gmail.com  
