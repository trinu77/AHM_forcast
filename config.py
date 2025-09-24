


# import os
# import pandas as pd
# from dotenv import load_dotenv
# from sqlalchemy import create_engine
# from datetime import timedelta
# from io import StringIO
# import numpy as np
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
# import google.generativeai as genai
# from google.generativeai import GenerativeModel

# # --- Load Environment Variables ---
# load_dotenv(override=True)
# DB_USER = os.getenv("DB_USER")
# DB_PASSWORD = os.getenv("DB_PASSWORD")
# DB_HOST = os.getenv("DB_HOST")
# DB_PORT = os.getenv("DB_PORT")
# DB_NAME = os.getenv("DB_NAME")
# DB_DRIVER = os.getenv("DB_DRIVER", "mysqlconnector")

# # --- Strict Table Access ---
# TABLE_NAME = "sensor_data_IEMA6012001"

# # --- Database Connection ---
# db_uri = f"mysql+{DB_DRIVER}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# engine = create_engine(db_uri)

# # --- Gemini API Configuration ---
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# model = GenerativeModel("gemini-2.5-pro")

# # --- Global variable to store last forecast result ---
# LAST_FORECAST_RESULT = None

# # --- Data Loader ---
# def load_sensor_data(limit: int = 500, drop_duplicates: bool = True):
#     query = f"SELECT * FROM {TABLE_NAME} LIMIT {limit}"
#     df = pd.read_sql(query, engine)

#     if "timestamp" not in df.columns:
#         raise ValueError("Table does not contain 'timestamp' column.")

#     df["timestamp"] = pd.to_datetime(df["timestamp"])

#     if drop_duplicates:
#         df = df.drop_duplicates(subset=["timestamp"])

#     return df.reset_index(drop=True)

# # --- Dataset Info ---
# def get_dataset_info(limit: int = 500):
#     df = load_sensor_data(limit)
#     distinct_count = df["timestamp"].nunique()
#     return {
#         "db_name": DB_NAME,
#         "table_name": TABLE_NAME,
#         "row_count": len(df),
#         "distinct_timestamps": distinct_count,
#         "sample": df.head(5).to_dict(orient="records")
#     }

# # --- Forecast next N minutes at 10s intervals ---
# def forecast_next_minutes(minutes: int = 2, target_col: str = "temperature_one"):
#     global LAST_FORECAST_RESULT
#     from io import StringIO

#     df = load_sensor_data(limit=500, drop_duplicates=True)
#     if df.empty:
#         return pd.DataFrame({"error": ["No data found in DB"]})

#     if target_col not in df.columns:
#         return pd.DataFrame({"error": [f"Column '{target_col}' not found in DB"]})

#     df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("s")
#     last_ts = df["timestamp"].iloc[-1]
#     print(f"📌 Last available data point in DB (from first 500 rows): {last_ts}")

#     steps = minutes * 6  # 6 intervals per minute (10s each)
#     future_timestamps = [(last_ts + timedelta(seconds=10 * i)).replace(microsecond=0) for i in range(1, steps + 1)]
#     future_ts_strings = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in future_timestamps]

#     prompt = f"""
# You are a professional time-series forecasting assistant.

# Task:
# - Forecast the sensor column '{target_col}' using the first {len(df)} rows of data.
# - The dataset ends at {last_ts.strftime("%Y-%m-%d %H:%M:%S")}.
# - The data is sampled approximately every 10 seconds.
# - Predict the next {minutes} minute(s) at 10-second intervals ({steps} future points).
# - Give more weight to the last 50 values when extrapolating.
# - Ensure predictions stay within realistic bounds (±5% of recent min/max).
# - Format output strictly as CSV with columns: timestamp,predicted
# - Use these exact timestamps. Numeric predictions only (≤2 decimals). No explanations.

# Future timestamps to predict:
# {', '.join(future_ts_strings)}

# Historical dataset (first {len(df)} rows, distinct by timestamp):
# {df[['timestamp', target_col]].to_csv(index=False)}
# """

#     response = model.generate_content(prompt)
#     text_output = response.text.strip().replace("```csv", "").replace("```", "").strip()

#     try:
#         forecast_df = pd.read_csv(StringIO(text_output), parse_dates=["timestamp"])
#         forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"]).dt.floor("s")
#     except Exception as e:
#         return pd.DataFrame({"raw_output": [text_output], "error": [f"Parse failed: {e}"]})

#     if "predicted" not in forecast_df.columns:
#         cols = [c for c in forecast_df.columns if c.lower() != "timestamp"]
#         if cols:
#             forecast_df.rename(columns={cols[0]: "predicted"}, inplace=True)
#     forecast_df["predicted"] = pd.to_numeric(forecast_df["predicted"], errors="coerce").round(2)
#     forecast_df = forecast_df[["timestamp", "predicted"]].sort_values("timestamp").reset_index(drop=True)

#     # Fetch actuals
#     min_ts = last_ts
#     max_ts = future_timestamps[-1]
#     future_query = f"""
#         SELECT timestamp, {target_col} AS actual
#         FROM {TABLE_NAME}
#         WHERE timestamp > '{min_ts.strftime("%Y-%m-%d %H:%M:%S")}'
#           AND timestamp <= '{max_ts.strftime("%Y-%m-%d %H:%M:%S")}'
#         ORDER BY timestamp ASC
#     """
#     future_df = pd.read_sql(future_query, engine)
#     if not future_df.empty:
#         future_df["timestamp"] = pd.to_datetime(future_df["timestamp"]).dt.floor("s")
#         future_df = future_df.drop_duplicates(subset=["timestamp"], keep="first")
#     else:
#         future_df = pd.DataFrame(columns=["timestamp", "actual"])

#     merged = forecast_df.merge(future_df, on="timestamp", how="left")
#     merged["actual"] = merged["actual"].fillna("Not Available")
#     merged = merged[["timestamp", "actual", "predicted"]]

#     LAST_FORECAST_RESULT = merged.copy()
#     return merged

# # --- Forecast Accuracy Metrics ---
# def get_last_forecast_metrics():
#     global LAST_FORECAST_RESULT
#     if LAST_FORECAST_RESULT is None:
#         return {"error": "No forecast has been run yet."}

#     df = LAST_FORECAST_RESULT.copy()
#     df = df[df["actual"].notna()]
#     df = df[df["actual"] != "Not Available"]

#     if df.empty:
#         return {"error": "No valid actual values available."}

#     actual = df["actual"].astype(float).values
#     predicted = df["predicted"].astype(float).values
#     residuals = actual - predicted

#     mse = mean_squared_error(actual, predicted)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(actual, predicted)
#     rse = np.sqrt(np.sum(residuals**2) / (len(residuals) - 2))

#     return {
#         "MSE": round(mse, 5),
#         "RMSE": round(rmse, 5),
#         "R2": round(r2, 5),
#         "RSE": round(rse, 5)
#     }

# # --- Plot Forecast ---
# def plot_forecast(df=None):
#     global LAST_FORECAST_RESULT
#     if df is None:
#         if LAST_FORECAST_RESULT is None or LAST_FORECAST_RESULT.empty:
#             print("⚠️ No forecast available to plot.")
#             return
#         df = LAST_FORECAST_RESULT.copy()

#     df_plot = df.copy()
#     df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])

#     plt.figure(figsize=(12, 5))
#     plt.plot(df_plot["timestamp"], df_plot["predicted"], label="Predicted", marker="o")
    
#     if "actual" in df_plot.columns:
#         actual_numeric = pd.to_numeric(df_plot["actual"], errors="coerce")
#         plt.plot(df_plot["timestamp"], actual_numeric, label="Actual", marker="x")

#     plt.title("Forecast vs Actual")
#     plt.xlabel("Timestamp")
#     plt.ylabel("Value")
#     plt.xticks(rotation=45)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


























import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from datetime import timedelta
from io import StringIO
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import google.generativeai as genai
from google.generativeai import GenerativeModel

# --- Load Environment Variables ---
load_dotenv(override=True)
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_DRIVER = os.getenv("DB_DRIVER", "mysqlconnector")

# --- Strict Table Access ---
TABLE_NAME = "sensor_data_IEMA6012001"

# --- Database Connection ---
db_uri = f"mysql+{DB_DRIVER}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_uri)

# --- Gemini API Configuration ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = GenerativeModel("gemini-2.5-pro")

# --- Global variable to store last forecast result ---
LAST_FORECAST_RESULT = None

# --- Data Loader ---
def load_sensor_data(limit: int = 500, drop_duplicates: bool = True):
    query = f"SELECT * FROM {TABLE_NAME} LIMIT {limit}"
    df = pd.read_sql(query, engine)

    if "timestamp" not in df.columns:
        raise ValueError("Table does not contain 'timestamp' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if drop_duplicates:
        df = df.drop_duplicates(subset=["timestamp"])

    return df.reset_index(drop=True)

# --- Dataset Info ---
def get_dataset_info(limit: int = 500):
    df = load_sensor_data(limit)
    distinct_count = df["timestamp"].nunique()
    return {
        "db_name": DB_NAME,
        "table_name": TABLE_NAME,
        "row_count": len(df),
        "distinct_timestamps": distinct_count,
        "sample": df.head(5).to_dict(orient="records")
    }

# --- Forecast next N minutes at 10s intervals ---
# --- Forecast next N minutes at 10s intervals from a hardcoded timestamp ---
def forecast_next_minutes_from_timestamp(minutes: int = 2, target_col: str = "temperature_one"):
    global LAST_FORECAST_RESULT
    from io import StringIO

    # Hardcoded timestamp to start fetching data
    start_timestamp_str = "2025-06-24 11:22:21"  # Change this as needed
    start_ts = pd.to_datetime(start_timestamp_str)

    # Fetch 500 rows starting from this timestamp
    query = f"""
        SELECT * FROM {TABLE_NAME}
        WHERE timestamp >= '{start_ts.strftime("%Y-%m-%d %H:%M:%S")}'
        ORDER BY timestamp ASC
        LIMIT 500
    """
    df = pd.read_sql(query, engine)

    if df.empty:
        return pd.DataFrame({"error": ["No data found starting from the given timestamp"]})

    if target_col not in df.columns:
        return pd.DataFrame({"error": [f"Column '{target_col}' not found in DB"]})

    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("s")
    last_ts = df["timestamp"].iloc[-1]
    print(f"📌 Last data point used for forecasting: {last_ts}")

    steps = minutes * 6  # 6 intervals per minute (10s each)
    future_timestamps = [(last_ts + timedelta(seconds=10 * i)).replace(microsecond=0) for i in range(1, steps + 1)]
    future_ts_strings = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in future_timestamps]

    # Gemini prompt
    prompt = f"""
You are a professional time-series forecasting assistant.

Task:
- Forecast the sensor column '{target_col}' using the {len(df)} rows of data starting from {start_ts}.
- The dataset ends at {last_ts}.
- The data is sampled approximately every 10 seconds.
- Predict the next {minutes} minute(s) at 10-second intervals ({steps} future points).
- Give more weight to the last 50 values when extrapolating.
- Ensure predictions stay within realistic bounds (±5% of recent min/max).
- Format output strictly as CSV with columns: timestamp,predicted
- Use these exact timestamps. Numeric predictions only (≤2 decimals). No explanations.

Future timestamps to predict:
{', '.join(future_ts_strings)}

Historical dataset:
{df[['timestamp', target_col]].to_csv(index=False)}
"""

    response = model.generate_content(prompt)
    text_output = response.text.strip().replace("```csv", "").replace("```", "").strip()

    try:
        forecast_df = pd.read_csv(StringIO(text_output), parse_dates=["timestamp"])
        forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"]).dt.floor("s")
    except Exception as e:
        return pd.DataFrame({"raw_output": [text_output], "error": [f"Parse failed: {e}"]})

    if "predicted" not in forecast_df.columns:
        cols = [c for c in forecast_df.columns if c.lower() != "timestamp"]
        if cols:
            forecast_df.rename(columns={cols[0]: "predicted"}, inplace=True)

    forecast_df["predicted"] = pd.to_numeric(forecast_df["predicted"], errors="coerce").round(2)
    forecast_df = forecast_df[["timestamp", "predicted"]].sort_values("timestamp").reset_index(drop=True)

    # Fetch actuals
    min_ts = last_ts
    max_ts = future_timestamps[-1]
    future_query = f"""
        SELECT timestamp, {target_col} AS actual
        FROM {TABLE_NAME}
        WHERE timestamp > '{min_ts.strftime("%Y-%m-%d %H:%M:%S")}'
          AND timestamp <= '{max_ts.strftime("%Y-%m-%d %H:%M:%S")}'
        ORDER BY timestamp ASC
    """
    future_df = pd.read_sql(future_query, engine)
    if not future_df.empty:
        future_df["timestamp"] = pd.to_datetime(future_df["timestamp"]).dt.floor("s")
        future_df = future_df.drop_duplicates(subset=["timestamp"], keep="first")
    else:
        future_df = pd.DataFrame(columns=["timestamp", "actual"])

    merged = forecast_df.merge(future_df, on="timestamp", how="left")
    merged["actual"] = merged["actual"].fillna("Not Available")
    merged = merged[["timestamp", "actual", "predicted"]]

    LAST_FORECAST_RESULT = merged.copy()
    return merged



# --- Forecast Accuracy Metrics ---
def get_last_forecast_metrics():
    global LAST_FORECAST_RESULT
    if LAST_FORECAST_RESULT is None:
        return {"error": "No forecast has been run yet."}

    df = LAST_FORECAST_RESULT.copy()
    df = df[df["actual"].notna()]
    df = df[df["actual"] != "Not Available"]

    if df.empty:
        return {"error": "No valid actual values available."}

    actual = df["actual"].astype(float).values
    predicted = df["predicted"].astype(float).values
    residuals = actual - predicted

    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    rse = np.sqrt(np.sum(residuals**2) / (len(residuals) - 2))

    return {
        "MSE": round(mse, 5),
        "RMSE": round(rmse, 5),
        "R2": round(r2, 5),
        "RSE": round(rse, 5)
    }

# --- Plot Forecast ---
def plot_forecast(df=None):
    global LAST_FORECAST_RESULT
    if df is None:
        if LAST_FORECAST_RESULT is None or LAST_FORECAST_RESULT.empty:
            print("⚠️ No forecast available to plot.")
            return
        df = LAST_FORECAST_RESULT.copy()

    df_plot = df.copy()
    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])

    plt.figure(figsize=(12, 5))
    plt.plot(df_plot["timestamp"], df_plot["predicted"], label="Predicted", marker="o")
    
    if "actual" in df_plot.columns:
        actual_numeric = pd.to_numeric(df_plot["actual"], errors="coerce")
        plt.plot(df_plot["timestamp"], actual_numeric, label="Actual", marker="x")

    plt.title("Forecast vs Actual")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
