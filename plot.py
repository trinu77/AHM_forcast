# # plt.py
# import matplotlib.pyplot as plt
# import pandas as pd
# import state  # ✅ import state module

# def plot_forecast(df: pd.DataFrame = None):
#     """
#     Plot the last forecast (actual vs predicted) or use provided dataframe.
#     """
#     if df is None:
#         if state.LAST_FORECAST_RESULT is None or state.LAST_FORECAST_RESULT.empty:
#             print("⚠️ No forecast available to plot.")
#             return
#         df = state.LAST_FORECAST_RESULT.copy()

#     df_plot = df.copy()
#     df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])

#     plt.figure(figsize=(12,5))
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
