


from a import create_agent, HumanMessage
from config import (
    get_dataset_info,
    load_sensor_data,
    forecast_next_minutes,
    get_last_forecast_metrics,
    plot_forecast  # for plotting
)
import re

def check_for_forecast(user_input: str) -> bool:
    return any(word in user_input.lower() for word in ["forecast", "predict", "future"])

if __name__ == "__main__":
    print("🤖 Forecast Bot is running. Type 'exit' to quit.\n")
    print("Examples:")
    print("- what dataset you have")
    print("- show all values")
    print("- show distinct values")
    print("- forecast temperature_one")
    print("- forecast vibration_x")
    print("- forecast next 2 minutes temperature_one")
    print("- show accuracy")
    print("- plot it\n")

    graph = create_agent()
    thread_id = "1"

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        try:
            # --- Dataset / Table Queries ---
            if "table" in user_input.lower() or "dataset" in user_input.lower() or "values" in user_input.lower():
                if "all" in user_input.lower():
                    df = load_sensor_data(limit=500, drop_duplicates=False)
                    print(f"Bot: Showing {len(df)} rows from table '{df.columns[0]}':")
                    for row in df.to_dict(orient="records"):
                        print(row)

                elif "distinct" in user_input.lower():
                    df = load_sensor_data(limit=500, drop_duplicates=False)
                    print("Bot: Distinct values per column:")
                    for col in df.columns:
                        unique_vals = df[col].drop_duplicates().tolist()
                        print(f"- {col} ({len(unique_vals)} distinct): {unique_vals}")

                else:
                    info = get_dataset_info(limit=5)
                    print(f"Bot: Connected to DB '{info['db_name']}'")
                    print(f"Bot: I can only access this table: {info['table_name']}")
                    print(f"Bot: Retrieved {info['row_count']} rows, {info['distinct_timestamps']} distinct timestamps")
                    print("Bot: Sample rows:", info["sample"])

            # --- Forecast Queries ---
            elif check_for_forecast(user_input):
                target_col = "temperature_one"
                for col in ["temperature_one", "temperature_two", "vibration_x", "vibration_y", "vibration_z"]:
                    if col in user_input.lower():
                        target_col = col
                        break

                # extract number of minutes (default=2)
                match = re.search(r"next\s+(\d+)\s*minute", user_input.lower())
                minutes = int(match.group(1)) if match else 2

                # Limit forecast to 1–5 minutes
                if minutes > 5:
                    print("Bot: ⚠️ Maximum forecast is 5 minutes. Reducing to 5.")
                    minutes = 5
                elif minutes < 1:
                    minutes = 1

                print(f"Bot: Forecasting next {minutes} minute(s) (10-sec intervals) on '{target_col}'...")
                result = forecast_next_minutes(minutes=minutes, target_col=target_col)
                print("Bot: Forecast result:\n", result)

            # --- Accuracy Metrics ---
            elif "accuracy" in user_input.lower():
                metrics = get_last_forecast_metrics()
                if "error" in metrics:
                    print(f"Bot: {metrics['error']}")
                else:
                    print("Bot: Forecast Accuracy:")
                    for k, v in metrics.items():
                        print(f"- {k}: {v}")

            # --- Plot Forecast ---
            elif "plot" in user_input.lower():
                plot_forecast()
                print("Bot: ✅ Forecast plotted successfully.")

            # --- Other / Chat Queries ---
            else:
                state = {"messages": [HumanMessage(content=user_input)]}
                out = graph.invoke(
                    state,
                    config={
                        "recursion_limit": 50,
                        "configurable": {"thread_id": thread_id}
                    }
                )
                print("Bot:", out["messages"][-1].content)

        except Exception as e:
            print(f"⚠️ Error processing query '{user_input}': {e}")
