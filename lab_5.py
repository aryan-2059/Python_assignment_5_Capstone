from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_all_csvs(data_dir: Path) -> pd.DataFrame:
    all_rows = []
    logs = []

    for csv_path in data_dir.glob("*.csv"):
        try:
            print(f"Loading {csv_path.name}")
            # Example filename patterns:
            #   admin_block_2025-01.csv  -> building="admin", month="2025-01"
            #   hostel_a_2025-01.csv     -> building="hostel", month="2025-01"
            name_parts = csv_path.stem.split("_")
            building = name_parts[0] if len(name_parts) > 0 else "Unknown"
            month = name_parts[-1] if len(name_parts) > 1 else "Unknown"

            # Skip bad lines so a few corrupt rows don't break ingestion
            df = pd.read_csv(csv_path, on_bad_lines="skip")

            # Handle files where the whole line is quoted like "timestamp,kwh"
            # and pandas reads everything into a single column.
            if ("timestamp" not in df.columns or "kwh" not in df.columns) and len(df.columns) == 1:
                single_col = df.columns[0]
                if "timestamp" in str(single_col) and "kwh" in str(single_col):
                    df.columns = ["timestamp_kwh"]
                    # Split the combined string into two separate columns
                    ts_kwh = df["timestamp_kwh"].astype(str).str.strip('"')
                    df[["timestamp", "kwh"]] = ts_kwh.str.split(",", n=1, expand=True)
                    df.drop(columns=["timestamp_kwh"], inplace=True)

            # Ensure kwh is numeric for all valid files
            if "kwh" in df.columns:
                df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")

            # Basic validation
            if "timestamp" not in df.columns or "kwh" not in df.columns:
                logs.append(f"Invalid columns in {csv_path.name}")
                continue

            df["building"] = building
            df["month"] = month
            all_rows.append(df)
        except FileNotFoundError:
            logs.append(f"Missing file: {csv_path}")
        except Exception as e:
            logs.append(f"Error reading {csv_path.name}: {e}")

    if not all_rows:
        raise RuntimeError("No valid CSV files found in data directory.")

    df_combined = pd.concat(all_rows, ignore_index=True)
    return df_combined, logs


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")
    return df

def calculate_daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    # total kWh per day per building
    return df.groupby("building")["kwh"].resample("D").sum().reset_index()


def calculate_weekly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    # weekly totals per building
    return df.groupby("building")["kwh"].resample("W-MON").sum().reset_index()  # week starting Monday


def building_wise_summary(df: pd.DataFrame) -> pd.DataFrame:
    # summary per building over whole period
    grouped = df.groupby("building")["kwh"]
    summary = grouped.agg(["mean", "min", "max", "sum"]).rename(columns={"sum": "total"})
    return summary.reset_index()


class MeterReading:
    def __init__(self, timestamp, kwh):
        self.timestamp = timestamp
        self.kwh = float(kwh)

class Building:
    def __init__(self, name):
        self.name = name
        self.meter_readings = []

    def add_reading(self, reading: MeterReading):
        self.meter_readings.append(reading)

    def to_dataframe(self) -> pd.DataFrame:
        data = [{"timestamp": r.timestamp, "kwh": r.kwh, "building": self.name}
                for r in self.meter_readings]
        return pd.DataFrame(data)

    def calculate_total_consumption(self) -> float:
        return sum(r.kwh for r in self.meter_readings)

    def generate_report(self) -> str:
        total = self.calculate_total_consumption()
        return f"Building {self.name}: total consumption = {total:.2f} kWh"

class BuildingManager:
    def __init__(self):
        self.buildings = {}

    def get_or_create(self, name: str) -> Building:
        if name not in self.buildings:
            self.buildings[name] = Building(name)
        return self.buildings[name]

    def from_dataframe(self, df: pd.DataFrame):
        for _, row in df.reset_index().iterrows():
            b = self.get_or_create(row["building"])
            b.add_reading(MeterReading(row["timestamp"], row["kwh"]))

import matplotlib.pyplot as plt


def create_dashboard(daily_df, weekly_df, df_pre):
    plt.style.use("seaborn-v0_8")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax1, ax2, ax3 = axes[0,0], axes[0,1], axes[1,0]
    fig.delaxes(axes[1,1])  # remove unused fourth

    # Trend line – daily consumption over time (all buildings)
    for bld, grp in daily_df.groupby("building"):
        ax1.plot(grp["timestamp"], grp["kwh"], label=bld)
    ax1.set_title("Daily Consumption")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("kWh")
    ax1.legend(fontsize=8)

    # Bar chart – average weekly usage across buildings
    weekly_avg = weekly_df.groupby("building")["kwh"].mean().reset_index()
    ax2.bar(weekly_avg["building"], weekly_avg["kwh"])
    ax2.set_title("Average Weekly Usage")
    ax2.set_ylabel("kWh")
    ax2.set_xticklabels(weekly_avg["building"], rotation=45, ha="right")

    # Scatter – peak-hour consumption vs time
    # assume df_pre has hourly or sub-hourly; find peaks per day-building
    hourly = df_pre.groupby(["building"]).resample("H")["kwh"].sum()
    hourly = hourly.reset_index()
    peak_rows = hourly.loc[hourly.groupby(["building", hourly["timestamp"].dt.date])["kwh"].idxmax()]
    ax3.scatter(peak_rows["timestamp"], peak_rows["kwh"],
                c=peak_rows["building"].astype("category").cat.codes, alpha=0.6)
    ax3.set_title("Peak-Hour Consumption")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("kWh")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "dashboard.png", dpi=300)
    plt.close(fig)


def export_outputs(df_pre, daily_df, weekly_df, building_summary_df):
    df_pre.to_csv(OUTPUT_DIR / "cleaned_energy_data.csv")
    building_summary_df.to_csv(OUTPUT_DIR / "building_summary.csv", index=False)

    total_campus = building_summary_df["total"].sum()
    highest_row = building_summary_df.loc[building_summary_df["total"].idxmax()]
    highest_bld = highest_row["building"]
    highest_val = highest_row["total"]

    # crude peak load time: max kWh row
    peak_row = df_pre["kwh"].idxmax()
    peak_time = peak_row

    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Campus Energy Executive Summary\n\n")
        f.write(f"Total campus consumption: {total_campus:.2f} kWh\n")
        f.write(f"Highest-consuming building: {highest_bld} ({highest_val:.2f} kWh)\n")
        f.write(f"Peak load time: {peak_time}\n\n")
        f.write("Daily and weekly trends:\n")
        f.write("- See dashboard.png and building_summary.csv for detailed patterns.\n")

    print("Summary written to output/summary.txt")


def main():
    # Task 1: Data ingestion and validation
    if not DATA_DIR.exists():
        raise RuntimeError(f"Data directory {DATA_DIR} does not exist. Please create it and add CSV files.")

    df_combined, logs = load_all_csvs(DATA_DIR)

    # Write any ingestion logs
    if logs:
        log_path = OUTPUT_DIR / "ingestion_log.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Ingestion / Validation Log\n\n")
            for line in logs:
                f.write(line + "\n")
        print(f"Ingestion issues logged to {log_path}")

    # Task 2: Core aggregation logic
    df_pre = preprocess(df_combined)
    daily_df = calculate_daily_totals(df_pre)
    weekly_df = calculate_weekly_aggregates(df_pre)
    building_summary_df = building_wise_summary(df_pre)

    # Task 3: Object-Oriented modeling
    manager = BuildingManager()
    manager.from_dataframe(df_pre)
    print("\nPer-building reports:")
    for building in manager.buildings.values():
        print(building.generate_report())

    # Task 4: Visual dashboard
    create_dashboard(daily_df, weekly_df, df_pre)
    print(f"Dashboard saved to {OUTPUT_DIR / 'dashboard.png'}")

    # Task 5: Persistence and executive summary
    export_outputs(df_pre, daily_df, weekly_df, building_summary_df)


if __name__ == "__main__":
    main()
