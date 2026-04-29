# download_additional_weather_data.py
# Open-Meteo → NIWA-like hourly -> pseudo-10min radiation conversion

import io
import re
import time
import requests
import pandas as pd

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

def _strip_units(col: str) -> str:
    # e.g. "temperature_2m (°C)" -> "temperature_2m"
    return re.sub(r"\s*\([^)]*\)\s*$", "", col)

def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_strip_units(c).strip() for c in df.columns]
    return df

def fetch_open_meteo_csv(
    lat=-40.35,
    lon=175.61,
    start_date="2023-08-05",
    end_date="2024-12-09",
    timezone="Pacific/Auckland",
    vars_=(
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "windspeed_10m",
        "winddirection_10m",
        "shortwave_radiation",
    ),
    max_retries=3,
    timeout=60,
) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(vars_),
        "timezone": timezone,
        "format": "csv",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
            r.raise_for_status()
            text = r.text

            # Skip metadata lines until the actual header starts with 'time,'
            lines = text.splitlines()
            header_idx = next((i for i, line in enumerate(lines) if line.lower().startswith("time,")), None)
            if header_idx is None:
                raise ValueError("Could not find 'time,' header in CSV output.")

            csv_text = "\n".join(lines[header_idx:])
            df = pd.read_csv(io.StringIO(csv_text))
            if "time" not in df.columns:
                raise ValueError("Parsed CSV missing 'time' column.")

            # Normalize headers to drop unit suffixes
            df = _normalize_headers(df)
            return df

        except Exception as e:
            last_err = e
            if attempt == max_retries:
                raise
            time.sleep(2 * attempt)
    raise last_err

def build_palmy_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Map Open-Meteo hourly variables to NIWA-like columns and expand
    each hour into six 10-min pseudo-records.
    """
    df = df_raw.copy()
    df["time"] = pd.to_datetime(df["time"])

    # Helper to safely pull normalized columns
    def col(name):
        return df[name] if name in df.columns else pd.Series([pd.NA] * len(df))

    # Radiation: hourly mean irradiance (W/m²)
    sw = pd.to_numeric(col("shortwave_radiation"), errors="coerce")
    hourly_mj = sw * 3600.0 / 1_000_000.0

    base = pd.DataFrame({
        "datetime": df["time"],
        "MnDir(degT)": col("winddirection_10m"),
        "MnSpd(mps)":  col("windspeed_10m"),
        "MnTemp(C)":   col("temperature_2m"),
        "MnRH(%)":     col("relative_humidity_2m"),
        "Rain(mm)":    col("precipitation"),
        "Average W/m2": sw,
        "RadGlb(MJ/m2)": hourly_mj,
    })

    # Expand each hourly row into 6 rows of 10 minutes
    expanded = []
    for _, row in base.iterrows():
        for minutes in range(0, 60, 10):
            dt10 = row["datetime"] + pd.Timedelta(minutes=minutes)
            expanded.append({
                "datetime": dt10,
                "MnDir(degT)": row["MnDir(degT)"],
                "MnSpd(mps)": row["MnSpd(mps)"],
                "MnTemp(C)": row["MnTemp(C)"],
                "MnRH(%)": row["MnRH(%)"],
                "Rain(mm)": row["Rain(mm)"] / 6.0 if pd.notna(row["Rain(mm)"]) else pd.NA,
                "Average W/m2": row["Average W/m2"] / 6.0 if pd.notna(row["Average W/m2"]) else pd.NA,
                "RadGlb(MJ/m2)": row["RadGlb(MJ/m2)"] / 6.0 if pd.notna(row["RadGlb(MJ/m2)"]) else pd.NA,
                "StdDir(degT)": pd.NA,
                "StdSpd(mps)": pd.NA,
            })

    df10 = pd.DataFrame(expanded)
    df10["Date(NZST)"] = df10["datetime"].dt.strftime("%d/%m/%Y")
    df10["Time(NZST)"] = df10["datetime"].dt.strftime("%H:%M:%S")

    cols_out = [
        "Date(NZST)", "Time(NZST)",
        "MnDir(degT)", "MnSpd(mps)",
        "StdDir(degT)", "StdSpd(mps)",
        "MnTemp(C)", "MnRH(%)",
        "Rain(mm)", "RadGlb(MJ/m2)", "Average W/m2",
    ]
    return df10[cols_out]


if __name__ == "__main__":
    df_raw = fetch_open_meteo_csv(
        lat=-40.35, lon=175.61,
        start_date="2023-08-05", end_date="2024-12-09",
        timezone="Pacific/Auckland",
    )
    df_out = build_palmy_dataframe(df_raw)

    out_csv = "data/palmerston_north_openmeteo_20230805_20241209_hourly_pseudo10min_rad.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(df_out.head(10).to_string(index=False))
