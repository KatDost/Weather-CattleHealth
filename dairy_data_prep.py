import pandas as pd

# Read in the main table
df = pd.read_excel("data/dairy_raw/Dairy_stx_genes.xlsx", header=[0, 1])
df.columns = ['Date', 'Stx (RT-PCR)', 'Stx (NeoSeek)',
       'Top 7 STEC (RT-PCR)', 'Top 7 STEC_NeoSeek', 'C. lari (RT-PCR)',
       'Salmonella sp. (RT-PCR)',
       'Heat stress_THI (>68 for Friesan; >75 for Jersey)',
       'Heat stress_Farmers observations (shade, behaviour)']
df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), format="mixed", dayfirst=True)
# df["Date"] = pd.to_datetime(df["Date"].astype("string").str.strip(), format="%d/%m/%Y", errors="raise")

neo = df["Top 7 STEC_NeoSeek"].fillna("").astype(str)
df["O103 (NeoSeek)"] = neo.str.contains("O103", regex=False).astype(int)
df["O45 (NeoSeek)"]  = neo.str.contains("O45",  regex=False).astype(int)
df["O145 (NeoSeek)"] = neo.str.contains("O145", regex=False).astype(int)
df["O103/45/145 (NeoSeek)"] = ((df["O103 (NeoSeek)"] | df["O45 (NeoSeek)"] | df["O145 (NeoSeek)"]) > 0).astype(int)
df = df.drop(columns=["Top 7 STEC_NeoSeek"])

# Add season column
def get_season(month):
    if month in [11, 12, 1, 2]:
        return "summer"
    elif month in [5, 6, 7, 8]:
        return "winter"
    else:
        return None

df["season"] = df["Date"].dt.month.apply(get_season)

# Read in the weather data
df_w = pd.read_excel("data/dairy_raw/Dairy_weather.xlsx", header=[0, 1])
df_w.columns = ['Date', 'Max Air Temp (oC)', 'Mean RH (%)', 'Total Solar Radiation (MJ/m2/day)',
                'Mean Wind Speed (m/s)', 'Total Rainfall (mm)']
df_w = df_w.reset_index(drop=True)


# Calculate heat stress indices
# -----------------------------
temp = df_w['Max Air Temp (oC)']
rh = df_w['Mean RH (%)']
ws = df_w['Mean Wind Speed (m/s)']
rad = df_w['Total Solar Radiation (MJ/m2/day)']

# THI = Temperature-Humidity Index
df_w['THI'] = 0.8 * temp + rh * (temp - 14.4) + 46.4

# Grazing HLI = Heat Load Index
df_w['Grazing HLI'] = 61.78 + 4.21 * (temp - 22.48) - 1.7 * (ws - 7.05) + 5.89 * (rad - 2.41)


weather_cols = df_w.columns[1:]  # the 5 weather columns
pathogen_cols = list(df.columns[1:6]) + list(df.columns[-2:-1])