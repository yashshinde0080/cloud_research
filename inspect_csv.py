
import pandas as pd
import json

try:
    df = pd.read_csv('d:/Cloud_research/data/raw/borg_traces_data.csv', nrows=5)
    print("Columns:", df.columns.tolist())
    for col in df.columns:
        print(f"Col: {col}, First Val: {df[col].iloc[0]}, Type: {type(df[col].iloc[0])}")
except Exception as e:
    print(e)
