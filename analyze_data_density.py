
import pandas as pd
import numpy as np
import ast
from preprocessing.data_loader import DataLoader

def analyze_data():
    loader = DataLoader('d:/Cloud_research/data/raw')
    df = loader.load_all_data()
    
    print(f"Total rows: {len(df)}")
    print(f"Time range: {df['time'].min()} to {df['time'].max()}")
    print(f"Duration: {df['time'].max() - df['time'].min()}")
    print(f"Unique machines: {df['machine_id'].nunique()}")
    
    # Check frequency per machine
    counts = df.groupby('machine_id').size()
    print("\nTop 10 machines by row count:")
    print(counts.sort_values(ascending=False).head(10))
    
    # Check time gaps for top machine
    top_machine = counts.idxmax()
    machine_data = df[df['machine_id'] == top_machine].sort_values('time')
    print(f"\nTop Machine {top_machine} time diffs description:")
    print(machine_data['time'].diff().describe())
