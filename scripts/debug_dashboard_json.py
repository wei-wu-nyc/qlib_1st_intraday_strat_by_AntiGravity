
import json
import re
from pathlib import Path
import pandas as pd

def check_html_data():
    path = Path('/Volumes/SAMSUNG_2TB/WorkMac/AntiGravity/qlib_first_intraday_test/results/dashboard.html')
    content = path.read_text()
    
    # Extract JSON
    match = re.search(r'const chartData = ({.*?});', content, re.DOTALL)
    if not match:
        print("Could not find chartData in HTML")
        return
        
    data_str = match.group(1)
    try:
        data = json.loads(data_str)
    except:
        print("JSON parse failed (might be truncated in regex)")
        return

    # Check LightGBM
    if 'LightGBM' in data:
        display_data(data['LightGBM'], 'LightGBM')
    
    # Check Benchmark
    if 'Benchmark (SPY)' in data:
        display_data(data['Benchmark (SPY)'], 'Benchmark')

def display_data(series, name):
    print(f"\n--- {name} Data around 2022-01 ---")
    timestamps = series['timestamps']
    values = series['values']
    
    # Find index near 2022-01-01
    for i, ts in enumerate(timestamps):
        if '2021-12-25' <= ts <= '2022-01-10':
            print(f"{ts}: {values[i]}")

if __name__ == "__main__":
    check_html_data()
