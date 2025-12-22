import pandas as pd
from datetime import datetime
import os

def save_dataframe(df, filename, directory="data/processed"):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    
    if filename.endswith('.xlsx'):
        df.to_excel(filepath, index=False)
    elif filename.endswith('.csv'):
        df.to_csv(filepath, index=False)
    else:
        raise ValueError("Unsupported file format. Use .xlsx or .csv")
    
    print(f"Saved {len(df)} records to {filepath}")
    return filepath

def load_dataframe(filepath):
    if filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)
    elif filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    else:
        raise ValueError("Unsupported file format. Use .xlsx or .csv")

def format_currency(amount):
    return f"${amount:,.2f}"

def format_date(date_str):
    try:
        dt = pd.to_datetime(date_str)
        return dt.strftime('%Y-%m-%d')
    except:
        return date_str

def calculate_days_between(start_date, end_date):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    return (end - start).days

def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def print_section(title, width=80):
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")

def print_subsection(title, width=80):
    print("\n" + "-" * width)
    print(title)
    print("-" * width)