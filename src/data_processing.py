import pandas as pd
import numpy as np
import socket
import struct
import os

def load_data(fraud_path, ip_path):
    """Loads the raw CSV datasets."""
    fraud_df = pd.read_csv(fraud_path)
    ip_df = pd.read_csv(ip_path)
    return fraud_df, ip_df

def ip_to_int(ip):
    """Converts an IP string to an integer."""
    try:
        return struct.unpack("!I", socket.inet_aton(str(ip)))[0]
    except (socket.error, ValueError):
        return 0

def clean_data(df):
    """Handles basic cleaning: types, duplicates, and missing values."""
    # Convert timestamps
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Fill missing values for key columns if any
    df = df.dropna(subset=['user_id', 'ip_address'])
    
    return df

def merge_with_geo(fraud_df, ip_df):

    fraud_df = fraud_df.copy()
    ip_df = ip_df.copy()

    # Convert to int64 WITHOUT rounding
    fraud_df['ip_int'] = fraud_df['ip_address'].astype('int64')

    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype('int64')
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype('int64')

    # Sort (mandatory for merge_asof)
    fraud_df = fraud_df.sort_values('ip_int')
    ip_df = ip_df.sort_values('lower_bound_ip_address')

    merged_df = pd.merge_asof(
        fraud_df,
        ip_df,
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )

    # Range validation
    valid = (
        (merged_df['ip_int'] >= merged_df['lower_bound_ip_address']) &
        (merged_df['ip_int'] <= merged_df['upper_bound_ip_address'])
    )

    merged_df.loc[~valid, 'country'] = 'Unknown'
    merged_df['country'] = merged_df['country'].fillna('Unknown')

    return merged_df

def engineer_features(df):
    """Creates new features to help identify fraud patterns."""
    # 1. Time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # 2. Velocity: Time since signup
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    
    # 3. Velocity: Transaction frequency per device/IP
    df['user_id_count'] = df.groupby('user_id')['user_id'].transform('count')
    df['device_id_count'] = df.groupby('device_id')['device_id'].transform('count')
    df['ip_address_count'] = df.groupby('ip_address')['ip_address'].transform('count')
    
    return df

def save_processed_data(df, output_path):
    """Saves the final dataframe to the processed data folder."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully saved processed data to: {output_path}")

if __name__ == "__main__":
    # Define Paths
    RAW_FRAUD = "../data/raw/Fraud_Data.csv"
    RAW_IP = "../data/raw/IpAddress_to_Country.csv"
    PROCESSED_OUTPUT = "../data/processed/processed_fraud_data.csv"
    
    print("🚀 Starting Data Processing Pipeline...")
    
    # Execute Pipeline
    f_df, i_df = load_data(RAW_FRAUD, RAW_IP)
    f_df = clean_data(f_df)
    f_df = merge_with_geo(f_df, i_df)
    f_df = engineer_features(f_df)
    
    # Save Output
    save_processed_data(f_df, PROCESSED_OUTPUT)
    
    print("✅ Processing Complete.")