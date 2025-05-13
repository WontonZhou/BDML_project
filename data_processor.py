import pandas as pd
import numpy as np

# clean the iPinYou 1.37 GB CSV, read in 50k chuncks to dodge RAM boom
# keep 8 hot featrues which can be seen in Fig 1, rest tossed
# clean price/size str→int, regex grab digts

filename = 'A:\\data.txt'
column_names = [
    "bid_id", "timestamp", "log_type", "ipinyou_id", "user_agent",
    "ip", "region", "city", "ad_exchange", "domain",
    "url", "anon_url_id", "ad_slot_id", "ad_slot_width",
    "ad_slot_height", "ad_slot_visibility", "ad_slot_format",
    "ad_slot_floor_price", "creative_id", "bidding_price",
    "paying_price", "key_page_url", "advertiser_id", "user_tags",
    "col25", "col26"
]
keep_columns = [
    "ad_slot_width", "ad_slot_height", "ad_slot_visibility", "ad_slot_format",
    "ad_slot_floor_price", "bidding_price", "paying_price", "advertiser_id"
]

def read_and_process_data(filename, column_names, keep_columns, sample_size=100000, output_file='processed_data_for_rl.csv'):
    print(f"Starting to process file: {filename}")
    chunks = []
    for chunk in pd.read_csv(filename, sep='\t', names=column_names, chunksize=50000, 
                           engine='python', on_bad_lines='skip', dtype=str,
                           skip_blank_lines=True, encoding='utf-8'):
        chunk = chunk[keep_columns]
        for col in ['ad_slot_width', 'ad_slot_height', 'ad_slot_floor_price', 'bidding_price', 'paying_price']:
            chunk[col] = chunk[col].astype(str).str.extract('(\\d+)').iloc[:, 0]
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        chunk = chunk.dropna()
        chunk['dimension_combo'] = chunk['ad_slot_width'].astype(int).astype(str) + 'x' + chunk['ad_slot_height'].astype(int).astype(str)
        chunks.append(chunk)
        if sum(len(c) for c in chunks) >= sample_size:
            break
    df = pd.concat(chunks, ignore_index=True)
    if len(df) > sample_size
        df = df.sample(sample_size, random_state=42)
    for col in ['ad_slot_width', 'ad_slot_height', 'ad_slot_floor_price', 'bidding_price', 'paying_price']:
        df[col] = df[col].astype(float)
    df['area'] = df['ad_slot_width'] * df['ad_slot_height']
    df['aspect_ratio'] = df['ad_slot_width'] / df['ad_slot_height']
    dim_stats = df.groupby('dimension_combo').agg({
        'bidding_price': 'mean',
        'paying_price': 'mean',
        'area': 'mean'
    }).reset_index()
    dim_stats.columns = ['dimension_combo', 'avg_bidding_price', 'avg_paying_price', 'avg_area']
    df = df.merge(dim_stats[['dimension_combo', 'avg_bidding_price', 'avg_paying_price']], 
                 on='dimension_combo', how='left')
    df.to_csv(output_file, index=False)
    return df

processed_df = read_and_process_data(filename, column_names, keep_columns)

