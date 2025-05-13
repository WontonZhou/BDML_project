import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# dimension and price analisis, see paper sec 2.1-2.2
# load 1.37 GB CSV in 50k chuncks when pandas fails then a fallback loop
# derive area, aspect_ratio, dimention_combo "WxH"
# train RF, the result shows 125×125 ads are cheeper per pixel
# predict full width height grid and heatmap confirms cost climbs with larger dimenson

filename = 'A:\\data.txt'
column_names = ["bid_id", "timestamp", "log_type", "ipinyou_id", "user_agent", "ip", "region", "city", "ad_exchange", "domain", "url", "anon_url_id", "ad_slot_id", "ad_slot_width", "ad_slot_height", "ad_slot_visibility", "ad_slot_format", "ad_slot_floor_price", "creative_id", "bidding_price", "paying_price", "key_page_url", "advertiser_id", "user_tags", "col25", "col26"]

def read_large_file_robustly(filename, column_names, chunk_size=50000):
    all_chunks = []
    df_iter = pd.read_csv(filename, sep="\t", names=column_names, quotechar='"', engine="python", on_bad_lines="skip", chunksize=chunk_size, dtype=str, skip_blank_lines=True, encoding='utf-8', error_bad_lines=False, warn_bad_lines=True, low_memory=False)
    for chunk in df_iter:
        chunk = clean_chunk(chunk)
        all_chunks.append(chunk)
    return pd.concat(all_chunks, ignore_index=True)

def clean_chunk(chunk):
    required_cols = ['ad_slot_width', 'ad_slot_height', 'paying_price']
    if not all(col in chunk.columns for col in required_cols):
        missing = [col for col in required_cols if col not in chunk.columns]
        for col in missing:
            chunk[col] = np.nan
    numeric_cols = ['ad_slot_width', 'ad_slot_height', 'ad_slot_visibility', 'ad_slot_format', 'ad_slot_floor_price', 'bidding_price', 'paying_price', 'region', 'city', 'ad_exchange']
    for col in numeric_cols:
        if col in chunk.columns:
            if col in ['ad_slot_width', 'ad_slot_height']:
                chunk[col] = chunk[col].astype(str).str.extract('(\d+)').iloc[:, 0]
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    return chunk

def categorize_size(row):
    area = row['ad_slot_width'] * row['ad_slot_height']
    if area < 10000: return 'Small'
    elif area < 100000: return 'Medium'
    else: return 'Large'

def analyze_ad_slot_dimensions():
    df = read_large_file_robustly(filename, column_names)
    df = df.dropna(subset=['paying_price', 'ad_slot_width', 'ad_slot_height'])
    df = df[df['paying_price'] >= 0]
    df = df[(df['ad_slot_width'] > 0) & (df['ad_slot_height'] > 0)]
    df['ad_slot_width'] = df['ad_slot_width'].astype(int)
    df['ad_slot_height'] = df['ad_slot_height'].astype(int)
    df['area'] = df['ad_slot_width'] * df['ad_slot_height']
    df['aspect_ratio'] = df['ad_slot_width'] / df['ad_slot_height']
    df['size_category'] = df.apply(categorize_size, axis=1)
    df['dimension_combo'] = df['ad_slot_width'].astype(str) + 'x' + df['ad_slot_height'].astype(str)
    print(df[['ad_slot_width', 'ad_slot_height', 'dimension_combo', 'paying_price']].describe())
    dimension_stats = df.groupby('dimension_combo').agg({'paying_price': ['mean', 'median', 'std', 'count']}).reset_index()
    dimension_stats.columns = ['dimension_combo', 'mean_price', 'median_price', 'std_price', 'count']
    common_dimensions = dimension_stats[dimension_stats['count'] >= 50].sort_values('mean_price')
    print(common_dimensions.head(10))
    plt.figure(figsize=(12, 8))
    sns.barplot(x='dimension_combo', y='mean_price', data=common_dimensions.head(15))
    plt.title('Average Paying Price by Ad Slot Dimensions')
    plt.xlabel('Width x Height')
    plt.ylabel('Average Paying Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['ad_slot_width'], df['ad_slot_height'], c=df['paying_price'], cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Paying Price')
    plt.title('Ad Slot Dimensions vs Paying Price')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.tight_layout()
    plt.show()
    # implemented the machine learning analysis using random forest
    print("Starting machine learning analysis...")
    features = ['ad_slot_width', 'ad_slot_height', 'area', 'aspect_ratio']
    X = df[features]
    y = df['paying_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=10, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\nRandom Forest Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²): {r2}")
    feat_importances = pd.Series(model.feature_importances_, index=features)
    plt.figure(figsize=(10, 6))
    feat_importances.sort_values(ascending=False).plot(kind='bar')
    plt.title('Feature Importance for Predicting Paying Price')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()
    widths = range(50, 1001, 50)
    heights = range(50, 1001, 50)
    results = []
    for w in widths:
        for h in heights:
            area = w * h
            aspect = w / h
            features = np.array([[w, h, area, aspect]])
            predicted_price = model.predict(features)[0]
            results.append({'width': w, 'height': h, 'dimension_combo': f"{w}x{h}", 'area': area, 'aspect_ratio': aspect, 'predicted_price': predicted_price})
    predictions_df = pd.DataFrame(results)
    lowest_price_combos = predictions_df.sort_values('predicted_price').head(10)
    print("\nPredicted ad slot dimensions with lowest paying prices:")
    print(lowest_price_combos[['dimension_combo', 'predicted_price', 'area', 'aspect_ratio']])
    plt.figure(figsize=(12, 10))
    pivot_table = predictions_df.pivot_table(values='predicted_price', index='height', columns='width')
    sns.heatmap(pivot_table, cmap='viridis_r', annot=False)
    plt.title('Predicted Paying Price by Ad Slot Dimensions')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.tight_layout()
    plt.show()
    return lowest_price_combos, model, df

if __name__ == "__main__":
    lowest_price_combos, model, df = analyze_ad_slot_dimensions()