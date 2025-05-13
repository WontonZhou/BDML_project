import pandas as pd
import numpy as np
import math

# add win_prob to dataset as the professor told in our presentation
# w = .3*p + .25*d + .1*s + .05*v + .1*c + .2*r - 0.1
# p  → sigmoid(bid / floor)  the biggest weight
# d  → price vs avg for same dimenson
# s  → size sweet-spot 60-100k px penalize XXL slots
# v  → tiny hit for high visibilty  
# c  → market compontent pressure
# r  → random 

def add_balanced_win_probability(input_file='processed_data_for_rl.csv', output_file='processed_data_with_probability.csv'):
    df = pd.read_csv(input_file)
    
    if 'dimension_combo' not in df.columns:
        df['dimension_combo'] = df['ad_slot_width'].astype(str) + 'x' + df['ad_slot_height'].astype(str)
    
    dim_stats = df.groupby('dimension_combo').agg({
        'paying_price': 'mean',
        'bidding_price': 'mean',
        'ad_slot_floor_price': 'mean'
    }).rename(columns={
        'paying_price': 'avg_paying_price_by_dim',
        'bidding_price': 'avg_bidding_price_by_dim',
        'ad_slot_floor_price': 'avg_floor_price_by_dim'
    })
    
    df = df.merge(dim_stats, on='dimension_combo', how='left')
    
    df['price_ratio'] = df['bidding_price'] / df['ad_slot_floor_price'].replace(0, 0.01)
    df['price_ratio'] = df['price_ratio'].clip(0, 100)
    k1 = 0.01 
    df['price_component'] = 1 / (1 + np.exp(-k1 * (df['price_ratio'] - 5))) 

    df['price_vs_dim_avg'] = df['bidding_price'] / df['avg_paying_price_by_dim'].replace(0, 0.01)
    df['price_vs_dim_avg'] = df['price_vs_dim_avg'].clip(0, 10)
    k2 = 0.2 
    df['dim_price_component'] = 1 / (1 + np.exp(-k2 * (df['price_vs_dim_avg'] - 1.5)))  
    
    if 'area' not in df.columns:
        df['area'] = df['ad_slot_width'] * df['ad_slot_height']
    df['area_normalized'] = (df['area'] - df['area'].min()) / (df['area'].max() - df['area'].min())
    df['size_component'] = 1 - (df['area_normalized'] * 0.3)
    
    df['visibility_component'] = 1 - (df['ad_slot_visibility'] * 0.05)
    
    global_avg_price = df['paying_price'].mean()
    df['dim_price_ratio'] = df['avg_paying_price_by_dim'] / global_avg_price
    df['competition_component'] = 1 - (0.2 * (df['dim_price_ratio'] - 1).clip(-0.5, 2))
    
    np.random.seed(35)
    df['random_component'] = np.random.normal(0, 0.25, size=len(df)) 
    
    offset = 0.1
  
    df['win_probability'] = (
        df['price_component'] * 0.3 +
        df['dim_price_component'] * 0.25 +
        df['size_component'] * 0.1 +
        df['visibility_component'] * 0.05 +
        df['competition_component'] * 0.1 +
        df['random_component'] * 0.2  
    ) - offset  
    
    df['win_probability'] = df['win_probability'].clip(0.01, 0.99)

    np.random.seed(35)
    random_values = np.random.random(len(df))
    df['win_outcome'] = (random_values < df['win_probability']).astype(int)
    
    drop_columns = ['price_ratio', 'price_component', 'price_vs_dim_avg', 'dim_price_component',
                  'area_normalized', 'size_component', 'visibility_component', 
                  'dim_price_ratio', 'competition_component', 'random_component']
    df = df.drop(drop_columns, axis=1)

    total_wins = df['win_outcome'].sum()
    win_rate = total_wins / len(df) * 100

    probability_buckets = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    probability_counts = pd.cut(df['win_probability'], probability_buckets).value_counts().sort_index()

    for i in range(len(probability_buckets)-1):
        count = probability_counts.iloc[i]
        percent = count / len(df) * 100
        print(f"  {probability_buckets[i]:.1f}-{probability_buckets[i+1]:.1f}: {count} rows ({percent:.2f}%)")
    
    dim_win_stats = df.groupby('dimension_combo').agg({
        'win_probability': ['mean', 'std', 'count'],
        'win_outcome': 'mean'
    })
    
    top_dims = dim_win_stats.sort_values(('win_probability', 'count'), ascending=False).head(10)
    for dim, stats in top_dims.iterrows():
        print(f"  {dim}: Prob {stats[('win_probability', 'mean')]:.4f}, Win Rate {stats[('win_outcome', 'mean')]:.4f}, Count {stats[('win_probability', 'count')]}")
    
    df.to_csv(output_file, index=False)
    
    
    return {
        'total_rows': len(df),
        'total_wins': total_wins,
        'win_rate': win_rate,
        'avg_win_probability': df['win_probability'].mean(),
        'std_win_probability': df['win_probability'].std()
    }

if __name__ == "__main__":
    stats = add_balanced_win_probability()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")