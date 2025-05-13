import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# XGBoost based RTB bid optimizer paper sec 5
# for each ad impression using GBDT to compare LLaMA strategy in train5.py
# preprocess data into four size/price categories
# and assigns index values 1.3, 1.1, 1.0, 0.9
# computes win-rate gap so if current win rate < 0.50 then raises α value accordingly
# so TARGET_WIN_RATE = 0.50, PRICE_PER_PERCENT = 3 CNY

class RTBOptimizer:
    def __init__(self, data_path='processed_data_with_probability.csv'):
        self.data = pd.read_csv(data_path)
        self.processed_data = self.result_data = self.ml_model = None
        
    def preprocess(self):
        df = self.data.copy()
        df['area'] = df['ad_slot_width'] * df['ad_slot_height']
        df['aspect_ratio'] = df['ad_slot_width'] / df['ad_slot_height']
        df['dimension_combo'] = df['ad_slot_width'].astype(str) + 'x' + df['ad_slot_height'].astype(str)
        avg_price = df['paying_price'].mean()
        print(f"Average paying price: {avg_price:.2f}")
        
        conditions = [
            (df['area'] >= 60000) & (df['area'] <= 100000) & (df['paying_price'] < avg_price),
            (df['area'] >= 60000) & (df['area'] <= 100000) & (df['paying_price'] >= avg_price),
            ((df['area'] < 60000) | (df['area'] > 100000)) & (df['paying_price'] < avg_price),
            ((df['area'] < 60000) | (df['area'] > 100000)) & (df['paying_price'] >= avg_price)
        ]
        choices = ['Lower paying price & Stable click rate', 'Higher paying price & Stable click rate',
                  'Lower paying price & Unstable click rate', 'Higher paying price & Unstable click rate']
        df['category'] = np.select(conditions, choices, default='Unknown')
        df.loc[conditions[0], 'index'] = 1.3
        df.loc[conditions[1], 'index'] = 1.1
        df.loc[conditions[2], 'index'] = 1.0
        df.loc[conditions[3], 'index'] = 0.9
        
        alpha_values = {
            'Lower paying price & Stable click rate': 1.5,
            'Higher paying price & Stable click rate': 1.3,
            'Lower paying price & Unstable click rate': 1.2,
            'Higher paying price & Unstable click rate': 1.1
        }
        
        TARGET_WIN_RATE, PRICE_PER_PERCENT = 0.5, 3
        df['win_prob_adjustment'] = 0.0
        if 'win_probability' in df.columns:
            print("Using win probability to adjust alpha values...")
            for i, row in df.iterrows():
                if not np.isnan(row['paying_price']) and row['paying_price'] > 0:
                    prob_diff = (TARGET_WIN_RATE - row['win_probability']) * 100
                    price_adjustment = prob_diff * PRICE_PER_PERCENT
                    alpha_adjustment = price_adjustment / (row['paying_price'] * row.get('index', 1.0))
                    df.loc[i, 'win_prob_adjustment'] = alpha_adjustment
        
        df['initial_alpha'] = df['category'].map(alpha_values).fillna(1.2)
        df['adjusted_alpha'] = (df['initial_alpha'] + df['win_prob_adjustment']).clip(0.8, 2.0)
        np.random.seed(35)
        df['alpha_noise'] = np.random.normal(0, 0.1, size=len(df))
        df['training_alpha'] = (df['adjusted_alpha'] + df['alpha_noise']).clip(0.8, 2.0)
        
        self.processed_data = df
        return df
    
    def train_ml_model(self):
        df = self.processed_data.copy()
        features = ['ad_slot_width', 'ad_slot_height', 'ad_slot_visibility', 'ad_slot_format',
                   'ad_slot_floor_price', 'area', 'aspect_ratio', 'index']
        if 'win_probability' in df.columns: features.append('win_probability')
        
        X = df[features]
        y = df['training_alpha']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training ML model on {len(X_train)} samples...")
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, subsample=0.8,
                           colsample_bytree=0.8, random_state=42)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"Model R² on training data: {train_score:.4f}")
        print(f"Model R² on testing data: {test_score:.4f}")
        
        importance = dict(zip(features, model.feature_importances_))
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feat}: {imp:.4f}")
        
        plt.figure(figsize=(10, 6))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1])
        plt.barh([x[0] for x in sorted_imp], [x[1] for x in sorted_imp])
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        self.ml_model = model
        return model
    
    def calculate_optimal_bids(self):
        df = self.processed_data.copy()
        features = ['ad_slot_width', 'ad_slot_height', 'ad_slot_visibility', 'ad_slot_format',
                   'ad_slot_floor_price', 'area', 'aspect_ratio', 'index']
        if 'win_probability' in df.columns: features.append('win_probability')
        
        if self.ml_model is not None:
            print("Using ML model to predict alpha values...")
            df['alpha'] = self.ml_model.predict(df[features]).clip(0.8, 2.0)
        else:
            print("ML model not trained, using adjusted alpha values...")
            df['alpha'] = df['adjusted_alpha']
        
        df['optimal_bid'] = (df['alpha'] * df['paying_price'] * df['index']).clip(upper=df['paying_price'] * 1.1)
        if 'win_probability' in df.columns:
            df['expected_value'] = df['win_probability'] * df['paying_price'] - (1 - df['win_probability']) * df['optimal_bid']
        
        self.processed_data = df
        agg_dict = {
            'ad_slot_width': 'first', 'ad_slot_height': 'first', 'area': 'first',
            'category': lambda x: x.mode()[0], 'index': 'mean', 'alpha': 'mean',
            'paying_price': 'mean', 'optimal_bid': 'mean', 'dimension_combo': 'count'
        }
        if 'expected_value' in df.columns: agg_dict['expected_value'] = 'mean'
        if 'win_probability' in df.columns: agg_dict['win_probability'] = 'mean'
        
        result = df.groupby('dimension_combo').agg(agg_dict).rename(columns={'dimension_combo': 'count'})
        self.result_data = result.sort_values('count', ascending=False)
        return self.result_data
    
    def run(self):
        print("start training")
        self.preprocess()
        self.train_ml_model()
        result = self.calculate_optimal_bids()
        top_results = result.head(10)
        
        for _, row in top_results.iterrows():
            output_str = f"Size: {row.name}, Area: {row['area']:.0f}, Category: {row['category']}, Alpha: {row['alpha']:.2f}, Index: {row['index']:.1f}, Avg Price: {row['paying_price']:.2f}, Optimal Bid: {row['optimal_bid']:.2f}"
            if 'expected_value' in row: output_str += f", Expected Value: {row['expected_value']:.2f}"
            if 'win_probability' in row: output_str += f", Win Prob: {row['win_probability']:.2f}"
            print(output_str)
        
        result.to_csv('optimal_bids_xgboost.csv')
        return result

def main():
    optimizer = RTBOptimizer()
    optimizer.run()

if __name__ == "__main__":
    main()