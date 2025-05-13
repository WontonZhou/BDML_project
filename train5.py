import pandas as pd
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# llama 3.2 I think paper section 4
# preprocess_data and group by size
# categorize_ad_sizes  four categoris stable 60-100k px etc  each has an exponnent
# create_model_prompt  α = opt_bid/(avg_pay*index) text, ask llama for alpha value 
# the details are in the prompt and paper

model_path = r"A:\A3\Llama3.2-3B-hf"
csv_path = "processed_data_with_probability.csv"

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['dim_combo'] = df['ad_slot_width'].astype(str) + 'x' + df['ad_slot_height'].astype(str)
    df['area'] = df['ad_slot_width'] * df['ad_slot_height']
    dim_stats = df.groupby('dim_combo').agg({
        'paying_price': ['mean', 'std', 'count'], 'bidding_price': ['mean'],
        'ad_slot_width': 'first', 'ad_slot_height': 'first', 'area': 'first',
        'win_probability': ['mean', 'std']
    })
    dim_stats.columns = ['avg_price', 'std_price', 'imp_count', 'avg_bid',
                        'width', 'height', 'area', 'avg_win_prob', 'std_win_prob']
    return df, dim_stats.reset_index()

def categorize_ad_sizes(dim_stats):
    size_cats = {
        'Lower paying price & Stable click rate': {'sizes': ['250x250'], 'idx': 1.3},
        'Higher paying price & Stable click rate': {'sizes': ['160x600', '960x90', '336x280', '910x90'], 'idx': 1.1},
        'Lower paying price & Unstable click rate': {'sizes': ['125x125', '234x60', '120x240', '180x600', '640x90', '180x150', '320x50'], 'idx': 1.0},
        'Higher paying price & Unstable click rate': {'sizes': ['468x60', '620x60', '200x200'], 'idx': 0.9}
    }
    dim_stats['cat'] = 'Unknown'
    dim_stats['idx'] = 1.0
    for cat, info in size_cats.items():
        mask = dim_stats['dim_combo'].isin(info['sizes'])
        dim_stats.loc[mask, 'cat'] = cat
        dim_stats.loc[mask, 'idx'] = info['idx']
    area_filter = (dim_stats['area'] >= 60000) & (dim_stats['area'] <= 100000)
    target_sizes = dim_stats[area_filter].copy()
    return dim_stats, target_sizes, size_cats

def calculate_optimal_bids(target_sizes):
    TARGET_WIN_RATE, PRICE_PER_PERCENT = 0.5, 3
    for i, row in target_sizes.iterrows():
        prob_diff = (TARGET_WIN_RATE - row['avg_win_prob']) * 100
        price_adj = prob_diff * PRICE_PER_PERCENT
        opt_bid = row['avg_price'] + price_adj
        opt_bid = max(opt_bid, row['avg_price'] * 0.8)
        opt_bid = min(opt_bid, row['avg_price'] * 1.1)
        target_sizes.loc[i, 'opt_bid'] = opt_bid
        target_sizes.loc[i, 'price_adj'] = price_adj
    return target_sizes

def create_model_prompt(dim_stats, target_sizes, size_cats):
    prompt = """Calculate the optimal alpha values for RTB bidding from the given optimal bids.

Formula: Bid price = α * (Average paying price * index)
So: α = Bid price / (Average paying price * index)

Our data science team found that to achieve a target win rate of 80%, we need specific optimal bid prices for different ad sizes. Your task is to calculate the alpha value that gives us exactly these optimal bids.

Here is the data with already calculated optimal bids:
"""
    for _, row in target_sizes.iterrows():
        prompt += f"Size: {row['dim_combo']}, Area: {row['area']:.0f}, Avg Price: {row['avg_price']:.2f}, Current Win Prob: {row['avg_win_prob']:.2f}, Optimal Bid: {row['opt_bid']:.2f}, Category: {row['cat']}\n"
    prompt += """
Calculate the alpha value for each category by solving:
α = Optimal Bid / (Average paying price * index)

For example, if:
- Optimal Bid = 120
- Average paying price = 100
- index = 1.1
Then α = 120 / (100 * 1.1) = 1.09

RESPOND USING EXACTLY THIS FORMAT:

Category: Lower paying price & Stable click rate
Alpha value: [calculated value]

Category: Higher paying price & Stable click rate
Alpha value: [calculated value]

Category: Lower paying price & Unstable click rate
Alpha value: [calculated value]

Category: Higher paying price & Unstable click rate
Alpha value: [calculated value]

NO other text or explanations.
"""
    return prompt

def extract_alpha_values(response_text, size_cats):
    alphas = {}
    extracted_any = False
    for cat in size_cats.keys():
        pattern = fr"Category:\s*{re.escape(cat)}[\s\n]*Alpha value:\s*([0-9.]+)"
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if matches:
            try:
                val = float(matches[0])
                if 0.1 <= val <= 3.0:
                    alphas[cat] = val
                    extracted_any = True
                    print(f"Extracted alpha for {cat}: {val}")
            except (ValueError, IndexError): pass
    if not extracted_any:
        for cat in size_cats.keys():
            pattern = fr"{re.escape(cat)}.*?([0-9]+\.[0-9]+)"
            matches = re.findall(pattern, response_text, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        val = float(match)
                        if 0.1 <= val <= 3.0:
                            alphas[cat] = val
                            extracted_any = True
                            print(f"Extracted alpha for {cat} (loose match): {val}")
                            break
                    except ValueError: pass
    if not extracted_any or len(alphas) < len(size_cats):
        def_alphas = {
            'Lower paying price & Stable click rate': 1.3,
            'Higher paying price & Stable click rate': 1.1,
            'Lower paying price & Unstable click rate': 1.0,
            'Higher paying price & Unstable click rate': 0.9
        }
        for cat in size_cats.keys():
            if cat not in alphas:
                alphas[cat] = def_alphas[cat] + 0.2
                print(f"Using calculated alpha for {cat}: {alphas[cat]}")
    return alphas

def apply_alpha_to_data(df, dim_stats, alphas, size_cats):
    dim_stats['alpha'] = dim_stats['opt_bid'] = dim_stats['exp_val'] = np.nan
    for cat, alpha in alphas.items():
        cat_sizes = size_cats.get(cat, {}).get('sizes', [])
        idx = size_cats.get(cat, {}).get('idx', 1.0)
        for size in cat_sizes:
            mask = dim_stats['dim_combo'] == size
            if mask.any():
                dim_stats.loc[mask, 'alpha'] = alpha
                dim_stats.loc[mask, 'opt_bid'] = alpha * dim_stats.loc[mask, 'avg_price'] * idx
                win_prob = dim_stats.loc[mask, 'avg_win_prob'].values[0]
                exp_val = win_prob * dim_stats.loc[mask, 'avg_price'].values[0] - (1-win_prob) * dim_stats.loc[mask, 'opt_bid'].values[0]
                dim_stats.loc[mask, 'exp_val'] = exp_val
                print(f"Applied alpha {alpha} to {size} (category: {cat}, win prob: {win_prob:.2f}, exp value: {exp_val:.2f})")
    area_mask = (dim_stats['area'] >= 60000) & (dim_stats['area'] <= 100000)
    unknown_mask = (dim_stats['cat'] == 'Unknown') & area_mask
    if unknown_mask.any():
        print("Applying alphas to unknown categories in target area range...")
        med_price = dim_stats['avg_price'].median()
        for _, row in dim_stats[unknown_mask].iterrows():
            cat = 'Higher paying price & Unstable click rate' if row['avg_price'] >= med_price else 'Lower paying price & Unstable click rate'
            alpha = alphas.get(cat, 1.0)
            idx = 1.0
            size_mask = dim_stats['dim_combo'] == row['dim_combo']
            dim_stats.loc[size_mask, 'alpha'] = alpha
            dim_stats.loc[size_mask, 'opt_bid'] = alpha * row['avg_price'] * idx
            dim_stats.loc[size_mask, 'cat'] = cat
            win_prob = row['avg_win_prob']
            exp_val = win_prob * row['avg_price'] - (1-win_prob) * (alpha * row['avg_price'] * idx)
            dim_stats.loc[size_mask, 'exp_val'] = exp_val
            print(f"Applied alpha {alpha} to {row['dim_combo']} (assigned category: {cat}, win prob: {win_prob:.2f}, exp value: {exp_val:.2f})")
    return dim_stats

def format_results(dim_stats):
    result_rows = []
    valid_mask = dim_stats['alpha'].notnull()
    if valid_mask.any():
        for _, row in dim_stats[valid_mask].iterrows():
            result_rows.append({
                'dim_combo': row['dim_combo'],
                'area': row['area'],
                'avg_price': row['avg_price'],
                'avg_win_prob': row['avg_win_prob'],
                'opt_bid': row['opt_bid'],
                'alpha': row['alpha'],
                'cat': row['cat'],
                'exp_val': row['exp_val']
            })
    return pd.DataFrame(result_rows)

def main():
    model, tokenizer = load_model_and_tokenizer()
    df, dim_stats = preprocess_data(csv_path)
    dim_stats, target_sizes, size_cats = categorize_ad_sizes(dim_stats)
    target_sizes = calculate_optimal_bids(target_sizes)
    prompt = create_model_prompt(dim_stats, target_sizes, size_cats)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    alphas = extract_alpha_values(response, size_cats)
    dim_stats = apply_alpha_to_data(df, dim_stats, alphas, size_cats)
    results = format_results(dim_stats)
    results.to_csv('optimal_bids_results.csv', index=False)

if __name__ == "__main__":
    main()