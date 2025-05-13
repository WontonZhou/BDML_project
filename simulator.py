import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# RTB simulaton paper section 6
# 5 agents bid in standard 2-price auction on 5k random ads; winner pays 2nd bid.

# BaselineAgent
#   bid = floor * 1.1  ±5 % noise, this one ignores size, visibilty, history etc

# XGBoostAgent1
#   this is rule based alpha value strategy using alpha values obtained from XGBoost training outputs ranging from 1.1-1.5.
#   It determines stability based on whether the ad area is in the range of 60K-100K pixels then categorize 4 categories

# XGBoostAgent2
#   tuned α 1.04-1.53 based on xgBoost training output, +10 % for hot dimenson, +5 % high visibilty,
#   extra index multiplier 

# LlamaAgent1
#   α range = 0.9-1.3 we try exact size match else fall to category rule,

# LlamaAgent2
#   uses llama-3.2 traning output as α value, and set optimal_bid directly and +10 % on good visibilty,
#   if size unkown fall back to α×price the most agressive strategy.

# Code outputs win-rate, spend, profit so we judge each alogrithm under same noise.

class BiddingAgent:
    def __init__(self, name):
        self.name = name
        self.bids = self.wins = self.spend = self.revenue = self.total_value = 0
        self.imps = []

    def record_result(self, ad_info, bid_price, won, actual_price=0):
        self.bids += 1
        if won:
            self.wins += 1
            self.spend += actual_price
            self.imps.append(ad_info["dimension_combo"])
            val = ad_info["paying_price"] * 1.2
            self.revenue += val
            self.total_value += val - actual_price
    
    def get_metrics(self):
        return {
            "name": self.name, "bids": self.bids, "wins": self.wins,
            "win_rate": self.wins / self.bids if self.bids > 0 else 0,
            "spend": self.spend, "revenue": self.revenue,
            "total_value": self.total_value,
            "average_cost": self.spend / self.wins if self.wins > 0 else 0
        }

class BaselineAgent(BiddingAgent):
    def __init__(self):
        super().__init__("Baseline")
    
    def calculate_bid(self, ad_info):
        return ad_info["ad_slot_floor_price"] * 1.1 * np.random.uniform(0.95, 1.05)

class XGBoostAgent1(BiddingAgent):
    def __init__(self, avg_price):
        super().__init__("XGBoost-1")
        self.avg_price = avg_price
        self.alphas = {
            'Lower paying price & Stable click rate': 1.5,
            'Higher paying price & Stable click rate': 1.3,
            'Lower paying price & Unstable click rate': 1.2,
            'Higher paying price & Unstable click rate': 1.1
        }
    
    def get_category(self, ad_info):
        area, price = ad_info["area"], ad_info["paying_price"]
        is_stable = 60000 <= area <= 100000
        is_lower = price < self.avg_price
        return f"{'Lower' if is_lower else 'Higher'} paying price & {'Stable' if is_stable else 'Unstable'} click rate"
    
    def calculate_bid(self, ad_info):
        return max(self.alphas[self.get_category(ad_info)] * ad_info["paying_price"],
                  ad_info["ad_slot_floor_price"] * 1.05)

class XGBoostAgent2(BiddingAgent):
    def __init__(self, avg_price):
        super().__init__("XGBoost-2")
        self.avg_price = avg_price
        self.ml_alphas = {
            'Higher paying price & Stable click rate': 1.39,
            'Higher paying price & Unstable click rate': 1.04,
            'Lower paying price & Stable click rate': 1.53,
            'Lower paying price & Unstable click rate': 1.12
        }
        self.idx_vals = {
            'Higher paying price & Stable click rate': 1.1,
            'Higher paying price & Unstable click rate': 0.9,
            'Lower paying price & Stable click rate': 1.3,
            'Lower paying price & Unstable click rate': 1.0
        }
        self.pop_sizes = ['300x250', '728x90', '336x280', '160x600', '950x90']
    
    def get_category(self, ad_info):
        area, price = ad_info["area"], ad_info["paying_price"]
        is_stable = 60000 <= area <= 100000
        is_lower = price < self.avg_price
        return f"{'Lower' if is_lower else 'Higher'} paying price & {'Stable' if is_stable else 'Unstable'} click rate"
    
    def calculate_bid(self, ad_info):
        category = self.get_category(ad_info)
        alpha = self.ml_alphas[category]
        if ad_info["dimension_combo"] in self.pop_sizes: alpha *= 1.1
        if ad_info["ad_slot_visibility"] > 0: alpha *= 1.05
        return max(alpha * ad_info["paying_price"] * self.idx_vals[category], ad_info["ad_slot_floor_price"] * 1.05)

class LlamaAgent1(BiddingAgent):
    def __init__(self, avg_price):
        super().__init__("LLaMA-1")
        self.avg_price = avg_price
        self.size_cats = {
            'Lower paying price & Stable click rate': {
                'sizes': ['250x250'],
                'idx': 1.3
            },
            'Higher paying price & Stable click rate': {
                'sizes': ['160x600', '960x90', '336x280', '910x90'],
                'idx': 1.1
            },
            'Lower paying price & Unstable click rate': {
                'sizes': ['125x125', '234x60', '120x240', '180x600', '640x90', '180x150', '320x50', '120x600'],
                'idx': 1.0
            },
            'Higher paying price & Unstable click rate': {
                'sizes': ['468x60', '620x60', '200x200', '300x250', '300x300', '728x90', '1000x90', '950x90'],
                'idx': 0.9
            }
        }
        self.alphas = {
            'Lower paying price & Stable click rate': 1.3,
            'Higher paying price & Stable click rate': 1.1,
            'Lower paying price & Unstable click rate': 1.0,
            'Higher paying price & Unstable click rate': 0.9
        }
        self.opt_bids = {
            '1000x90': 90.82,
            '120x600': 61.27,
            '160x600': 39.79,
            '250x250': 41.61,
            '300x250': 107.38,
            '300x300': 94.19,
            '336x280': 60.15,
            '728x90': 69.11,
            '910x90': 120.45,
            '950x90': 98.93,
            '960x90': 110.22
        }
    
    def get_category_by_size(self, dimension):
        return next((cat for cat, info in self.size_cats.items() if dimension in info['sizes']), None)
    
    def get_category_by_area_price(self, ad_info):
        area, price = ad_info["area"], ad_info["paying_price"]
        is_stable = 60000 <= area <= 100000
        is_lower = price < self.avg_price
        return f"{'Lower' if is_lower else 'Higher'} paying price & {'Stable' if is_stable else 'Unstable'} click rate"
    
    def get_optimal_bid_factor(self, dimension, paying_price):
        std_dimension = f"{int(dimension.split('x')[0])}x{int(dimension.split('x')[1])}"
        if std_dimension in self.opt_bids:
            optimal_ratio = self.opt_bids[std_dimension] / paying_price if paying_price > 0 else 1.0
            return min(max(optimal_ratio * 0.9, 0.8), 1.5)
        return 1.0
    
    def calculate_bid(self, ad_info):
        dimension = ad_info["dimension_combo"]
        category = self.get_category_by_size(dimension) or self.get_category_by_area_price(ad_info)
        base_bid = self.alphas[category] * ad_info["paying_price"] * self.size_cats[category]['idx']
        optimal_factor = self.get_optimal_bid_factor(dimension, ad_info["paying_price"])
        adjusted_bid = base_bid * optimal_factor
        if ad_info["ad_slot_visibility"] > 0:
            adjusted_bid *= 1.05
        return max(adjusted_bid, ad_info["ad_slot_floor_price"] * 1.05)

class LlamaAgent2(BiddingAgent):
    def __init__(self, avg_price):
        super().__init__("LLaMA-2")
        self.avg_price = avg_price
        self.top_sizes = {
            '1000x90': {'alpha': 1.1, 'idx': 1.0, 'cat': 'Higher paying price & Unstable click rate', 'opt_bid': 90.82},
            '120x600': {'alpha': 1.2, 'idx': 1.0, 'cat': 'Lower paying price & Unstable click rate', 'opt_bid': 71.53},
            '160x600': {'alpha': 1.2, 'idx': 1.0, 'cat': 'Lower paying price & Unstable click rate', 'opt_bid': 56.40},
            '250x250': {'alpha': 1.2, 'idx': 1.0, 'cat': 'Lower paying price & Unstable click rate', 'opt_bid': 55.90},
            '300x250': {'alpha': 1.1, 'idx': 1.0, 'cat': 'Higher paying price & Unstable click rate', 'opt_bid': 107.38},
            '300x300': {'alpha': 1.1, 'idx': 1.0, 'cat': 'Higher paying price & Unstable click rate', 'opt_bid': 100.76},
            '336x280': {'alpha': 1.2, 'idx': 1.0, 'cat': 'Lower paying price & Unstable click rate', 'opt_bid': 66.26},
            '728x90': {'alpha': 1.1, 'idx': 1.0, 'cat': 'Higher paying price & Unstable click rate', 'opt_bid': 71.84},
            '910x90': {'alpha': 1.1, 'idx': 1.0, 'cat': 'Higher paying price & Unstable click rate', 'opt_bid': 120.45},
            '950x90': {'alpha': 1.1, 'idx': 1.0, 'cat': 'Higher paying price & Unstable click rate', 'opt_bid': 98.93},
            '960x90': {'alpha': 1.1, 'idx': 1.0, 'cat': 'Higher paying price & Unstable click rate', 'opt_bid': 110.22}
        }
        self.def_vals = {
            'Higher paying price & Stable click rate': {'alpha': 1.3, 'idx': 1.1},
            'Higher paying price & Unstable click rate': {'alpha': 1.1, 'idx': 1.0},
            'Lower paying price & Stable click rate': {'alpha': 1.5, 'idx': 1.3},
            'Lower paying price & Unstable click rate': {'alpha': 1.2, 'idx': 1.0}
        }
    
    def get_category(self, ad_info):
        area, price = ad_info["area"], ad_info["paying_price"]
        is_stable = 60000 <= area <= 100000
        is_lower = price < self.avg_price
        return f"{'Lower' if is_lower else 'Higher'} paying price & {'Stable' if is_stable else 'Unstable'} click rate"
    
    def calculate_bid(self, ad_info):
        dimension = ad_info["dimension_combo"]
        if dimension in self.top_sizes:
            size_info = self.top_sizes[dimension]
            base_bid = size_info['opt_bid']
            if ad_info["ad_slot_visibility"] > 0:
                base_bid *= 1.1
            return max(base_bid, ad_info["ad_slot_floor_price"] * 1.05)
        else:
            category = self.get_category(ad_info)
            alpha = self.def_vals[category]['alpha']
            idx = self.def_vals[category]['idx']
            return max(alpha * ad_info["paying_price"] * idx, ad_info["ad_slot_floor_price"] * 1.05)

class RTBSimulation:
    def __init__(self, data_path='processed_data_for_rl.csv', sample_size=5000):
        self.data = pd.read_csv(data_path).dropna(subset=['paying_price', 'ad_slot_floor_price'])
        self.data = self.data.sample(min(sample_size, len(self.data)), random_state=42)
        self.avg_price = self.data['paying_price'].mean()
        self.agents = [
            BaselineAgent(),
            XGBoostAgent1(self.avg_price),
            XGBoostAgent2(self.avg_price),
            LlamaAgent1(self.avg_price),
            LlamaAgent2(self.avg_price)
        ]
        print(f"Loaded {len(self.data)} records for simulation, average paying price: {self.avg_price:.2f}")
    
    def run(self):
        print("Starting bidding simulation...")
        for _, ad in tqdm(self.data.iterrows(), total=len(self.data)):
            ad_info = ad.to_dict()
            bids = [(agent, agent.calculate_bid(ad_info)) for agent in self.agents]
            bids.sort(key=lambda x: x[1], reverse=True)
            winner = bids[0]
            second_price = bids[1][1] if len(bids) > 1 else ad_info["ad_slot_floor_price"]
            
            if winner[1] >= ad_info["ad_slot_floor_price"]:
                actual_price = max(second_price, ad_info["ad_slot_floor_price"])
                for agent, bid_price in bids:
                    agent.record_result(ad_info, bid_price, agent == winner[0], actual_price if agent == winner[0] else 0)
            else:
                for agent, bid_price in bids:
                    agent.record_result(ad_info, bid_price, False, 0)
        
        print("Simulation completed!")
        return self.get_results()
    
    def get_results(self):
        results = [agent.get_metrics() for agent in self.agents]
        headers = ["Agent", "Bids", "Wins", "Win Rate", "Total Spend", "Revenue", "Net Value", "Avg Cost"]
        print("\n" + "="*100)
        print(f"{headers[0]:<12} {headers[1]:<10} {headers[2]:<10} {headers[3]:<8} {headers[4]:<12} {headers[5]:<12} {headers[6]:<12} {headers[7]:<10}")
        print("-"*100)
        
        for r in results:
            print(f"{r['name']:<12} {r['bids']:<10} {r['wins']:<10} {r['win_rate']:.4f} {r['spend']:<12.2f} {r['revenue']:<12.2f} {r['total_value']:<12.2f} {r['average_cost']:<10.2f}")
        
        print("\nTop 5 ad sizes won by each agent:")
        for agent in self.agents:
            if agent.wins > 0:
                size_counts = {}
                for dim in agent.imps:
                    size_counts[dim] = size_counts.get(dim, 0) + 1
                top_sizes = sorted(size_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"{agent.name}: {', '.join([f'{dim}({count})' for dim, count in top_sizes])}")
        
        return results
    
    def plot_results(self, results):
        plt.figure(figsize=(15, 10))
        names = [r['name'] for r in results]
        
        plt.subplot(2, 2, 1)
        win_rates = [r['win_rate'] for r in results]
        plt.bar(names, win_rates)
        plt.title('Win Rate Comparison')
        plt.ylim(0, max(win_rates) * 1.2)
        
        plt.subplot(2, 2, 2)
        spends = [r['spend'] for r in results]
        revenues = [r['revenue'] for r in results]
        x = np.arange(len(names))
        width = 0.35
        plt.bar(x - width/2, spends, width, label='Total Spend')
        plt.bar(x + width/2, revenues, width, label='Revenue')
        plt.xticks(x, names)
        plt.title('Spend vs Revenue Comparison')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        values = [r['total_value'] for r in results]
        plt.bar(names, values)
        plt.title('Net Value Comparison')
        
        plt.subplot(2, 2, 4)
        avg_costs = [r['average_cost'] for r in results]
        plt.bar(names, avg_costs)
        plt.title('Average Cost Comparison')
        
        plt.tight_layout()
        plt.savefig('rtb_simulation_results.png')
        print("\nResults chart has been saved as 'rtb_simulation_results.png'")

def main():
    simulation = RTBSimulation(data_path='processed_data_for_rl.csv', sample_size=5000)
    results = simulation.run()
    simulation.plot_results(results)

if __name__ == "__main__":
    main()