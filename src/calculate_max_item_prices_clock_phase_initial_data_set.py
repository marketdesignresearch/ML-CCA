# %%
import json
import re

domains = ['GSVM', 'LSVM']
max_item_prices_CCA = {}

for domain in domains:


    price_file_name = 'values_for_null_price_seeds1-100'
    price_dict =  json.load(open(f'{domain}_{price_file_name}.json', 'r')) # AVG value per item



    
    all_bidders_max_linear_prices = {}
    for key in price_dict.keys():
        if 'max_value_per_item' in key:
            id = int(re.findall(r'\d+', key)[0])
            all_bidders_max_linear_prices[id] = price_dict[key]['mean']





    
    print(all_bidders_max_linear_prices)
    max_item_price = max(all_bidders_max_linear_prices.values())
    print(max_item_price)
    max_item_prices_CCA[domain] = max_item_price
    json.dump(max_item_prices_CCA,open('max_item_prices_CCA_seeds1-100.json','w'))

# %%
