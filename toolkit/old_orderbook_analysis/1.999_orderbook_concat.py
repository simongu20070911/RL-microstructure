import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from utils import cleaner
dates =['23-Aug-2024','24-Aug-2024','25-Aug-2024','26-Aug-2024','27-Aug-2024','29-Aug-2024']
dates = ['08-Feb-2025','09-Feb-2025']
dates = ['08-Feb-2025','09-Feb-2025','10-Feb-2025','11-Feb-2025','12-Feb-2025','13-Feb-2025','14-Feb-2025','15-Feb-2025','16-Feb-2025','17-Feb-2025','18-Feb-2025','19-Feb-2025','20-Feb-2025','21-Feb-2025','22-Feb-2025','23-Feb-2025']



orderbook = []

for idx, date in enumerate(dates):
    orderbook.append(pd.read_csv(f'/home/gaen/Documents/billions/RL/data/input_data/{date}/orderbook.csv'))

for idx, date in enumerate(dates):
    orderbook[idx]['price'] = (orderbook[idx]['ask1']*orderbook[idx]['askqty1']+orderbook[idx]['bid1']*orderbook[idx]['bidqty1'])/(orderbook[idx]['askqty1']+orderbook[idx]['bidqty1'])

orderbook_all = pd.concat(orderbook)
orderbook_all.index = orderbook_all['datetime']

orderbook_all.shape
orderbook_all.to_csv('/home/gaen/Documents/billions/RL/data/orderbook.csv')

