import pandas as pd
import os
PATH ="/home/flask_app/trader_stats/data"
files = os.listdir(PATH)

for file in files:
    if 'csv' in file:
        df = pd.read_csv(PATH+"/"+file,header=0)
        df.to_parquet(PATH+"/"+file.split(".")[0]+".parquet")
