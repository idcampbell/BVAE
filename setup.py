import os
import json

with open("params.json") as json_file:  
    params = json.load(json_file)
    conds = params["conds"]
    
if __name__ == '__main__':
    os.makedirs("CSVs")
    os.makedirs("Data")
    for cond in conds:
        os.makedirs("Output/" + cond, exist_ok=True)
        os.makedirs("Logs/" + cond, exist_ok=True)
        os.makedirs("Checkpoints/" + cond, exist_ok=True)
