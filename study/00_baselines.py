import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader import DataLoader
from trainer import train_paraphrase, train_mvp, train_dlo
import json
import time

from helper import clean_up, create_output_directory

# nvidia-smi

dataloader = DataLoader("./datasets")

for i in range(5):
 for ds_name in ["gerest"]:
   for task in ["asqp"]:
         train_ds = dataloader.load_data(ds_name, "train", cv=False, target=task)
         test_ds = dataloader.load_data(ds_name, "test", cv=False, target=task)
      
         for ml_method in ["dlo"]: #mvp, dlo
            print(f"Task:", task, "Dataset:", ds_name, "Seed:", i, "ML-Method:", ml_method)
            filename = f"./generations/asqp/dlo/t5-large/training_{task}_{ds_name}seed-{i}_n-train_{ml_method}.json"
            # check if file already exists
            if os.path.exists(filename):
               print(f"File {filename} already exists. Skipping.")
               continue
            else:
            
               clean_up()
               create_output_directory()
              
               if ml_method == "paraphrase":
                  scores = train_paraphrase(train_ds=train_ds, test_ds=test_ds, seed=i, dataset=ds_name, task=task, model_name_or_path="t5-large", train_batch_size=8, test_batch_size=8)
               if ml_method == "mvp":
                  scores = train_mvp(train_ds=train_ds, test_ds=test_ds, seed=i, dataset=ds_name, task=task, model_name_or_path="t5-large", train_batch_size=4, eval_batch_size=4)
               if ml_method == "dlo":
                  scores = train_dlo(train_ds=train_ds, test_ds=test_ds, seed=i, dataset=ds_name, task=task, model_name_or_path="t5-large", train_batch_size=4, eval_batch_size=4)
    
               with open(filename, 'w', encoding='utf-8') as json_file:
                  json.dump(scores, json_file, ensure_ascii=False, indent=4)