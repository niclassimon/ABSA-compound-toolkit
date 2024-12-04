import os
import ast
import random
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

class DataLoader:
    def __init__(self, base_path="datasets"):
        self.base_path = base_path

    def load_data(self, name, data_type, cv=False, seed=42, target="asqp"):
        data_types = ["train", "test", "dev"] if data_type == "all" else [data_type]
        data = []

        for dtype in data_types:
            dataset_path = os.path.join(self.base_path, target, name, f"{dtype}.txt")
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file {dataset_path} not found.")
            
            with open(dataset_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            for idx, line in enumerate(lines):
                try:
                    text, aspects_str = line.split("####")
                    aspects = ast.literal_eval(aspects_str.strip())
                    aspect_list = []

                    for aspect in aspects:
                        aspect_dict = {
                            "aspect_term": aspect[0],
                            "aspect_category": aspect[1],
                            "polarity": aspect[2]
                        }
                        # Add 'opinion_term' only if target is 'asqp'
                        if target == "asqp":
                            aspect_dict["opinion_term"] = aspect[3]
                        aspect_list.append(aspect_dict)

                    if len(aspects) > 0:
                        data.append({
                            "id": f"{idx}_{name}_{dtype}",
                            "text": text.strip(),
                            "aspects": aspect_list,
                            "tuple_list": [tuple(aspect) for aspect in aspects]
                        })
                except ValueError as e:
                    print(f"Skipping line {idx} in {dtype} due to ValueError: {e}")
                    continue
        
        if cv:
            return self.random_cross_validation_split(data, seed)
        
        return data




    def random_cross_validation_split(self, data, seed=42):
        categories = [[el["aspect_category"] for el in an["aspects"]] for an in data]

        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(categories)

        n_splits = 5
        mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
        splits = []
        for train_index, test_index in mskf.split(np.zeros(len(Y)), Y):
           splits.append([data[i] for i in test_index])
    
    
        return splits