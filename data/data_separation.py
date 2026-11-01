import pandas as pd
import shutil

splits = pd.read_csv("splits.csv")
train_df = splits[splits["split"] == "train"] 
val_df = splits[splits["split"] == "val"] 
test_df = splits[splits["split"] == "test"]

for row in train_df.iterrows():
    row_info = row[1]
    filepath = row_info['filepath']
    filename = filepath.split('/')[4]
    label = row_info['label']
    
    shutil.copy(filepath, f'TrainingPetImages/{label}/{filename}')

for row in val_df.iterrows():
    row_info = row[1]
    filepath = row_info['filepath']
    filename = filepath.split('/')[4]
    label = row_info['label']
    
    shutil.copy(filepath, f'ValidationPetImages/{label}/{filename}')

for row in test_df.iterrows():
    row_info = row[1]
    filepath = row_info['filepath']
    filename = filepath.split('/')[4]
    label = row_info['label']
    
    shutil.copy(filepath, f'TestPetImages/{label}/{filename}')