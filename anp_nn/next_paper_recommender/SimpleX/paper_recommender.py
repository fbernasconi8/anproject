import os
import numpy as np
import torch
import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed
from recbole.model.general_recommender.simplex import SimpleX
from recbole.data.interaction import Interaction
from tqdm import tqdm
import logging

logging.getLogger().setLevel(logging.ERROR)

MODEL_PATH = None

config_dict = {
    'model': 'SimpleX',
    'dataset': 'academic_dataset',
    'data_path': '/home/francesca/academic_network_project/anp_nn/anp_data/data_path',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'field_separator': '\t',
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'TIME_FIELD': 'citation_year',
    'USER_FEATURE_FIELD': ['num_papers'],
    'ITEM_FEATURE_FIELD': ['num_citations'],
    'load_col': {
        'inter': ['user_id', 'item_id', 'paper_citations', 'paper_year'],
        'user': ['user_id', 'num_papers'],
        'item': ['item_id', 'num_citations', 'year'],
    },
    'field_type': {
        'user_id': 'token',
        'item_id': 'token',
        'paper_year': 'float',
        'num_papers': 'float',
        'num_citations': 'float',
        'paper_citations': 'float',
        'year': 'float'
    },
    'log_level': 'ERROR',
}

config = Config(model='SimpleX', dataset='academic_dataset', config_dict=config_dict)
init_seed(config['seed'], config['reproducibility'])

dataset = create_dataset(config)

train_data, valid_data, test_data = data_preparation(config, dataset)

model = SimpleX(config, train_data.dataset).to(config['device'])

if MODEL_PATH and os.path.exists(MODEL_PATH):
    print(f"Caricamento del modello dal percorso: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=config['device'])
    model_state_dict = checkpoint['state_dict']
    model.load_state_dict(model_state_dict)
else:
    print("Nessun modello salvato trovato, si procede senza il caricamento di un checkpoint.")

model.eval()

user_ids = list(dataset.field2token_id[config['USER_ID_FIELD']].values())
item_ids = list(dataset.field2token_id[config['ITEM_ID_FIELD']].values())

user_ids = [uid for uid in user_ids if uid != 0]
item_ids = [iid for iid in item_ids if iid != 0]

item_features = dataset.get_item_feature()
item_info = pd.DataFrame({
    'item_id': item_features['item_id'].numpy(),
    'num_citations': item_features['num_citations'].numpy(),
    'year': item_features['year'].numpy()
}).set_index('item_id')

num_users_to_predict = 100
user_ids_to_predict = user_ids[:num_users_to_predict]

#batch processing
batch_size = 10  # batch_size
top_n = 10       # number of recommendations per user
num_items = len(item_ids)

all_user_ids = []
all_item_ids = []
all_scores = []

#item_ids in NumPy array
item_ids_array = np.array(item_ids)

print("Inizio delle predizioni...")
for i in tqdm(range(0, len(user_ids_to_predict), batch_size)):
    user_ids_batch = user_ids_to_predict[i:i+batch_size]
    interaction = Interaction({config['USER_ID_FIELD']: torch.tensor(user_ids_batch)})
    interaction = interaction.to(config['device'])
    
    with torch.no_grad():
        scores = model.full_sort_predict(interaction)
        scores = scores.cpu().numpy()
        num_users_in_batch = len(user_ids_batch)
        scores = scores.reshape(num_users_in_batch, -1)
    
    #for every user
    for idx, user_id in enumerate(user_ids_batch):
        user_scores = scores[idx]
        
        num_citations = item_info.loc[item_ids_array, 'num_citations'].values
        years = item_info.loc[item_ids_array, 'year'].values

        if len(user_scores) > len(item_ids_array):
            user_scores = user_scores[:len(item_ids_array)]

        if len(user_scores) == len(item_ids_array) == len(num_citations) == len(years):
            items_df = pd.DataFrame({
                'item_id': item_ids_array,
                'score': user_scores,
                'num_citations': num_citations,
                'year': years
            })
        else:
            print("Le lunghezze dei dati non corrispondono. Verifica i dati:")
            print(f"Len user_scores: {len(user_scores)}")
            print(f"Len item_ids_array: {len(item_ids_array)}")
            print(f"Len num_citations: {len(num_citations)}")
            print(f"Len years: {len(years)}")
            raise ValueError("Le lunghezze dei dati non corrispondono.")

        items_df['normalized_citations'] = (items_df['num_citations'] - items_df['num_citations'].min()) / (items_df['num_citations'].max() - items_df['num_citations'].min())
        items_df['normalized_year'] = (items_df['year'] - items_df['year'].min()) / (items_df['year'].max() - items_df['year'].min())

        items_df['combined_score'] = (
            0.4 * items_df['score'] + 
            0.4 * items_df['normalized_citations'] + 
            0.2 * items_df['normalized_year']
        )
        
        #'combined_score' in descending order
        items_df = items_df.sort_values(by='combined_score', ascending=False)
        
        #top N articles
        top_items = items_df.head(top_n)
        
        all_user_ids.extend([user_id] * len(top_items))
        all_item_ids.extend(top_items['item_id'].values)
        all_scores.extend(top_items['combined_score'].values)


#DataFrame creation with results
results_df = pd.DataFrame({
    'user_id': all_user_ids,
    'item_id': all_item_ids,
    'score': all_scores
})

#IDs in token
results_df['user_id_token'] = dataset.id2token(config['USER_ID_FIELD'], results_df['user_id'])
results_df['item_id_token'] = dataset.id2token(config['ITEM_ID_FIELD'], results_df['item_id'])

#order based on the score
results_df = results_df.sort_values(by=['user_id', 'score'], ascending=[True, False])

#save results in a CSV file
output_file = 'predictions_output.csv'

#results DataFrame in a CSV file
results_df.to_csv(output_file, index=False)

print(f"Risultati salvati nel file {output_file}")
