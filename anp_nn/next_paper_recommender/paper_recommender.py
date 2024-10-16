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

# Disabilita i messaggi di logging meno importanti
logging.getLogger().setLevel(logging.ERROR)

# Percorso al modello salvato
MODEL_PATH = None

# Configurazione
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

# Creazione del dataset
dataset = create_dataset(config)

# Preparazione dei dati (utilizziamo tutto il dataset per il test)
train_data, valid_data, test_data = data_preparation(config, dataset)

# Inizializza il modello
model = SimpleX(config, train_data.dataset).to(config['device'])

if MODEL_PATH and os.path.exists(MODEL_PATH):
    print(f"Caricamento del modello dal percorso: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=config['device'])
    model_state_dict = checkpoint['state_dict']
    model.load_state_dict(model_state_dict)
else:
    print("Nessun modello salvato trovato, si procede senza il caricamento di un checkpoint.")

model.eval()  # Imposta il modello in modalità valutazione

# Ottieni tutti gli utenti e gli item
user_ids = list(dataset.field2token_id[config['USER_ID_FIELD']].values())
item_ids = list(dataset.field2token_id[config['ITEM_ID_FIELD']].values())

# Filtra eventuali ID non validi (ID 0)
user_ids = [uid for uid in user_ids if uid != 0]
item_ids = [iid for iid in item_ids if iid != 0]

# Carica le informazioni sugli articoli (citazioni e anno)
item_features = dataset.get_item_feature()
item_info = pd.DataFrame({
    'item_id': item_features['item_id'].numpy(),
    'num_citations': item_features['num_citations'].numpy(),
    'year': item_features['year'].numpy()
}).set_index('item_id')


# Esempio: Prevedi per i primi 100 utenti e tutti gli items
num_users_to_predict = 100  # Puoi modificare questo valore in base alle tue esigenze
user_ids_to_predict = user_ids[:num_users_to_predict]

# Parametri per il batch processing
batch_size = 10  # Scegli un batch_size che il tuo sistema può gestire
top_n = 10       # numero di raccomandazioni per utente
num_items = len(item_ids)

all_user_ids = []
all_item_ids = []
all_scores = []

# Converti item_ids in un array NumPy per un accesso più efficiente
item_ids_array = np.array(item_ids)

# Processa gli utenti in batch
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
    
    # Per ogni utente nel batch
    for idx, user_id in enumerate(user_ids_batch):
        user_scores = scores[idx]
        
        # Ottieni i dati relativi a citazioni e anno per ciascun item
        num_citations = item_info.loc[item_ids_array, 'num_citations'].values
        years = item_info.loc[item_ids_array, 'year'].values

        # Se 'user_scores' ha un elemento in più, rimuovilo
        if len(user_scores) > len(item_ids_array):
            user_scores = user_scores[:len(item_ids_array)]

        # Verifica che tutte le lunghezze siano coerenti
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

        # Normalizza 'num_citations' e 'year' per portarli su una scala comune
        items_df['normalized_citations'] = (items_df['num_citations'] - items_df['num_citations'].min()) / (items_df['num_citations'].max() - items_df['num_citations'].min())
        items_df['normalized_year'] = (items_df['year'] - items_df['year'].min()) / (items_df['year'].max() - items_df['year'].min())

        # Calcola un punteggio combinato dando più peso a citazioni e anno recente
        items_df['combined_score'] = (
            0.4 * items_df['score'] + 
            0.4 * items_df['normalized_citations'] + 
            0.2 * items_df['normalized_year']
        )
        
        # Ordina gli articoli per 'combined_score' in ordine decrescente
        items_df = items_df.sort_values(by='combined_score', ascending=False)
        
        # Prendi i top N articoli
        top_items = items_df.head(top_n)
        
        all_user_ids.extend([user_id] * len(top_items))
        all_item_ids.extend(top_items['item_id'].values)
        all_scores.extend(top_items['combined_score'].values)


# Creare un DataFrame con i risultati
results_df = pd.DataFrame({
    'user_id': all_user_ids,
    'item_id': all_item_ids,
    'score': all_scores
})

# Converti gli ID numerici in token (se necessario)
results_df['user_id_token'] = dataset.id2token(config['USER_ID_FIELD'], results_df['user_id'])
results_df['item_id_token'] = dataset.id2token(config['ITEM_ID_FIELD'], results_df['item_id'])

# Ordina i risultati per utente e punteggio
results_df = results_df.sort_values(by=['user_id', 'score'], ascending=[True, False])

# Salviamo i risultati in un file CSV
output_file = 'predictions_output.csv'

# Scrivi il DataFrame dei risultati in un file CSV
results_df.to_csv(output_file, index=False)

print(f"Risultati salvati nel file {output_file}")
