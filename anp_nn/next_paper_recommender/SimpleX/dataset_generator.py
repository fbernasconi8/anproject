
import pandas as pd
import os

#CSV file path
about_csv = '/home/francesca/academic_network_project/anp_data/raw/about.csv'
cites_csv = '/home/francesca/academic_network_project/anp_data/raw/cites.csv'
papers_csv = '/home/francesca/academic_network_project/anp_data/raw/papers.csv'
publication_csv = '/home/francesca/academic_network_project/anp_data/raw/publication.csv'
authors_topic_2019_csv = '/home/francesca/academic_network_project/anp_data/raw/sorted_authors_topics_2019.csv'
papers_about_csv = '/home/francesca/academic_network_project/anp_data/raw/sorted_papers_about.csv'
writes_csv = '/home/francesca/academic_network_project/anp_data/raw/writes.csv'

about_df = pd.read_csv(about_csv)
cites_df = pd.read_csv(cites_csv)
papers_df = pd.read_csv(papers_csv)
publication_df = pd.read_csv(publication_csv)
authors_topic_2019_df = pd.read_csv(authors_topic_2019_csv)
papers_about_df = pd.read_csv(papers_about_csv)
writes_df = pd.read_csv(writes_csv)

#authors ID from 5 CSV files (if available)
csv_folder_path = '/home/francesca/academic_network_project/anp_data/split'
author_ids_dfs = []
column_names = ['author_id']
for i in range(0, 5):
    filename = f'authors_{i}.csv'
    filepath = os.path.join(csv_folder_path, filename)
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, header=None, names=column_names)
        author_ids_dfs.append(df)
    else:
        print(f"File {filepath} non esiste.")

if author_ids_dfs:
    author_ids_df = pd.concat(author_ids_dfs, ignore_index=True)
    all_author_ids = set(author_ids_df['author_id'].unique())
else:
    print("non sono stati salvati correttamente gli autori")

all_author_ids = all_author_ids | set(writes_df['author_id'].unique())

#'topic_id' in int, remove NaN
def ensure_int_topic_id(df, column='topic_id'):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df.dropna(subset=[column], inplace=True)
    df[column] = df[column].astype(int)

ensure_int_topic_id(about_df)
ensure_int_topic_id(authors_topic_2019_df)
ensure_int_topic_id(papers_about_df)

#####

author_citing_paper_df = pd.merge(
    writes_df,
    papers_df[['id', 'citations', 'year']],
    left_on='paper_id',
    right_on='id' 
)

author_citing_paper_df.rename(columns={
    'author_id': 'user_id',
    'id': 'item_id',
    'citations': 'paper_citations',
    'year': 'paper_year'
}, inplace=True)

author_citing_paper_df.drop(columns=['paper_id'], inplace=True)

#####

#save .inter
data_path = '/home/francesca/academic_network_project/anp_nn/anp_data/data_path'  
if not os.path.exists(data_path):
    os.makedirs(data_path)
author_citing_paper_df.to_csv(os.path.join(data_path, 'academic_dataset_inter.csv'), index=False)
print("academic_dataset_inter.csv creato correttamente.")

print("inizio conversione .csv to .inter")

csv_inter_path = '/home/francesca/academic_network_project/anp_nn/anp_data/data_path/academic_dataset_inter.csv'

interactions = pd.read_csv(csv_inter_path)

interactions['user_id'] = interactions['user_id'].astype(int)
interactions['item_id'] = interactions['item_id'].astype(int)
interactions['paper_citations'] = interactions['paper_citations'].astype(int)
interactions['paper_year'] = interactions['paper_year'].astype(int)

inter_field_types = {
    'user_id': 'token',
    'item_id': 'token',
    'paper_citations': 'float',
    'paper_year': 'float'
}

header_line = '\t'.join([f"{col}:{inter_field_types[col]}" for col in interactions.columns])

#save
inter_path = '/home/francesca/academic_network_project/anp_nn/anp_data/data_path/academic_dataset/academic_dataset.inter'
with open(inter_path, 'w') as f:
    f.write(header_line + '\n')
    interactions.to_csv(f, sep='\t', index=False, header=False)

print("Conversione .inter completata.")

#####   DATASET.USER

author_paper_counts = writes_df.groupby('author_id').size().reset_index(name='num_papers')

author_paper_counts.rename(columns={'author_id': 'user_id'}, inplace=True)

#save .user
author_paper_counts.to_csv(os.path.join(data_path, 'academic_dataset_user.csv'), index=False)
print("academic_dataset_user.csv creato correttamente.")

#####

print("inizio conversione .csv to .user")

csv_user_path = '/home/francesca/academic_network_project/anp_nn/anp_data/data_path/academic_dataset_user.csv'

users = pd.read_csv(csv_user_path)

users['user_id'] = users['user_id'].astype(int)
users['num_papers'] = users['num_papers'].astype(int)

users_field_types = {
    'user_id': 'token',
    'num_papers': 'float',
}

header_line = '\t'.join([f"{col}:{users_field_types[col]}" for col in users.columns])

user_path = '/home/francesca/academic_network_project/anp_nn/anp_data/data_path/academic_dataset/academic_dataset.user'
with open(user_path, 'w') as f:
    f.write(header_line + '\n')
    users.to_csv(f, sep='\t', index=False, header=False)

print("Conversion .user completata.")

##### DATASET.ITEM

paper_citations = papers_df[['id', 'citations', 'year']].copy()

paper_citations.rename(columns={'id': 'item_id', 'citations': 'num_citations'}, inplace=True)

paper_citations.fillna(0, inplace=True)

#save.item
paper_citations.to_csv(os.path.join(data_path, 'academic_dataset_item.csv'), index=False)
print("academic_dataset.item creato correttamente.")

#####

print("inizio conversione .csv to .item")

csv_item_path = '/home/francesca/academic_network_project/anp_nn/anp_data/data_path/academic_dataset_item.csv'

items = pd.read_csv(csv_item_path)

items['item_id'] = items['item_id'].astype(int)
items['num_citations'] = items['num_citations'].astype(int)
items['year'] = items['year'].astype(int)

items_field_types = {
    'item_id': 'token',
    'num_citations': 'float',
    'year': 'float',
}

header_line = '\t'.join([f"{col}:{items_field_types[col]}" for col in items.columns])

item_path = '/home/francesca/academic_network_project/anp_nn/anp_data/data_path/academic_dataset/academic_dataset.item'
with open(item_path, 'w') as f:
    f.write(header_line + '\n')
    items.to_csv(f, sep='\t', index=False, header=False)

print("Conversion .item completata.")
