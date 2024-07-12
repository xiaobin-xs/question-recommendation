import h5py, os, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import prepare_chat_data

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        with h5py.File(self.hdf5_file, 'r') as h5f:
            self.interactionId_list = list(h5f.keys())
            self.num_observations = len(self.interactionId_list)

    def __len__(self):
        return self.num_observations

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as h5f:
            obs_group = h5f[self.interactionId_list[idx]]
            
            current_query = torch.tensor(obs_group['current_query'][:], dtype=torch.float32)
            query_history = torch.tensor(obs_group['query_history'][:], dtype=torch.float32)
            candidate_embeddings = torch.tensor(obs_group['candidate_embeddings'][:], dtype=torch.float32)
            candidate_labels = torch.tensor(obs_group['candidate_labels'][:], dtype=torch.long)

            return {
                'current_query': current_query,
                'query_history': query_history,
                'candidate_embeddings': candidate_embeddings,
                'candidate_labels': candidate_labels
            }

def chat_collate_fn(batch, max_history_length=-1):
    current_queries = torch.stack([item['current_query'] for item in batch])
    embed_size = current_queries.size(1)

    query_histories = [item['query_history'] for item in batch]
    if max_history_length > 0:
        query_histories = [hist[:max_history_length] for hist in query_histories]
    history_lengths = torch.tensor([len(hist) for hist in query_histories])
    max_history_length = max(history_lengths)
    padded_histories = [torch.cat([hist, torch.zeros(max_history_length - len(hist), embed_size)]) 
                        for hist in query_histories]
    padded_histories = torch.stack(padded_histories)
    
    candidates = [item['candidate_embeddings'] for item in batch]
    candidate_lengths = [len(cand) for cand in candidates]
    max_candidates_length = max(candidate_lengths)
    padded_candidates = [torch.cat([cand, torch.zeros(max_candidates_length - len(cand), embed_size)]) 
                         for cand in candidates]
    padded_candidates = torch.stack(padded_candidates)
    
    labels = [torch.cat([label, torch.zeros(max_candidates_length - len(label))]) 
              for label in [item['candidate_labels'] for item in batch]]
    labels = torch.stack(labels)
    
    return current_queries, padded_histories, history_lengths, padded_candidates, candidate_lengths, labels

'''
# Example dataloader usage:
batch_size = 32
max_history_length = 20
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                        collate_fn=lambda batch: chat_collate_fn(batch, max_history_length))
'''

def get_data_loaders(args):

    data_folder = args.data_folder
    raw_json_file = args.raw_json_file
    preprocessed_data_filename = 'chat_preprocessed'
    max_history_len = args.max_history_len
    batch_size = args.batch_size

    if os.path.exists(os.path.join(data_folder, raw_json_file)):
        # Preprocess the raw JSON file
        if not os.path.exists(os.path.join(data_folder, f'{preprocessed_data_filename}_test_{args.sentence_transformer_type}-seed_{args.seed}.h5')):
            print(f"Preprocessing data from {raw_json_file}...")
            prepare_chat_data(data_folder, raw_json_file, preprocessed_data_filename, args)
        else:
            print(f"Data already preprocessed data for {raw_json_file}, directly loading the preprocessed data...")
        train_dataset = HDF5Dataset(os.path.join(data_folder, f'{preprocessed_data_filename}_train_{args.sentence_transformer_type}-seed_{args.seed}.h5'))
        val_dataset   = HDF5Dataset(os.path.join(data_folder, f'{preprocessed_data_filename}_val_{args.sentence_transformer_type}-seed_{args.seed}.h5'))
        test_dataset  = HDF5Dataset(os.path.join(data_folder, f'{preprocessed_data_filename}_test_{args.sentence_transformer_type}-seed_{args.seed}.h5'))
        
        # to ensure reproducibility
        g = torch.Generator()
        g.manual_seed(args.seed)
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  collate_fn=lambda batch: chat_collate_fn(batch, max_history_len),
                                  worker_init_fn=seed_worker, generator=g)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                  collate_fn=lambda batch: chat_collate_fn(batch, max_history_len),
                                  worker_init_fn=seed_worker, generator=g)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                  collate_fn=lambda batch: chat_collate_fn(batch, max_history_len),
                                  worker_init_fn=seed_worker, generator=g)
        # print train, val, test dataloader sizes
        embed_size = train_dataset[0]['current_query'].size(0)
        print(f"Using {args.sentence_transformer_type} sentence transformer with embedding size {embed_size}")
        print(f"Train dataloader size: {len(train_loader)}")
        print(f"Val dataloader size: {len(val_loader)}")
        print(f"Test dataloader size: {len(test_loader)}")
    else:
        raise FileNotFoundError(f"File {raw_json_file} not found in {data_folder}")

    return train_loader, val_loader, test_loader, embed_size


