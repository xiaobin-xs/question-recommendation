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
            'candidate_labels': candidate_labels,
        }
        
class HDF5DatasetText(Dataset):
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
            current_query = obs_group['current_query_text'][()].decode('utf-8')
            query_history = [text.decode('utf-8') for text in obs_group['query_history_text'][()]]
            candidates = [text.decode('utf-8') for text in obs_group['candidate_queries_text'][()]]
            next_query = obs_group['next_query_text'][()].decode('utf-8')

            return {
                'current_query': current_query,
                'query_history': query_history,
                'candidates': candidates,
                'next_query': next_query
            }

    def get_all_candidates(self):
        all_candidates_text = []
        all_candidates_embed = []
        with h5py.File(self.hdf5_file, 'r') as h5f:
            for interactionId in list(h5f.keys()):
                obs_group = h5f[interactionId]
                candidate_text = [text.decode('utf-8') for text in obs_group['candidate_queries_text'][()]]
                candidate_embed = obs_group['candidate_embeddings'][:]
                all_candidates_text.extend(candidate_text)
                all_candidates_embed.extend(candidate_embed)
        all_candidates_embed = np.vstack(all_candidates_embed)
        return all_candidates_text, all_candidates_embed

def chat_collate_fn(batch, max_history_length=-1):
    '''
    Only look at each query's own candidates (as compared to look at candidates from one batch)
    '''
    current_queries = torch.stack([item['current_query'] for item in batch])
    batch_size = current_queries.size(0)
    embed_size = current_queries.size(1)

    query_histories = [item['query_history'] for item in batch]
    if max_history_length > 0:
        query_histories = [hist[:max_history_length] for hist in query_histories]
    history_lengths = torch.tensor([len(hist) for hist in query_histories])
    max_history_length = max(1, max(history_lengths).item()) # ensure there is at least one history, even if empty pad with all zeros
    padded_histories = [torch.cat([hist, torch.zeros(max_history_length - len(hist), embed_size)]) 
                        for hist in query_histories]
    padded_histories = torch.stack(padded_histories)
    
    candidates = [item['candidate_embeddings'] for item in batch]
    candidate_lengths = torch.tensor([len(cand) for cand in candidates])
    max_candidates_length = max(candidate_lengths).item()
    padded_candidates = [torch.cat([cand, torch.zeros(max_candidates_length - len(cand), embed_size)]) 
                         for cand in candidates]
    padded_candidates = torch.stack(padded_candidates)
    
    labels = [item['candidate_labels'] for item in batch]
    labels_for_pad_cand = [torch.cat([label, torch.zeros(max_candidates_length - len(label))]) 
              for label in labels]
    labels_for_pad_cand = torch.stack(labels_for_pad_cand)
    
    candidate_lengths_cumsum = torch.cumsum(candidate_lengths, dim=0)
    candidate_lengths_cumsum = torch.cat((torch.tensor([0], dtype=candidate_lengths_cumsum.dtype), candidate_lengths_cumsum)) # insert 0 at the beginning
    batch_candidates = torch.vstack(candidates)
    total_num_batch_cand = batch_candidates.size(0)

    labels_for_all_cand = torch.zeros(len(labels), batch_candidates.size(0))
    for r, label in enumerate(labels):
        labels_for_all_cand[r, candidate_lengths_cumsum[r]:candidate_lengths_cumsum[r+1]] = label

    batch_candidates = batch_candidates.unsqueeze(0).expand(batch_size, -1, -1)
    candidate_lengths_for_batch_cand = total_num_batch_cand * torch.ones_like(candidate_lengths)

    return current_queries, padded_histories, history_lengths, padded_candidates, candidate_lengths, labels_for_pad_cand, \
        batch_candidates, candidate_lengths_for_batch_cand, labels_for_all_cand

def chat_collate_fn_batch_cand(batch, max_history_length=-1):
    '''
    Look at candidates from the batch (as compared to only look at each query's own candidates)
    '''
    current_queries = torch.stack([item['current_query'] for item in batch])
    batch_size = current_queries.size(0)
    embed_size = current_queries.size(1)

    query_histories = [item['query_history'] for item in batch]
    if max_history_length > 0:
        query_histories = [hist[:max_history_length] for hist in query_histories]
    history_lengths = torch.tensor([len(hist) for hist in query_histories])
    max_history_length = max(1, max(history_lengths).item()) # ensure there is at least one history, even if empty pad with all zeros
    padded_histories = [torch.cat([hist, torch.zeros(max_history_length - len(hist), embed_size)]) 
                        for hist in query_histories]
    padded_histories = torch.stack(padded_histories)
    
    candidates = [item['candidate_embeddings'] for item in batch]
    candidate_lengths = torch.tensor([len(cand) for cand in candidates])
    candidate_lengths_cumsum = torch.cumsum(candidate_lengths, dim=0)
    candidate_lengths_cumsum = torch.cat((torch.tensor([0], dtype=candidate_lengths_cumsum.dtype), candidate_lengths_cumsum)) # insert 0 at the beginning
    batch_candidates = torch.vstack(candidates)
    total_num_batch_cand = batch_candidates.size(0)
    labels = [item['candidate_labels'] for item in batch]

    labels_for_all_cand = torch.zeros(len(labels), batch_candidates.size(0))
    for r, label in enumerate(labels):
        labels_for_all_cand[r, candidate_lengths_cumsum[r]:candidate_lengths_cumsum[r+1]] = label

    batch_candidates = batch_candidates.unsqueeze(0).expand(batch_size, -1, -1)
    candidate_lengths = total_num_batch_cand * torch.ones_like(candidate_lengths)
    
    return current_queries, padded_histories, history_lengths, batch_candidates, candidate_lengths, labels_for_all_cand

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
    preprocessed_data_filename = args.preprocessed_data_filename # 'chat_preprocessed'
    max_history_len = args.max_history_len
    batch_size = args.batch_size
    # candidate_scope = args.candidate_scope

    # if candidate_scope == 'own':
    #     chat_collate_fn = chat_collate_fn_own_cand
    # elif candidate_scope == 'batch':
    #     chat_collate_fn = chat_collate_fn_batch_cand
    # else:
    #     raise ValueError(f"Invalid candidate scope: {candidate_scope}")

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


