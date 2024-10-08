import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import os, h5py

from embed import get_sentence_embedding_model

def prepare_chat_data(data_folder, save_file_name, args, raw_json_file=None):
    seed = args.seed
    
    if raw_json_file is None: # process multiple json files from a dir
        df, last_date = process_chat_multi_jsons(data_folder, args, remove_punc=False)
    else: # process a single json file
        df = process_chat_json(data_folder, raw_json_file, args)
        last_date = str(raw_json_file[-15:-5])
    chatId = df['chatId'].unique()
    sorted_chatId = np.sort(chatId)

    # Split the chatId into train, val, test
    np.random.seed(seed)
    np.random.shuffle(sorted_chatId)
    train_chatId = sorted_chatId[:int(0.7*len(sorted_chatId))]
    val_chatId = sorted_chatId[int(0.7*len(sorted_chatId)):int(0.85*len(sorted_chatId))]
    test_chatId = sorted_chatId[int(0.85*len(sorted_chatId)):]

    embed_type, embed_model, sent_trans_embed_size = \
        get_sentence_embedding_model(args)
    encode_func = embed_model.embed_documents if embed_type == 'SentenceTransformerEmbeddings' else embed_model.encode

    for chatId_list, split in zip([train_chatId, val_chatId, test_chatId], ['train', 'val', 'test']):
        df_split = df[df['chatId'].isin(chatId_list)]
        with h5py.File(os.path.join(args.root_dir, data_folder, 'preprocessed', 
                                    f'{save_file_name}_{split}_{args.sentence_transformer_type}_{last_date}-seed_{args.seed}.h5'), 'w') as h5f:
            for interactionId in tqdm(df_split['id'].tolist()):
                curr_data = df_split[df_split['id'] == interactionId]
                chatId = curr_data['chatId'].values[0]
                current_query = curr_data['Query'].values[0]
                next_query = curr_data['next_query'].values[0]
                query_history = curr_data['query_history'].values[0]
                candidate_query = curr_data['prompt_suggestions'].values[0]
                candidate_query = candidate_query if isinstance(candidate_query, list) else []
                candidate_labels = [1 if q == next_query else 0 for q in candidate_query]
                if sum(candidate_labels) == 0 and next_query is not None:
                    candidate_query.append(next_query)
                    candidate_labels.append(1)
                if len(candidate_labels) < 1:
                    continue
                
                # Create a group for each observation
                obs_group = h5f.create_group(interactionId)
                obs_group.create_dataset('chatId', data=chatId)

                # Add raw text
                obs_group.create_dataset('current_query_text', data=current_query.encode('utf-8'))
                obs_group.create_dataset('query_history_text', data=[seq.encode('utf-8') for seq in query_history])
                obs_group.create_dataset('candidate_queries_text', data=[seq.encode('utf-8') for seq in candidate_query])
                obs_group.create_dataset('next_query_text', data=(next_query if next_query is not None else '').encode('utf-8'))
                
                # Add current query embedding
                current_query = ( encode_func([current_query]) ) [0]
                obs_group.create_dataset('current_query', data=current_query)

                # Add query history embeddings
                query_history = np.array([encode_func([seq])[0] for seq in query_history])
                obs_group.create_dataset('query_history', data=query_history)
                
                # Add candidate embeddings and labels
                candidate_embeddings = np.array([encode_func([seq])[0] for seq in candidate_query])
                obs_group.create_dataset('candidate_embeddings', data=candidate_embeddings)
                obs_group.create_dataset('candidate_labels', data=candidate_labels)

def process_chat_multi_jsons(path, args, remove_punc=True):
    '''
    by: 'chatId' or 'userId'
    '''
    data_path = os.path.join(args.root_dir, path)
    files = sorted([json_file for json_file in os.listdir(data_path) if json_file.endswith('.json')])
    dates = [file[-15:-5] for file in files]
    last_date = str(max(dates))
    
    data = []
    for file in files:
        # print(file)
        file_path = os.path.join(data_path, file)
        print(file_path)
        with open(file_path, 'r') as f:
            for line in f:
                data.extend(json.loads(line))

    columns = ['id', 'request', 'response', 'trace', 'createdDate', 'chatId', 'imsOrgId', 'createdBy', 'sandboxName', 'sandboxId', 'userFeedback', 'smeFeedback'] 
    dt = pd.DataFrame(columns=columns)
    for i in tqdm(range(len(data))):
        d = data[i]
        dt.loc[i] = [d.get(c, np.nan) for c in columns]
    # remove duplicates
    dt = dt.drop_duplicates(subset=['id'])


    dt.createdDate = pd.to_datetime(dt.createdDate)
    dt = dt.sort_values(['chatId', 'createdDate'], ascending=[True, True], ignore_index=True)

    dt['Intent'] = dt["trace"].apply(
            lambda x: safe_get(x, 'routing', {'route_id': 'RoutingNotPresent'})['route_id'])
    print(f'\t Total:   # of interactions: {dt.shape[0]}, # of chatIds: {dt.chatId.nunique()}, # of userIds: {dt.createdBy.nunique()}')
    dtconcept = dt[dt.Intent=='ConceptsQA']
    print(f'\t Concept: # of interactions: {dtconcept.shape[0]}, # of chatIds: {dtconcept.chatId.nunique()}, # of userIds: {dtconcept.createdBy.nunique()}')
    
    dt['Response'] = dt['response'].apply(lambda x: x[0]['message'])
    dt['Query'] =dt['request'].apply(lambda x: x['message'])
    dt['Query_rewrite'] = dt['trace'].apply(lambda x: safe_get(x, 'question_rewrite', {'task_response': 'No question_rewrite'})['task_response'])

    dt['chat_history'] = dt['trace'].apply(lambda x: safe_get(x, 'chat_history', []))
    dt['query_history'] = dt.chat_history.apply(lambda chat_history: [ch['user'] for ch in chat_history])

    dt['prompt_suggestions'] = dt.trace.apply(lambda x: parse_prompt_suggestions(x) if type(x) == dict else np.nan)
    dt['num_prompt_suggestions'] = dt.prompt_suggestions.apply(lambda x: len(x) if type(x) == list else 0)

    # For each row, get the next query in the same chatId
    dt['next_query'] = None
    for i in range(dt.shape[0]):
        if i < dt.shape[0]-1:
            if dt.loc[i, 'chatId'] == dt.loc[i+1, 'chatId'] and dt.loc[i+1, 'Intent'] == 'ConceptsQA' :
                dt.loc[i, 'next_query'] = dt.loc[i+1, 'Query']
    dt['next_query_in_suggestions'] = dt.apply(lambda x: x['next_query'] in x['prompt_suggestions'] if type(x['prompt_suggestions']) == list else False, axis=1)


    # if Query_rewrite is not present, use Query
    dt['Query'] = dt.apply(lambda x: x['Query'] if x['Query_rewrite'] == 'No question_rewrite' 
                                                    or x['next_query_in_suggestions'] is True 
                                                else x['Query_rewrite'], axis=1)
    
    # repeat again so that next_query is updated with Query_rewrite
    dt['next_query'] = None
    for i in range(dt.shape[0]):
        if i < dt.shape[0]-1:
            if dt.loc[i, 'chatId'] == dt.loc[i+1, 'chatId'] and dt.loc[i+1, 'Intent'] == 'ConceptsQA' :
                dt.loc[i, 'next_query'] = dt.loc[i+1, 'Query']
    
    if remove_punc:
        # remove punctuation and question mark, convert to lowercase
        dt['prompt_suggestions'] = dt.prompt_suggestions.apply(lambda x: [s.replace(r'[^\w\s]', '').lower() for s in x] if type(x) == list else x)
        dt['Query'] = dt['Query'].str.replace(r'[^\w\s]', '').str.lower()
        dt['next_query'] = dt['next_query'].apply(lambda x: x.replace(r'[^\w\s]', '').lower() if type(x) == str else x)

    dt = dt[dt.Intent == 'ConceptsQA']
    # chatId_with_conceptsQA = dt[dt.Intent == 'ConceptsQA']['chatId'].unique()
    # dt = dt[dt.chatId.isin(chatId_with_conceptsQA)]
    print(f'\t Prompt suggest in next query: {dt.next_query_in_suggestions.sum() / dt.shape[0]:.2%}')
    dt = dt[['id', 'createdDate', 'chatId', 'Query', 'query_history', 'next_query', 'prompt_suggestions', 'Intent', 'next_query_in_suggestions']]

    return dt, last_date

def process_chat_json(path, file, args, remove_punc=True):
    '''
    by: 'chatId' or 'userId'
    '''
    file_path = os.path.join(args.root_dir, path, file)
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    columns = ['id', 'request', 'response', 'trace', 'createdDate', 'chatId', 'imsOrgId', 'createdBy', 'sandboxName', 'sandboxId', 'userFeedback', 'smeFeedback'] 
    dt = pd.DataFrame(columns=columns)
    for i, d in enumerate(data[0]):
        dt.loc[i] = [d.get(c, np.nan) for c in columns]

    dt.createdDate = pd.to_datetime(dt.createdDate)
    dt = dt.sort_values(['chatId', 'createdDate'], ascending=[True, True], ignore_index=True)

    dt['Intent'] = dt["trace"].apply(
            lambda x: safe_get(x, 'routing', {'route_id': 'RoutingNotPresent'})['route_id'])
    dt['Response'] = dt['response'].apply(lambda x: x[0]['message'])
    dt['Query'] =dt['request'].apply(lambda x: x['message'])
    dt['Query_rewrite'] = dt['trace'].apply(lambda x: safe_get(x, 'question_rewrite', {'task_response': 'No question_rewrite'})['task_response'])

    dt['chat_history'] = dt['trace'].apply(lambda x: safe_get(x, 'chat_history', []))
    dt['query_history'] = dt.chat_history.apply(lambda chat_history: [ch['user'] for ch in chat_history])

    dt['prompt_suggestions'] = dt.trace.apply(lambda x: parse_prompt_suggestions(x) if type(x) == dict else np.nan)

    # For each row, get the next query in the same chatId
    dt['next_query'] = None
    for i in range(dt.shape[0]):
        if i < dt.shape[0]-1:
            if dt.loc[i, 'chatId'] == dt.loc[i+1, 'chatId']:
                dt.loc[i, 'next_query'] = dt.loc[i+1, 'Query']
    dt['next_query_in_suggestions'] = dt.apply(lambda x: x['next_query'] in x['prompt_suggestions'] if type(x['prompt_suggestions']) == list else False, axis=1)


    # if Query_rewrite is not present, use Query
    dt['Query'] = dt.apply(lambda x: x['Query'] if x['Query_rewrite'] == 'No question_rewrite' 
                                                    or x['next_query_in_suggestions'] is True 
                                                else x['Query_rewrite'], axis=1)
    
    if remove_punc:
        # remove punctuation and question mark, convert to lowercase
        dt['prompt_suggestions'] = dt.prompt_suggestions.apply(lambda x: [s.replace(r'[^\w\s]', '').lower() for s in x] if type(x) == list else x)
        dt['Query'] = dt['Query'].str.replace(r'[^\w\s]', '').str.lower()
        dt['next_query'] = dt['next_query'].apply(lambda x: x.replace(r'[^\w\s]', '').lower() if type(x) == str else x)

    dt = dt[dt.Intent == 'ConceptsQA']
    # chatId_with_conceptsQA = dt[dt.Intent == 'ConceptsQA']['chatId'].unique()
    # dt = dt[dt.chatId.isin(chatId_with_conceptsQA)]

    dt = dt[['id', 'createdDate', 'chatId', 'Query', 'query_history', 'next_query', 'prompt_suggestions', 'Intent', 'trace']]

    return dt


def parse_prompt_suggestions(trace):
    if 'question_answering' in trace.keys() and type(trace['question_answering']) == dict:
        if 'prompt_suggestions' in trace['question_answering'].keys() and trace['question_answering']['prompt_suggestions'] is not None:
            if len(trace['question_answering']['prompt_suggestions']) > 0:
                return trace['question_answering']['prompt_suggestions']
            else:
                return 'Empty prompt_suggestions'
                # return np.nan
        else:
            return 'No prompt_suggestions'
            # return np.nan
    else:
        return 'No question_answering'
        # return np.nan


def safe_get(input_dict, key, default='none'):
    if input_dict:
        if key in input_dict:
            if input_dict[key]:
                return input_dict[key]
    return default