import os
import torch 
from torch.utils.tensorboard import SummaryWriter

from model import QuestionRecommender
from loss import RecommendationLoss
from logger import AverageMeterSet
from eval import calc_first_hit_perctg, recalls_and_ndcgs_for_ks
from embed import get_sentence_embedding_model
from data import HDF5DatasetText
from params import fix_random_seed_as

def train(args, train_loader, val_loader, test_loader):
    hidden_size = args.embed_size
    fix_random_seed_as(args.seed)
    model = QuestionRecommender(hidden_size, lstm_dropout=args.lstm_dropout, score_fn=args.score_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = RecommendationLoss(margin_hinge=args.margin_hinge, weight_bce=args.weight_bce)

    accum_iter = 0
    model = model.to(args.device)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'runs'))

    for epoch in range(args.epochs):
        print('-'*25 + f'Epoch {epoch+1}' + '-'*25)
        accum_iter = train_one_epoch(args, model, train_loader, optimizer, criterion, epoch, accum_iter)
        evaluate(args, model, train_loader, criterion, epoch, writer, mode='Tra', inference=True)
        evaluate(args, model, val_loader, criterion, epoch, writer, mode='Val')
        evaluate(args, model, test_loader, criterion, epoch, writer, mode='Test')
        # evaluate(args, model, val_loader, criterion, epoch)

def train_one_epoch(args, model, data_loader, optimizer, criterion, epoch, accum_iter):
    model = model.to(args.device)
    model.train()
    average_meter_set = AverageMeterSet()
    for batch in data_loader:
        current_queries, padded_histories, history_lengths, padded_candidates, candidate_lengths, labels, \
                batch_candidates, candidate_lengths_for_batch_cand, labels_for_all_cand = batch
        current_queries, padded_histories, history_lengths, padded_candidates, candidate_lengths, labels, \
                batch_candidates, candidate_lengths_for_batch_cand, labels_for_all_cand = \
                current_queries.to(args.device), padded_histories.to(args.device), history_lengths.to(args.device), \
                padded_candidates.to(args.device), candidate_lengths.to(args.device), labels.to(args.device), \
                batch_candidates.to(args.device), candidate_lengths_for_batch_cand.to(args.device), labels_for_all_cand.to(args.device)
        batch_size = current_queries.size(0)
        optimizer.zero_grad()
        if args.candidate_scope == 'own':
            scores = model(current_queries, padded_histories, history_lengths, padded_candidates)
            loss, hinge_rank_loss, bce_loss = criterion(scores, candidate_lengths, labels)
        else:
            scores = model(current_queries, padded_histories, history_lengths, batch_candidates)
            loss, hinge_rank_loss, bce_loss = criterion(scores, candidate_lengths_for_batch_cand, labels_for_all_cand)
        loss.backward()
        optimizer.step()

        average_meter_set.update('tra loss', loss.cpu().item())
        average_meter_set.update('tra hinge', hinge_rank_loss)
        average_meter_set.update('tra bce', bce_loss)
        accum_iter += batch_size

        if accum_iter % 40 == 0:
            print('Epoch {}, running loss {:.3f} '.format(epoch+1, average_meter_set['tra loss'].avg))
    
    # TODO: save model checkpoint

    return accum_iter

def evaluate(args, model, data_loader, criterion, epoch, writer, mode='Val', inference=False):
    model.eval()
    average_meter_set = AverageMeterSet()
    all_scores = []
    all_candidate_lengths = []
    all_labels = []
    all_candidates_no_pad = []
    all_labels_no_pad = []
    with torch.no_grad():
        for batch in data_loader:
            current_queries, padded_histories, history_lengths, padded_candidates, candidate_lengths, labels, \
                batch_candidates, candidate_lengths_for_batch_cand, labels_for_all_cand = batch
            if args.candidate_scope == 'own':
                scores = model(current_queries, padded_histories, history_lengths, padded_candidates)
                loss, hinge_rank_loss, bce_loss = criterion(scores, candidate_lengths, labels)
            else:
                scores = model(current_queries, padded_histories, history_lengths, batch_candidates)
                loss, hinge_rank_loss, bce_loss = criterion(scores, candidate_lengths_for_batch_cand, labels_for_all_cand)
                scores = model(current_queries, padded_histories, history_lengths, padded_candidates)

            average_meter_set.update(f'{mode} loss', loss.cpu().item())
            average_meter_set.update(f'{mode} hinge', hinge_rank_loss)
            average_meter_set.update(f'{mode} bce', bce_loss)

            all_scores.append(scores)
            all_candidate_lengths.append(candidate_lengths)
            all_labels.append(labels)

            all_candidates_no_pad.extend([padded_candidate[:candidate_length] for padded_candidate, candidate_length in zip(padded_candidates, candidate_lengths)])
            all_labels_no_pad.extend([label[:candidate_length] for label, candidate_length in zip(labels, candidate_lengths)])

        # TODO: use candidates from all observations to compute metrics
        # Have a list of candidates in raw format (text) for train/val/test, 
        # together with labels indicating which is the correct next query for each observation.
        # Use the scores from the model to rank the candidates for each observation.
        # Compute metrics like MRR, Recall@k, etc. using the ranked candidates and the labels.
    
    
    all_scores_for_all_cand, all_labels_for_all_cand = \
        infer_with_all_cand(all_candidates_no_pad, all_labels_no_pad, all_candidate_lengths, data_loader, model)

    first_hit_perctg = calc_first_hit_perctg(all_scores, all_candidate_lengths, all_labels)
    metrics = recalls_and_ndcgs_for_ks(all_scores_for_all_cand, all_labels_for_all_cand, args.ks)

    print(f'Epoch {epoch+1}, {mode} first hit%: {first_hit_perctg:.3f}')
    print('Epoch {}, {} loss {:.3f}, hinge {:.3f}, bce {:.3f}'.format(epoch+1, mode, average_meter_set[f'{mode} loss'].avg,
                                                                         average_meter_set[f'{mode} hinge'].avg,
                                                                         average_meter_set[f'{mode} bce'].avg
                                                          ))
    print(' '*25 + ', '.join([f'Recall@{k}: {metrics[f"Recall@{k}"]:.3f}' for k in args.ks]))
    print(' '*25 + ', '.join([f'  NDCG@{k}: {metrics[f"NDCG@{k}"]:.3f}' for k in args.ks]))

    if writer:
        writer.add_scalar(f'{mode}/Loss', average_meter_set[f'{mode} loss'].avg, epoch)
        writer.add_scalar(f'{mode}/HingeLoss', average_meter_set[f'{mode} hinge'].avg, epoch)
        writer.add_scalar(f'{mode}/BCELoss', average_meter_set[f'{mode} bce'].avg, epoch)
        writer.add_scalar(f'{mode}/FirstHit%', first_hit_perctg, epoch)
        if 10 in args.ks:
            writer.add_scalar(f'{mode}/Recall@10', metrics['Recall@10'], epoch)

    if inference:
        split = 'train' if mode == 'Tra' else mode.lower()
        dataset = HDF5DatasetText(os.path.join(args.data_folder, 
                                               f'{args.preprocessed_data_filename}_{split}_{args.sentence_transformer_type}-seed_{args.seed}.h5'))
        all_candidates_text, all_candidates_embed = dataset.get_all_candidates()
        pass # TODO: finish inference
    
def infer_with_all_cand(all_candidates_no_pad, all_labels_no_pad, all_candidate_lengths, data_loader, model):
    # TODO: what about the candidates with no next query?
    all_candidates_no_pad = torch.vstack(all_candidates_no_pad)
    all_labels_for_all_cand = torch.zeros(len(all_labels_no_pad), all_candidates_no_pad.size(0))
    all_candidate_lengths_flat = torch.concat(all_candidate_lengths)
    all_candidate_lengths_cumsum = torch.cumsum(all_candidate_lengths_flat, dim=0)
    all_candidate_lengths_cumsum = torch.cat((torch.tensor([0], dtype=all_candidate_lengths_cumsum.dtype), all_candidate_lengths_cumsum)) # insert 0 at the beginning
    for r, label in enumerate(all_labels_no_pad):
        all_labels_for_all_cand[r, all_candidate_lengths_cumsum[r]:all_candidate_lengths_cumsum[r+1]] = label

    all_scores_for_all_cand = []
    with torch.no_grad():
        for batch in data_loader:
            current_queries, padded_histories, history_lengths, _, _, _, _, _, _ = batch
            batch_size = current_queries.size(0)
            scores = model(current_queries, padded_histories, history_lengths, all_candidates_no_pad.unsqueeze(0).expand(batch_size, -1, -1))
            all_scores_for_all_cand.append(scores)
    all_scores_for_all_cand = torch.vstack(all_scores_for_all_cand)
    return all_scores_for_all_cand, all_labels_for_all_cand

def embed_query(query, args, type='single_sent'):
    '''
    type: 'single_sent' or 'multi_sent'
    '''
    embed_type, embed_model, sent_trans_embed_size = \
        get_sentence_embedding_model(args)
    encode_func = embed_model.embed_documents if embed_type == 'SentenceTransformerEmbeddings' else embed_model.encode
    if type == 'single_sent':
        query_embedding = ( encode_func([query]) ) [0]
    elif type == 'multi-sent':
        query_embedding = [encode_func([seq])[0] for seq in query]
    return torch.tensor(query_embedding)

def inference(args, model, query, history, candidate, candidate_embed=None, next_query=None):
    '''
    Inference for a single observation.
    '''
    model.eval()
    query_embed = embed_query(query, args, type='single_sent')
    history_embed = embed_query(history, args, type='multi_sent')
    if candidate_embed is None:
        candidate_embed = embed_query(candidate, args, type='multi_sent')
    with torch.no_grad():
        scores = model(query_embed.unsqueeze(0), history_embed.unsqueeze(0), 
                       torch.tensor([len(history)]), candidate_embed.unsqueeze(0))
    # TODO: to be finished... 
    # (1) Rank candidates based on predicted scores; 
    # (2) extract the top-k candidates; 
    # (3) display the results; compute metrics
    # TODO: move this to another main file for inference
        


