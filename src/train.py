import os
import torch 
from torch.utils.tensorboard import SummaryWriter

from model import QuestionRecommender
from loss import RecommendationLoss
from logger import AverageMeterSet
from eval import calc_first_hit_perctg, recalls_and_ndcgs_for_ks
from data import HDF5DatasetText
from params import fix_random_seed_as

def train(args, train_loader, val_loader, test_loader):
    hidden_size = args.embed_size
    device = torch.device(args.device)
    fix_random_seed_as(args.seed)
    model = QuestionRecommender(hidden_size, lstm_dropout=args.lstm_dropout, score_fn=args.score_fn, fc_dropout=args.fc_dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = RecommendationLoss(margin_hinge=args.margin_hinge, weight_bce=args.weight_bce, weight_sim=args.weight_sim)

    accum_iter = 0
    model = model.to(device)
    writer = SummaryWriter(log_dir=os.path.join(args.root_dir, args.log_dir, 'runs'))
    best_val_recall_at10 = 0
    early_stopping_counter = 0

    for epoch in range(args.epochs):
        print('-'*25 + f'Epoch {epoch+1}' + '-'*25)
        accum_iter = train_one_epoch(args, model, train_loader, optimizer, criterion, epoch, accum_iter, device)
        torch.save(model.to('cpu').state_dict(), os.path.join(args.root_dir, args.log_dir, 'last_model.pth'))
        _ = evaluate(args, model, train_loader, criterion, epoch, writer, device, mode='Tra', inference=True)
        val_recall_at10 = evaluate(args, model, val_loader, criterion, epoch, writer, device, mode='Val')
        _ = evaluate(args, model, test_loader, criterion, epoch, writer, device, mode='Test')

        if val_recall_at10 > best_val_recall_at10:
            best_val_recall_at10 = val_recall_at10
            torch.save(model.to('cpu').state_dict(), os.path.join(args.root_dir, args.log_dir, 'best_model.pth')) # always save on cpu for compatibility
            print(f'Better val recall@10: {val_recall_at10:.3f}, best model saved.')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(f'Early stopping at epoch {epoch+1} after {args.patience} epochs without improvement.')
                break

def train_one_epoch(args, model, data_loader, optimizer, criterion, epoch, accum_iter, device):
    model.train()
    model = model.to(device)
    average_meter_set = AverageMeterSet()
    for batch in data_loader:
        current_queries, padded_histories, history_lengths, padded_candidates, candidate_lengths, labels, \
                batch_candidates, candidate_lengths_for_batch_cand, labels_for_all_cand = batch
        current_queries, padded_histories, history_lengths, padded_candidates, candidate_lengths, labels, \
                batch_candidates, candidate_lengths_for_batch_cand, labels_for_all_cand = \
                current_queries.to(device), padded_histories.to(device), history_lengths.to(device), \
                padded_candidates.to(device), candidate_lengths.to(device), labels.to(device), \
                batch_candidates.to(device), candidate_lengths_for_batch_cand.to(device), labels_for_all_cand.to(device)
        batch_size = current_queries.size(0)
        optimizer.zero_grad()
        if args.candidate_scope == 'own':
            scores, similarity_top_cand = model(current_queries, padded_histories, history_lengths, padded_candidates)
            loss, hinge_rank_loss, bce_loss, similarity_loss = criterion(scores, candidate_lengths, labels, similarity_top_cand)
        else:
            scores, similarity_top_cand = model(current_queries, padded_histories, history_lengths, batch_candidates)
            loss, hinge_rank_loss, bce_loss, similarity_loss = criterion(scores, candidate_lengths_for_batch_cand, labels_for_all_cand, similarity_top_cand)
        loss.backward()
        optimizer.step()

        average_meter_set.update('tra loss', loss.cpu().item())
        average_meter_set.update('tra hinge', hinge_rank_loss)
        average_meter_set.update('tra bce', bce_loss)
        accum_iter += batch_size

        # if accum_iter % 40 == 0:
        #     print('Epoch {}, running loss {:.3f} '.format(epoch+1, average_meter_set['tra loss'].avg))
    
    return accum_iter

def evaluate(args, model, data_loader, criterion, epoch, writer, device, mode='Val', inference=False):
    model.eval()
    model = model.to(device)
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
            current_queries, padded_histories, history_lengths, padded_candidates, candidate_lengths, labels, \
                batch_candidates, candidate_lengths_for_batch_cand, labels_for_all_cand = \
                current_queries.to(device), padded_histories.to(device), history_lengths.to(device), \
                padded_candidates.to(device), candidate_lengths.to(device), labels.to(device), \
                batch_candidates.to(device), candidate_lengths_for_batch_cand.to(device), labels_for_all_cand.to(device)
            if args.candidate_scope == 'own':
                scores, similarity_top_cand = model(current_queries, padded_histories, history_lengths, padded_candidates)
                loss, hinge_rank_loss, bce_loss, similarity_loss = criterion(scores, candidate_lengths, labels, similarity_top_cand)
            else:
                scores, similarity_top_cand = model(current_queries, padded_histories, history_lengths, batch_candidates)
                loss, hinge_rank_loss, bce_loss, similarity_loss = criterion(scores, candidate_lengths_for_batch_cand, labels_for_all_cand, similarity_top_cand)
                scores, similarity_top_cand = model(current_queries, padded_histories, history_lengths, padded_candidates)

            average_meter_set.update(f'{mode} loss', loss.cpu().item())
            average_meter_set.update(f'{mode} hinge', hinge_rank_loss)
            average_meter_set.update(f'{mode} bce', bce_loss)
            average_meter_set.update(f'{mode} similarity', -similarity_loss)

            all_scores.append(scores)
            all_candidate_lengths.append(candidate_lengths)
            all_labels.append(labels)

            all_candidates_no_pad.extend([padded_candidate[:candidate_length] for padded_candidate, candidate_length in zip(padded_candidates, candidate_lengths)])
            all_labels_no_pad.extend([label[:candidate_length] for label, candidate_length in zip(labels, candidate_lengths)])
    
    all_scores_for_all_cand, all_labels_for_all_cand = \
        infer_with_all_cand(all_candidates_no_pad, all_labels_no_pad, all_candidate_lengths, data_loader, model, args, device)
    print(f'Selecting among {all_scores_for_all_cand.size(1)} candidates.')
    first_hit_perctg, first_hit_perctg_v2, first_hit_perctg_v3 = \
        calc_first_hit_perctg(all_scores, all_candidate_lengths, all_labels)
    metrics = recalls_and_ndcgs_for_ks(all_scores_for_all_cand, all_labels_for_all_cand, args.ks)

    print(f'Epoch {epoch+1}, {mode} first hit%: {first_hit_perctg:.3f}, {first_hit_perctg_v2:.3f}, {first_hit_perctg_v3:.3f}')
    print('Epoch {}, {} loss {:.3f}, hinge {:.3f}, bce {:.3f}, cosine {:.3f}'.format(epoch+1, mode, average_meter_set[f'{mode} loss'].avg,
                                                                                     average_meter_set[f'{mode} hinge'].avg,
                                                                                     average_meter_set[f'{mode} bce'].avg,
                                                                                     average_meter_set[f'{mode} similarity'].avg
                                                          ))
    print(' '*13 + ', '.join([f'Recall@{k}: {metrics[f"Recall@{k}"]:.3f}' for k in args.ks]))
    print(' '*13 + ', '.join([f'  NDCG@{k}: {metrics[f"NDCG@{k}"]:.3f}' for k in args.ks]))

    if writer:
        writer.add_scalar(f'{mode}/Loss', average_meter_set[f'{mode} loss'].avg, epoch)
        writer.add_scalar(f'{mode}/HingeLoss', average_meter_set[f'{mode} hinge'].avg, epoch)
        writer.add_scalar(f'{mode}/BCELoss', average_meter_set[f'{mode} bce'].avg, epoch)
        writer.add_scalar(f'{mode}/Similarity', average_meter_set[f'{mode} similarity'].avg, epoch)
        writer.add_scalar(f'{mode}/FirstHit%', first_hit_perctg, epoch)
        writer.add_scalar(f'{mode}/FirstHit%v2', first_hit_perctg_v2, epoch)
        writer.add_scalar(f'{mode}/FirstHit%v3', first_hit_perctg_v3, epoch)
        if 10 in args.ks:
            writer.add_scalar(f'{mode}/Recall@10', metrics['Recall@10'], epoch)
        if 5 in args.ks:
            writer.add_scalar(f'{mode}/Recall@5', metrics['Recall@5'], epoch)
        if 3 in args.ks:
            writer.add_scalar(f'{mode}/Recall@3', metrics['Recall@3'], epoch)

    # if inference:
    #     split = 'train' if mode == 'Tra' else mode.lower()
    #     dataset = HDF5DatasetText(os.path.join(args.root_dir, args.data_folder, 'preprocessed',
    #                                            f'{args.preprocessed_data_filename}_{split}_{args.sentence_transformer_type}-seed_{args.seed}.h5'))
    #     all_candidates_text, all_candidates_embed = dataset.get_all_candidates()
    #     pass # TODO: finish inference

    return metrics['Recall@10']
    
def infer_with_all_cand(all_candidates_no_pad, all_labels_no_pad, all_candidate_lengths, data_loader, model, args, device):
    # TODO: what about the candidates with no next query?
    model = model.to(device)
    all_candidates_no_pad = torch.vstack(all_candidates_no_pad)
    all_labels_for_all_cand = torch.zeros(len(all_labels_no_pad), all_candidates_no_pad.size(0))
    all_candidate_lengths_flat = torch.concat(all_candidate_lengths)
    all_candidate_lengths_cumsum = torch.cumsum(all_candidate_lengths_flat, dim=0)
    all_candidate_lengths_cumsum = torch.cat((torch.tensor([0], dtype=all_candidate_lengths_cumsum.dtype, device=all_candidate_lengths_flat.device), 
                                              all_candidate_lengths_cumsum)) # insert 0 at the beginning
    for r, label in enumerate(all_labels_no_pad):
        all_labels_for_all_cand[r, all_candidate_lengths_cumsum[r]:all_candidate_lengths_cumsum[r+1]] = label

    all_scores_for_all_cand = []
    with torch.no_grad():
        for batch in data_loader:
            current_queries, padded_histories, history_lengths, _, _, _, _, _, _ = batch
            current_queries, padded_histories, history_lengths = \
                current_queries.to(device), padded_histories.to(device), history_lengths.to(device)
            batch_size = current_queries.size(0)
            scores, similarity_top_cand = model(current_queries, padded_histories, history_lengths, all_candidates_no_pad.unsqueeze(0).expand(batch_size, -1, -1))
            all_scores_for_all_cand.append(scores)
    all_scores_for_all_cand = torch.vstack(all_scores_for_all_cand)
    return all_scores_for_all_cand, all_labels_for_all_cand
