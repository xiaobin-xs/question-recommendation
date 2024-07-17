import torch

def calc_first_hit_perctg(all_pred_scores, all_candidate_lengths, all_labels):
    '''
    Calculate the percentage of observations where the correct candidate is ranked first.
    Only among the candidates generated for the corresponding query.
    '''
    first_hit, valid_obs = 0, 0 # Skip obs with <=1 candidates or no positive labels; i.e. only obs with >1 candidates and at least one positive label
    first_hit_v2, valid_obs_v2 = 0, 0 # Skip obs with <=1 candidates; i.e. only obs with >1 candidates, include obs with no positive labels
    first_hit_v3, valid_obs_v3 = 0, 0 # Include all obs
    # interate over batches
    for (pred_scores, candidate_lengths, labels) in zip(all_pred_scores, all_candidate_lengths, all_labels):
        # iterate over observations in the batch
        for i in range(pred_scores.size(0)):
            cand_len = candidate_lengths[i]
            # Get the valid candidates for the current observation
            score = pred_scores[i, :cand_len]
            label = labels[i, :cand_len]
            
            if label[score.argmax()] == 1:
                first_hit_v3 += 1
            valid_obs_v3 += 1
            
            if cand_len > 1: # so that there is at least two candidates to rank
                if label[score.argmax()] == 1:
                    first_hit_v2 += 1
                valid_obs_v2 += 1
                if label.sum() > 0: # so that there is a positive label
                    if label[score.argmax()] == 1:
                        first_hit += 1
                    valid_obs += 1
            
            # if cand_len <= 1 or label.sum() <= 0:
            #     continue  # Skip observations with <= one candidate, i.e., no other candidates to rank
            
            # if label[score.argmax()] == 1:
            #     first_hit += 1
            # valid_obs += 1

    return first_hit / valid_obs, first_hit_v2 / valid_obs_v2, first_hit_v3 / valid_obs_v3

# code adpated from: https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch/blob/master/trainers/utils.py#L27
def recalls_and_ndcgs_for_ks(scores, labels, ks):
    '''
    Compute Recall@k and NDCG@k for each k in the list of k's.
    Among a pre-defined candidates.
    '''
    metrics = {}

    scores = scores.to('cpu')
    labels = labels.to('cpu')
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    labels_sum = labels.sum(1)
    mask = labels_sum != 0
    print(f'{mask.sum()/len(mask):.3f} of observations have no positive labels')
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
       hits = labels_float.gather(1, cut)
       
       metrics['Recall@%d' % k] = \
           (hits.sum(1)[mask] / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1)[mask].float())).mean().cpu().item()

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float())
       dcg = (hits * weights.to(hits.device)).sum(1)
       idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
       ndcg = (dcg[mask] / idcg[mask]).mean()
       metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics