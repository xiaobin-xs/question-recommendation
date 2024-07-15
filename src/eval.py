import torch

def calc_first_hit_perctg(all_pred_scores, all_candidate_lengths, all_labels):
    '''
    Calculate the percentage of observations where the correct candidate is ranked first.
    '''
    first_hit = 0
    valid_obs = 0
    # interate over batches
    for (pred_scores, candidate_lengths, labels) in zip(all_pred_scores, all_candidate_lengths, all_labels):
        # iterate over observations in the batch
        for i in range(pred_scores.size(0)):
            cand_len = candidate_lengths[i]
            if cand_len <= 1 or labels[i].sum() <= 0:
                continue  # Skip observations with <= one candidate, i.e., no other candidates to rank

            # Get the valid candidates for the current observation
            score = pred_scores[i, :cand_len]
            label = labels[i, :cand_len]
            
            if label[score.argmax()] == 1:
                first_hit += 1
            valid_obs += 1

    return first_hit / valid_obs