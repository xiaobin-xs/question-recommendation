import torch
import torch.nn as nn


class HingeRankLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(HingeRankLoss, self).__init__()
        self.margin = margin
    
    def forward(self, scores, candidate_lengths, labels):
        batch_size, max_cand_len = scores.shape
        total_loss = 0.0
        valid_observations = 0

        for i in range(batch_size):
            cand_len = candidate_lengths[i]
            if cand_len == 0:
                continue  # Skip observations with zero candidates

            # Get the valid candidates for the current observation
            score = scores[i, :cand_len]
            label = labels[i, :cand_len]

            # Get the indices of positive and negative labels
            pos_indices = (label == 1).nonzero(as_tuple=True)[0]
            neg_indices = (label == 0).nonzero(as_tuple=True)[0]

            if neg_indices.numel() == 0:
                continue  # Skip observations with no negative labels

            if label.sum() <= 0:
                chosen_scores = -self.margin # Use a negative margin to indicate no selection
            else:
                chosen_scores = score[pos_indices]
            ignored_scores = score[neg_indices]

            curr_loss = torch.clamp(self.margin + ignored_scores - chosen_scores, min=0)
            total_loss += curr_loss.mean()
            valid_observations += 1

        if valid_observations == 0:
            return torch.tensor(0.0, requires_grad=True)  # Return 0 if no valid observations
        
        average_loss = total_loss / valid_observations
        return average_loss
    
class BinaryCELoss(nn.Module):
    def __init__(self):
        super(BinaryCELoss, self).__init__()
        self.bceloss = nn.BCELoss(reduction='none')
    
    def forward(self, scores, candidate_lengths, labels):
        max_cand_len = scores.shape[1]
        indices = torch.arange(max_cand_len, device=scores.device)
        mask = indices < candidate_lengths.unsqueeze(1) # mask for non-padded candidates

        loss = self.bceloss(scores, labels)
        loss = (loss * mask).mean(1) / mask.sum(1)

        return loss.mean()


class RecommendationLoss(nn.Module):
    def __init__(self, margin_hinge=0.1, weight_bce=1.0, weight_sim=1.0):
        super(RecommendationLoss, self).__init__()
        self.hingeRankLoss = HingeRankLoss(margin=margin_hinge)
        self.bceLoss = BinaryCELoss()
        self.weight_bce = weight_bce
        self.weight_sim = weight_sim

    def forward(self, scores, candidate_lengths, labels, similarity_top_cand):
        # Scores: (batch_size, max_num_candidates)
        hinge_rank_loss = self.hingeRankLoss(scores, candidate_lengths, labels)
        bce_loss = self.bceLoss(scores, candidate_lengths, labels)
        similarity_loss = -similarity_top_cand.mean()
        combined_loss = hinge_rank_loss + self.weight_bce * bce_loss + self.weight_sim * similarity_loss
        return combined_loss, hinge_rank_loss.cpu().item(), bce_loss.cpu().item(), similarity_loss.cpu().item()