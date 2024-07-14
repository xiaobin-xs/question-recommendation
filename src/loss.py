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
        batch_size, max_cand_len = scores.shape
        total_loss = 0.0
        valid_observations = 0

        self.bceloss(scores, labels)

        # for i in range(batch_size):
        #     cand_len = candidate_lengths[i]
        #     if cand_len == 0:
        #         continue  # Skip observations with zero candidates

        #     # Get the valid candidates for the current observation
        #     score = scores[i, :cand_len]
        #     label = labels[i, :cand_len]

        #     curr_loss = self.bceloss(score.unsqueeze(0), label.unsqueeze(0))
        #     total_loss += curr_loss
        #     valid_observations += 1

        # if valid_observations == 0:
        #     return torch.tensor(0.0, requires_grad=True)  # Return 0 if no valid observations
        
        average_loss = total_loss / valid_observations
        return average_loss

class RecommendationLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(RecommendationLoss, self).__init__()
        self.margin = margin
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, scores, chosen_idx, ignored_idxs, all_ignored=False):
        # Scores is a tensor of shape (batch_size, num_candidates)
        
        # Ranking Loss: Hinge loss
        if all_ignored:
            # If all recommended questions are ignored, treat all as negative examples
            ignored_scores = scores  # All scores are considered as ignored
            chosen_scores = torch.full_like(scores[:, 0], -self.margin)  # Use a negative margin to indicate no selection
        else:
            chosen_scores = scores[:, chosen_idx]  # Shape (batch_size,)
            ignored_scores = scores[:, ignored_idxs]  # Shape (batch_size, num_ignored)

        ranking_loss = torch.mean(torch.clamp(self.margin + ignored_scores - chosen_scores.unsqueeze(1), min=0))

        # Cross-Entropy Loss for recommendation classification
        if all_ignored:
            # If all recommended questions are ignored, set target to zeros
            target_labels = torch.zeros_like(scores)
            ce_loss = self.cross_entropy_loss(scores, target_labels)
        else:
            ce_loss = self.cross_entropy_loss(scores, chosen_idx)

        # Combine losses with predefined weights
        combined_loss = ranking_loss + ce_loss  # Adjust weights as needed
        return combined_loss