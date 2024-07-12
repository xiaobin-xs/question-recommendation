import torch.nn as nn

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