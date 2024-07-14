import torch
import torch.nn as nn
import torch.nn.functional as F


# Sequential Model for User History
class HistoryEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(HistoryEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.attention = nn.Linear(hidden_size, 1)  # Attention layer to weigh history embeddings

    def forward(self, histories, history_lengths):
        # attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        # context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        # lstm_out, (hidden_state, _) = self.lstm(histories)

        # Pack the sequences # TODO: handle zero history lengths
        history_lengths_cp = history_lengths.clone()
        history_lengths_cp[history_lengths_cp == 0] = 1
        packed_histories = nn.utils.rnn.pack_padded_sequence(histories, history_lengths_cp, batch_first=True, enforce_sorted=False)
        
        # Pass through LSTM
        packed_outputs, (hidden, cell) = self.lstm(packed_histories)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        # Gather the last output for each sequence
        idx = (history_lengths_cp - 1).view(-1, 1).expand(len(history_lengths_cp), output.size(2)).unsqueeze(1)
        last_outputs = output.gather(1, idx).squeeze(1)

        last_outputs[history_lengths==0, :] = 0  # Zero out the outputs for sequences with zero length

        return last_outputs
    
class Score(nn.Module):
    def __init__(self, hidden_size):
        super(Score, self).__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
            )
        self.sigmoid = nn.Sigmoid()
            
    def forward(self, combined_embedding, candidate_embeddings):
        combined_embedding = combined_embedding.unsqueeze(1).expand_as(candidate_embeddings)
        x = torch.cat((combined_embedding, candidate_embeddings), dim=-1)
        scores = self.score_fn(x)
        scores = self.sigmoid(scores)
        scores = scores.squeeze(-1)
        return scores

# Main Recommendation Model
class QuestionRecommender(nn.Module):
    def __init__(self, hidden_size, score_fn='cosine'):
        super(QuestionRecommender, self).__init__()
        self.history_encoder = HistoryEncoder(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)  # Combine history and query embeddings
        if score_fn == 'cosine':
            self.score_fn = nn.CosineSimilarity(dim=-1)  # Compute cosine similarity
        else:
            self.score_fn = Score(hidden_size)

    def forward(self, current_query_embedding, history_embeddings, history_lengths, candidate_embeddings):
        # query, histories, history_lengths, candidates, candidate_lengths
        # Encode the user history
        history_context = self.history_encoder(history_embeddings, history_lengths)
        
        # Combine history context with current query embedding
        combined_context = torch.cat((history_context, current_query_embedding), dim=-1)
        combined_embedding = self.fc(combined_context)
        
        # Compute similarity scores with each candidate question
        scores = self.score_fn(combined_embedding, candidate_embeddings)
        return scores

