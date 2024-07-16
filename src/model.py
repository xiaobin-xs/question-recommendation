import torch
import torch.nn as nn
import torch.nn.functional as F


# Sequential Model for User History
class HistoryEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1, num_layers=1):
        super(HistoryEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, histories, history_lengths):

        # Pack the sequences # TODO: handle zero history lengths
        history_lengths_cp = history_lengths.clone()
        history_lengths_cp[history_lengths_cp == 0] = 1
        packed_histories = nn.utils.rnn.pack_padded_sequence(histories, history_lengths_cp.cpu(), batch_first=True, enforce_sorted=False)
        
        # Pass through LSTM
        packed_outputs, (hidden, cell) = self.lstm(packed_histories)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        # Gather the last output for each sequence
        idx = (history_lengths_cp - 1).view(-1, 1).expand(len(history_lengths_cp), output.size(2)).unsqueeze(1)
        last_outputs = output.gather(1, idx).squeeze(1)

        last_outputs[history_lengths==0, :] = 0  # Zero out the outputs for sequences with zero length

        return last_outputs
    
class MLP(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1):
        super(MLP, self).__init__()
        hidden_size = (input_size + output_size) // 2
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    
class CosineSimilarityCalculator(nn.Module):
    def __init__(self, dim=2, eps=1e-8):
        super(CosineSimilarityCalculator, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=dim, eps=eps)
    
    def forward(self, combined_embedding, candidate_embeddings):
        # combined_embedding: (batch_size, hidden_size)
        # candidate_embeddings: (batch_size, num_candidates, hidden_size)
        combined_embedding = combined_embedding.unsqueeze(1).expand_as(candidate_embeddings)
        cosine_sim = self.cosine_similarity(candidate_embeddings, combined_embedding)
        return (cosine_sim + 1) / 2  # Normalize to [0, 1]
    
class Score(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(Score, self).__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
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
    def __init__(self, hidden_size, lstm_dropout=0.1, score_fn='cosine', fc_dropout=0.1):
        '''
        fc_dropout: dropout rate for the fully connected layer in the Score module, if score_fn is 'custom'
        '''
        super(QuestionRecommender, self).__init__()
        self.history_encoder = HistoryEncoder(hidden_size, hidden_size, dropout=lstm_dropout, num_layers=1)
        self.fc = MLP(hidden_size * 2, hidden_size, dropout=fc_dropout)  # Combine history and query embeddings
        if score_fn == 'cosine':
            self.score_fn = CosineSimilarityCalculator()  # Compute cosine similarity
        else:
            self.score_fn = Score(hidden_size, dropout=fc_dropout)

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

