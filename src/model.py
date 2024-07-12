import torch
import torch.nn as nn
import torch.nn.functional as F


# Sequential Model for User History
class HistoryEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(HistoryEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)  # Attention layer to weigh history embeddings

    def forward(self, embeddings):
        lstm_out, (hidden_state, _) = self.lstm(embeddings)
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        return context_vector

# Main Recommendation Model
class QuestionRecommender(nn.Module):
    def __init__(self, hidden_size):
        super(QuestionRecommender, self).__init__()
        self.history_encoder = HistoryEncoder(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)  # Combine history and query embeddings
        self.similarity = nn.CosineSimilarity(dim=-1)  # Compute cosine similarity

    def forward(self, history_embeddings, current_query_embedding, candidate_embeddings):
        # Encode the user history
        history_context = self.history_encoder(history_embeddings)
        
        # Combine history context with current query embedding
        combined_context = torch.cat((history_context, current_query_embedding), dim=-1)
        combined_embedding = self.fc(combined_context)
        
        # Compute similarity scores with each candidate question
        scores = self.similarity(combined_embedding.unsqueeze(1), candidate_embeddings)
        return scores

