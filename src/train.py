import torch 
from model import QuestionRecommender
from loss import RecommendationLoss

def train(args, train_loader, val_loader):
    hidden_size = args.embed_size
    score = args.score

    model = QuestionRecommender(hidden_size, score='custom')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = RecommendationLoss()


    for epoch in range(args.epochs):
        train_one_epoch(args, model, train_loader, optimizer, criterion, epoch)
        evaluate(args, model, val_loader, criterion, epoch)

def train_one_epoch(args, model, data_loader, optimizer, criterion, epoch):
    model.train()
    for batch in data_loader:
        current_queries, padded_histories, history_lengths, padded_candidates, candidate_lengths, labels = batch
        optimizer.zero_grad()
        
        outputs = model(current_queries, padded_histories, history_lengths, padded_candidates)
        
        loss = 0
        for i in range(len(outputs)):
            loss += criterion(outputs[i], labels[i][:len(outputs[i])])
        
        loss.backward()
        optimizer.step()

def evaluate(args, model, data_loader, criterion, epoch, writer, mode='val'):
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            current_queries, padded_histories, history_lengths, padded_candidates, candidate_lengths, labels = batch
            outputs = model(current_queries, padded_histories, history_lengths, padded_candidates, candidate_lengths)
            
            loss = 0
            for i in range(len(outputs)):
                loss += criterion(outputs[i], labels[i][:len(outputs[i])])
            
            # Calculate metrics
            # metrics = calculate_metrics(outputs, labels)
            # log_metrics(metrics, epoch, mode, writer)
            # log_loss(loss, epoch, mode, writer)