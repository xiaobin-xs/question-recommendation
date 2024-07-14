import torch 
from model import QuestionRecommender
from loss import RecommendationLoss
from logger import AverageMeterSet

def train(args, train_loader, val_loader):
    hidden_size = args.embed_size
    score_fn = args.score_fn

    model = QuestionRecommender(hidden_size, score_fn='custom')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = RecommendationLoss(margin_hinge=args.margin_hinge, weight_bce=args.weight_bce)

    accum_iter = 0

    for epoch in range(args.epochs):
        accum_iter = train_one_epoch(args, model, train_loader, optimizer, criterion, epoch, accum_iter)
        # evaluate(args, model, val_loader, criterion, epoch)

def train_one_epoch(args, model, data_loader, optimizer, criterion, epoch, accum_iter):
    model.train()
    average_meter_set = AverageMeterSet()
    for batch in data_loader:
        current_queries, padded_histories, history_lengths, padded_candidates, candidate_lengths, labels = batch
        batch_size = current_queries.size(0)
        optimizer.zero_grad()
        scores = model(current_queries, padded_histories, history_lengths, padded_candidates)
        loss = criterion(scores, candidate_lengths, labels)
        loss.backward()
        optimizer.step()

        average_meter_set.update('tra loss', loss.item())
        accum_iter += batch_size

    print('Epoch {}, loss {:.3f} '.format(epoch+1, average_meter_set['tra loss'].avg))

    return accum_iter

def evaluate(args, model, data_loader, criterion, epoch, writer, mode='val'):
    model.eval()
    average_meter_set = AverageMeterSet()
    with torch.no_grad():
        for batch in data_loader:
            current_queries, padded_histories, history_lengths, padded_candidates, candidate_lengths, labels = batch
            pass