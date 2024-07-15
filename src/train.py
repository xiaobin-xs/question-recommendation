import os
import torch 
from torch.utils.tensorboard import SummaryWriter

from model import QuestionRecommender
from loss import RecommendationLoss
from logger import AverageMeterSet
from eval import calc_first_hit_perctg
from embed import get_sentence_embedding_model

def train(args, train_loader, val_loader, test_loader):
    hidden_size = args.embed_size

    model = QuestionRecommender(hidden_size, lstm_dropout=args.lstm_dropout, score_fn=args.score_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = RecommendationLoss(margin_hinge=args.margin_hinge, weight_bce=args.weight_bce)

    accum_iter = 0
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'runs'))

    for epoch in range(args.epochs):
        print('-'*25 + f'Epoch {epoch+1}' + '-'*25)
        accum_iter = train_one_epoch(args, model, train_loader, optimizer, criterion, epoch, accum_iter)
        evaluate(args, model, train_loader, criterion, epoch, writer, mode='Tra')
        evaluate(args, model, val_loader, criterion, epoch, writer, mode='Val')
        evaluate(args, model, test_loader, criterion, epoch, writer, mode='Test')
        # evaluate(args, model, val_loader, criterion, epoch)

def train_one_epoch(args, model, data_loader, optimizer, criterion, epoch, accum_iter):
    model.train()
    average_meter_set = AverageMeterSet()
    for batch in data_loader:
        current_queries, padded_histories, history_lengths, padded_candidates, candidate_lengths, labels = batch
        batch_size = current_queries.size(0)
        optimizer.zero_grad()
        scores = model(current_queries, padded_histories, history_lengths, padded_candidates)
        loss, hinge_rank_loss, bce_loss = criterion(scores, candidate_lengths, labels)
        loss.backward()
        optimizer.step()

        average_meter_set.update('tra loss', loss.item())
        average_meter_set.update('tra hinge', hinge_rank_loss)
        average_meter_set.update('tra bce', bce_loss)
        accum_iter += batch_size

        if accum_iter % 50 == 0:
            print('Epoch {}, running loss {:.3f} '.format(epoch+1, average_meter_set['tra loss'].avg))

    return accum_iter

def evaluate(args, model, data_loader, criterion, epoch, writer, mode='Val'):
    model.eval()
    average_meter_set = AverageMeterSet()
    all_scores = []
    all_candidate_lengths = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            current_queries, padded_histories, history_lengths, padded_candidates, candidate_lengths, labels = batch
            scores = model(current_queries, padded_histories, history_lengths, padded_candidates)
            loss, hinge_rank_loss, bce_loss = criterion(scores, candidate_lengths, labels)

            average_meter_set.update(f'{mode} loss', loss.item())
            average_meter_set.update(f'{mode} hinge', hinge_rank_loss)
            average_meter_set.update(f'{mode} bce', bce_loss)

            all_scores.append(scores)
            all_candidate_lengths.append(candidate_lengths)
            all_labels.append(labels)

        # TODO: use candidates from all observations to compute metrics
        # Have a list of candidates in raw format (text) for train/val/test, 
        # together with labels indicating which is the correct next query for each observation.
        # Use the scores from the model to rank the candidates for each observation.
        # Compute metrics like MRR, Recall@k, etc. using the ranked candidates and the labels.
    
    first_hit_perctg = calc_first_hit_perctg(all_scores, all_candidate_lengths, all_labels)
    print(f'Epoch {epoch+1}, {mode} first hit%: {first_hit_perctg:.3f}')
    print('Epoch {}, {} loss {:.3f}, hinge {:.3f}, bce {:.3f}'.format(epoch+1, mode, average_meter_set[f'{mode} loss'].avg,
                                                                         average_meter_set[f'{mode} hinge'].avg,
                                                                         average_meter_set[f'{mode} bce'].avg
                                                          ))

    if writer:
        writer.add_scalar(f'{mode}/Loss', average_meter_set[f'{mode} loss'].avg, epoch)
        writer.add_scalar(f'{mode}/HingeLoss', average_meter_set[f'{mode} hinge'].avg, epoch)
        writer.add_scalar(f'{mode}/BCELoss', average_meter_set[f'{mode} bce'].avg, epoch)
        writer.add_scalar(f'{mode}/FirstHit%', first_hit_perctg, epoch)

# def embed_query(query, args):
#     embed_type, embed_model, sent_trans_embed_size = \
#         get_sentence_embedding_model(args)
#     encode_func = embed_model.embed_documents if embed_type == 'SentenceTransformerEmbeddings' else embed_model.encode
#     query_embedding = ( encode_func([query]) ) [0]
