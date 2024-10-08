import torch
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

# Function to encode sentences into embeddings using BERT
def encode_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    sentence_embedding = outputs.last_hidden_state.mean(dim=1)
    return sentence_embedding

def get_sentence_embedding_model(args):
    '''
    args.sentence_transformer_type is 'finetuned' or 'pretrained'
    model_name: str, ['fine-tuned', 'sentence-transformers/paraphrase-MiniLM-L6-v2', other models from sentence-transformers library]
    '''
    if args.sentence_transformer_type == 'finetuned':
        model_type = 'SentenceTransformerEmbeddings'
        model = SentenceTransformerEmbeddings(model_name=args.sentence_transformer_path)
        embed_size = 384
    else:
        model_type = 'SentenceTransformer'
        model = SentenceTransformer(args.sentence_transformer_path)
        embed_size = model.get_sentence_embedding_dimension()
    return model_type, model, embed_size