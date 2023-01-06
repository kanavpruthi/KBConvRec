import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
from torch.utils.data import DataLoader

from transformers import GPT2Config, GPT2Tokenizer, BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from InductiveAttentionModels import GPT2InductiveAttentionHeadModel
from loss import SequenceCrossEntropyLoss

from trainer import Trainer
import time
import tqdm
from dataset import MovieRecDataset, RecDataset
from corrected_mese import C_UniversalCRSModel
from corrected_engine import C_Engine
from utilities import get_memory_free_MiB
from metrics import distinct_metrics, bleu_calc_all

bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model_recall = DistilBertModel.from_pretrained('distilbert-base-uncased')
bert_model_rerank = DistilBertModel.from_pretrained('distilbert-base-uncased')
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2InductiveAttentionHeadModel.from_pretrained('gpt2')

REC_TOKEN = "[REC]"
REC_END_TOKEN = "[REC_END]"
SEP_TOKEN = "[SEP]"
PLACEHOLDER_TOKEN = "[MOVIE_ID]"
gpt_tokenizer.add_tokens([REC_TOKEN, REC_END_TOKEN, SEP_TOKEN, PLACEHOLDER_TOKEN])
gpt2_model.resize_token_embeddings(len(gpt_tokenizer)) 

test_path = "data/processed/durecdial2_all_dev_placeholder_updated"
items_db_path = "data/processed/durecdial2_full_entity_db_placeholder"
output_file_path = 'out/retrieval_results_best.txt'

items_db = torch.load(items_db_path)

test_dataset = RecDataset(torch.load(test_path), bert_tokenizer, gpt_tokenizer)

criterion_language = SequenceCrossEntropyLoss()


device = torch.device(0)


model = C_UniversalCRSModel(
    gpt2_model, 
    bert_model_recall, 
    bert_model_rerank, 
    gpt_tokenizer, 
    bert_tokenizer, 
    device, 
    items_db, 
    rec_token_str=REC_TOKEN, 
    rec_end_token_str=REC_END_TOKEN
)

CKPT = 'runs/Durecdial_6.pt'

model.to(device)

model.load_state_dict(torch.load(CKPT,map_location='cuda:0')) 


progress_bar = tqdm.std.tqdm

# parameters
batch_size = 1
validation_recall_size = 500

temperature = 1.2

# Data loader 
test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, collate_fn=test_dataset.collate)

engine = C_Engine(device,
                criterion_language=criterion_language,
                validation_recall_size = validation_recall_size,
                temperature = temperature)



## Define Trainer
trainer = Trainer(
    model,
    engine,
    test_dataloader = test_dataloader,
    progress_bar = progress_bar
)


trainer.validate(output_file_path)