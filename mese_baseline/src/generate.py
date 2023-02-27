import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
from torch.utils.data import DataLoader

from transformers import GPT2Config, GPT2Tokenizer, BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from model.inductive_attention_model import GPT2InductiveAttentionHeadModel
from loss import SequenceCrossEntropyLoss

from trainer import Trainer
import time
import tqdm
from dataset import MovieRecDataset, RecDataset
from old.model.mese import UniversalCRSModel
from old.engine import Engine
from model.mese import C_UniversalCRSModel
from engine import C_Engine
from utilities import get_memory_free_MiB
from metrics import distinct_metrics, bleu_calc_all

device = torch.device(0)

bert_tokenizer = DistilBertTokenizer.from_pretrained("../../../../offline_transformers/distilbert-base-uncased/tokenizer/")
bert_model_recall = DistilBertModel.from_pretrained('../../../../offline_transformers/distilbert-base-uncased/model/')
bert_model_rerank = DistilBertModel.from_pretrained('../../../../offline_transformers/distilbert-base-uncased/model/')
gpt_tokenizer = GPT2Tokenizer.from_pretrained("../../../../offline_transformers/gpt2/tokenizer/")
gpt2_model = GPT2InductiveAttentionHeadModel.from_pretrained('../../../../offline_transformers/gpt2/model/')

REC_TOKEN = "[REC]"
REC_END_TOKEN = "[REC_END]"
SEP_TOKEN = "[SEP]"
PLACEHOLDER_TOKEN = "[MOVIE_ID]"
gpt_tokenizer.add_tokens([REC_TOKEN, REC_END_TOKEN, SEP_TOKEN, PLACEHOLDER_TOKEN])
gpt2_model.resize_token_embeddings(len(gpt_tokenizer)) 

# train_path = "data/processed/durecdial2_full_train_placeholder"
test_path = "data/processed/durecdial2_all_dev_placeholder_updated"
items_db_path = "data/processed/durecdial2_full_entity_db_placeholder"
items_db = torch.load(items_db_path)

CKPT = 'runs/new_model_rec.pt'

if "new" in CKPT:
    test_dataset = RecDataset(torch.load(test_path), bert_tokenizer, gpt_tokenizer)
else:
    test_dataset = MovieRecDataset(torch.load(test_path), bert_tokenizer, gpt_tokenizer)



if "new" in CKPT:
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
else:
    model = UniversalCRSModel(
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



model.to(device)

model.load_state_dict(torch.load(CKPT,map_location=device)) 


progress_bar = tqdm.std.tqdm

# parameters
batch_size = 1
validation_recall_size = 500

temperature = 1.2

# Data loader 
test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, collate_fn=test_dataset.collate)

if "new" in CKPT:
    engine = C_Engine(device,
                    validation_recall_size = validation_recall_size,
                    temperature = temperature)
else:
    engine = Engine(device,
                    validation_recall_size = validation_recall_size,
                    temperature = temperature)



## Define Trainer
trainer = Trainer(
    model,
    engine,
    test_dataloader = test_dataloader,
    progress_bar = progress_bar
)


total_sentences_original, total_sentences_generated, (valid_cnt, response_with_items, total_gen_cnt) = trainer.generate()
total_sentences_original = [item for sublist in total_sentences_original for item in sublist]
total_sentences_generated = [item for sublist in total_sentences_generated for item in sublist]
torch.save(total_sentences_generated, 'human_eval/preds.pt')
torch.save(total_sentences_original,'human_eval/labels.pt')