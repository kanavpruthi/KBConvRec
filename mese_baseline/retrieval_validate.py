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
from engine import Engine 
from mese import UniversalCRSModel
from utilities import get_memory_free_MiB
from metrics import distinct_metrics, bleu_calc_all

CKPT = 'runs/mese.pt'
device = torch.device('cpu')


bert_tokenizer = DistilBertTokenizer.from_pretrained("../../../../offline_transformers/distilbert-base-uncased/tokenizer")
bert_model_recall = DistilBertModel.from_pretrained('../../../../offline_transformers/distilbert-base-uncased/model')
bert_model_rerank = DistilBertModel.from_pretrained('../../../../offline_transformers/distilbert-base-uncased/model')
gpt_tokenizer = GPT2Tokenizer.from_pretrained("../../../../offline_transformers/gpt2/tokenizer")
gpt2_model = GPT2InductiveAttentionHeadModel.from_pretrained('../../../../offline_transformers/gpt2/model')

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

if "new" in CKPT:
    test_dataset = RecDataset(torch.load(test_path), bert_tokenizer, gpt_tokenizer)
else:
    test_dataset = MovieRecDataset(torch.load(test_path), bert_tokenizer, gpt_tokenizer)

num_pos_classes_dev = 530
num_neg_classes_dev = 5777
pos_weight = num_neg_classes_dev/num_pos_classes_dev
criterion_language = SequenceCrossEntropyLoss()
criterion_goal = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
criterion_recall = torch.nn.CrossEntropyLoss()

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
num_samples_recall_train = 500
validation_recall_size = 500

temperature = 1.2

# Data loader 
test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, collate_fn=test_dataset.collate)

if "new" in CKPT:
    engine = C_Engine(device,
                    criterion_language=criterion_language,
                    criterion_recall = criterion_recall,
                    criterion_goal= criterion_goal,
                    num_samples_recall_train= num_samples_recall_train,
                    validation_recall_size = validation_recall_size,
                    temperature = temperature)
else:
    engine = Engine(device,
                    criterion_language=criterion_language,
                    criterion_recall = criterion_recall,
                    num_samples_recall_train= num_samples_recall_train,
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