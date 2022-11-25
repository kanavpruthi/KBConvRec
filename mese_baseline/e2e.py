import torch
from torch.utils.data import DataLoader

from transformers import GPT2Config, GPT2Tokenizer, BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from InductiveAttentionModels import GPT2InductiveAttentionHeadModel
from loss import SequenceCrossEntropyLoss

from trainer import Trainer
import time
import tqdm
from dataset import MovieRecDataset
from mese import UniversalCRSModel
from engine import Engine
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

train_path = "data/processed/durecdial2_sub_train_placeholder"
test_path = "data/processed/durecdial2_sub_dev_placeholder"
items_db_path = "data/processed/durecdial2_full_movie_db_placeholder"
items_db = torch.load(items_db_path)

train_dataset = MovieRecDataset(torch.load(train_path), bert_tokenizer, gpt_tokenizer)
test_dataset = MovieRecDataset(torch.load(test_path), bert_tokenizer, gpt_tokenizer)


# print(get_memory_free_MiB(0))

device = torch.device(0)


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

# print(get_memory_free_MiB(0))

# parameters
batch_size = 1
num_epochs = 8
num_gradients_accumulation = 1
num_train_optimization_steps = len(train_dataset) * num_epochs // batch_size // num_gradients_accumulation

num_samples_recall_train = 100
num_samples_rerank_train = 150
rerank_encoder_chunk_size = int(num_samples_rerank_train / 15)
validation_recall_size = 150

temperature = 1.2

language_loss_train_coeff = 0.5
language_loss_train_coeff_beginnging_turn = 1.0
recall_loss_train_coeff = 0.3
rerank_loss_train_coeff = 1.0

# loss
criterion_language = SequenceCrossEntropyLoss()
criterion_recall = torch.nn.CrossEntropyLoss()
rerank_class_weights = torch.FloatTensor([1] * (num_samples_rerank_train-1) + [30]).to(model.device)
criterion_rerank_train = torch.nn.CrossEntropyLoss(weight=rerank_class_weights)

# optimizer and scheduler
param_optimizer = list(model.language_model.named_parameters()) + \
    list(model.recall_encoder.named_parameters()) + \
    list(model.item_encoder.named_parameters()) + \
    list(model.recall_lm_query_mapper.named_parameters()) + \
    list(model.recall_item_wte_mapper.named_parameters()) + \
    list(model.rerank_item_wte_mapper.named_parameters()) + \
    list(model.rerank_logits_mapper.named_parameters())

no_decay = ['bias', 'ln', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = AdamW(optimizer_grouped_parameters, 
                  lr=3e-5,
                  eps=1e-06)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataset) // num_gradients_accumulation , num_training_steps = num_train_optimization_steps)

scaler = torch.cuda.amp.GradScaler()

progress_bar = tqdm.std.tqdm

# Data loader 

train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size, collate_fn=train_dataset.collate, pin_memory=True)
test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, collate_fn=test_dataset.collate, pin_memory=True)


engine = Engine(device,
                criterion_language,
                criterion_recall,
                criterion_rerank_train,
                language_loss_train_coeff,
                recall_loss_train_coeff,
                rerank_loss_train_coeff,
                num_samples_recall_train,
                num_samples_rerank_train,
                rerank_encoder_chunk_size,
                validation_recall_size,
                temperature)


output_file_path = "out/CRS_Train_Redial.txt"
model_saved_path = "runs/Redial_"

## Define Trainer
trainer = Trainer(
    model,
    engine,
    train_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
    scaler,
    progress_bar
)

# print(get_memory_free_MiB(4))
trainer.train(
    num_epochs,
    num_gradients_accumulation,
    batch_size,
    output_file_path,
    model_saved_path
)


total_sentences_original, total_sentences_generated, (valid_cnt, response_with_items, total_gen_cnt) = trainer.generate()
total_sentences_original = [item for sublist in total_sentences_original for item in sublist]
total_sentences_generated = [item for sublist in total_sentences_generated for item in sublist]
print(valid_cnt / total_gen_cnt, response_with_items / total_gen_cnt)
dist1, dist2, dist3, dist4 = distinct_metrics(total_sentences_generated)
bleu1, bleu2, bleu3, bleu4 = bleu_calc_all(total_sentences_original, total_sentences_generated)
print(dist1, dist2, dist3, dist4)
print(bleu1, bleu2, bleu3, bleu4)

torch.save(total_sentences_generated, 'human_eval/mese.pt')
torch.save(total_sentences_original,'human_eval/gold.pt')