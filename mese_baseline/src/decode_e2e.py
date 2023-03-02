import json
import math
import argparse
import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from os import path
import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
from itertools import islice
from transformers import AutoTokenizer, AutoModelWithLMHead
from model.mese import C_UniversalCRSModel
from engine import C_Engine
from constrained_decoder.generate import generate
from constrained_decoder.utils import tokenize_constraints
from constrained_decoder.lexical_constraints import init_batch
from transformers import GPT2Config, GPT2Tokenizer, BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer
from model.inductive_attention_model import GPT2InductiveAttentionHeadModel
from utilities import replace_placeholder
from dataset import RecDataset
from torch.utils.data import DataLoader
from utilities import past_wtes_constructor

def parse_decoder_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_file",type=str,help='Output File to Store Predicted Sentences')
    parser.add_argument("--pretrained_model",type=str,help='Pretrained Model File to Use for Decoding')
    parser.add_argument("--kb_path",type=str, help= 'Location of the knowledge base to use')
    parser.add_argument("--test_file",type=str,help='Test file location to decode')
    parser.add_argument("--constraint_file", type=str, help="constraint file")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--beam_size', type=int, default=10,
                        help="Beam size for searching")
    parser.add_argument('--max_tgt_length', type=int, default=128,
                        help="maximum length of decoded sentences")
    parser.add_argument('--min_tgt_length', type=int, default=0,
                        help="minimum length of decoded sentences")
    parser.add_argument('--ngram_size', type=int, default=3,
                        help='all ngrams can only occur once')
    parser.add_argument('--length_penalty', type=float, default=0.6,
                        help="length penalty for beam search")

    parser.add_argument('--prune_factor', type=int, default=50,
                        help="fraction of candidates to keep based on score")
    parser.add_argument('--sat_tolerance', type=int, default=2,
                        help="minimum satisfied clause of valid candidates")

    # for A star deocding
    parser.add_argument('--look_ahead_step', type=int, default=5,
                        help="number of step to look ahead")
    parser.add_argument('--look_ahead_width', type=int, default=None,
                        help="width of beam in look ahead")
    parser.add_argument('--alpha', type=float, default=0.05,
                        help="decay factor for score in looking ahead")
    parser.add_argument('--fusion_t', type=float, default=None,
                        help="temperature to fuse word embedding for continuous looking ahead")
    parser.add_argument('--look_ahead_sample',  action='store_true',
                        help="whether use sampling for looking ahead")

    args = parser.parse_args()
    return args

CONSTRAINTS = [[['[MOVIE_ID]']]]
ENTITY_TOKEN = ['[MOVIE_ID]']

def get_special_tokens_ids(tokenizer):
    period_id = [tokenizer.convert_tokens_to_ids('.')]
    period_id.append(tokenizer.convert_tokens_to_ids('Ġ.'))
    eos_ids = [tokenizer.eos_token_id] + period_id
    bad_token = [':', "'", '-', '_', '@', 'Ċ', 'Ġ:', 'Ġwho', "'s"]
    bad_words_ids = [tokenizer.convert_tokens_to_ids([t]) for t in bad_token]

    return eos_ids, bad_words_ids


def generate_with_constraints(args, inputs, constraints_list, turn_num, eos_ids, bad_words_ids):
    constraints = init_batch(raw_constraints=constraints_list[turn_num:turn_num+1],
            key_constraints=constraints_list[turn_num:turn_num+1],
            beam_size=args.beam_size,
            eos_id=eos_ids)

    
    ###################################
    advanced_constraints = []
    for j, init_cons in enumerate(constraints):
        adv_cons = init_cons
        for token in inputs[j // args.beam_size]:
            adv_cons = adv_cons.advance(token)
        advanced_constraints.append(adv_cons)
    
    # print(len(advanced_constraints))
    # print(advanced_constraints[0])
    # raise('')
    ################################### 
    attention_mask = (~torch.eq(inputs, 628)).int()
    attention_mask = attention_mask.to(device)

    generated, _, _ = generate(   
                self=model.language_model,
                input_ids= inputs,
                attention_mask= attention_mask,
                pad_token_id=628,
                bad_words_ids=bad_words_ids,
                min_length=args.min_tgt_length,
                max_length=args.max_tgt_length,
                num_beams=args.beam_size,
                no_repeat_ngram_size=args.ngram_size,
                length_penalty=args.length_penalty,
                constraints=advanced_constraints,
                prune_factor=args.prune_factor,
                sat_tolerance=args.sat_tolerance,
                look_ahead_step=args.look_ahead_step,
                look_ahead_width=args.look_ahead_width,
                alpha=args.alpha,
                fusion_t=args.fusion_t,
                look_ahead_sample=args.look_ahead_sample,
                top_k=50,
                temperature=1.25,
                eos_token_id=628
        )
    
    return generated


def generate_without_constraints(inputs):
    generated = model.language_model.generate(
                        input_ids= torch.cat((inputs, torch.tensor([[32, 25]]).to(device)), dim=1),
                        max_length=1024,
                        num_return_sequences=1,
                        do_sample=True,
                        num_beams=2,
                        top_k=50,
                        temperature=1.25,
                        eos_token_id=628,
                        pad_token_id=628,
        #                 no_repeat_ngram_size=3,
                        output_scores=True,
                        return_dict_in_generate=True,
                        early_stopping=True
                    )

    return generated


def get_recommended_id(past_list, model):
    past_wtes = past_wtes_constructor(past_list, model)
    recalled_ids = model.validation_perform_recall(past_wtes, 500)
    rerank_logits = model.validation_perform_rerank(past_wtes, recalled_ids)
    recommended_id = recalled_ids[np.argsort(rerank_logits.cpu().detach().numpy())[-1]]

    rec_token_id = item_id_2_lm_token_id[recommended_id]
    title = model.items_db[recommended_id]
    title = title.split('[SEP]')[0].strip()
    recommended_item_id = torch.tensor([[rec_token_id]]).to(device)

    return title, recommended_item_id


def get_constraints(length, dialogues):
    CTRS = []

    multiplier = 0
    for turn_num in range(length):
        _, gold_recommended_ids, _ = dialogues[turn_num]
        if gold_recommended_ids != None:
            multiplier += 1
        CTRS.append([multiplier*ENTITY_TOKEN])

    return CTRS


@torch.no_grad()
def CDecode(model, tokenizer, device, args, batch, item_id_2_lm_token_id, e2e = False):
    
    eos_ids, bad_words_ids = get_special_tokens_ids(tokenizer)

    if device is torch.device('cuda:0'):
        torch.cuda.empty_cache()
    model.eval()
    model = model.to(device)

    writer = open(args.output_file,'w')
    ############################### ADDITION ######################
    role_ids, dialogues = batch
    dialog_tensors = [torch.LongTensor(utterance).to(model.device) for utterance, _ , _ in dialogues]



    past_tokens = None
    original_sentences = []
    tokenized_sentences = []
    
    
    ################## CONTRAINTS Handling ###################
    
    def expand_factor(items, factors):
        expanded_items = []
        for item, factor in zip(items, factors):
            expanded_items.extend([item] * factor)
        return expanded_items

    CTRS = get_constraints(len(role_ids), dialogues)

    constraints_list = tokenize_constraints(tokenizer, CTRS)
    # init_factor = [args.beam_size for _ in constraints_list]
    # constraints_list = expand_factor(constraints_list,init_factor)
    past_list = []

    for turn_num in range(len(role_ids)):
        dial_turn_inputs = dialog_tensors[turn_num]
        _, gold_recommended_ids, target_goal = dialogues[turn_num]
        
        gold_item_ids = []; gold_item_titles = []
        if gold_recommended_ids != None:
            for r_id in gold_recommended_ids:
                gold_item_ids.append(item_id_2_lm_token_id[r_id])
                title = model.items_db[r_id]
                title = title.split('[SEP]')[0].strip()
                gold_item_titles.append(title)
            gold_item_ids = torch.tensor([gold_item_ids]).to(device)
        

        if role_ids[turn_num] == 0: # User
            if turn_num == 0:
                past_tokens = dial_turn_inputs
            elif turn_num!= 0:
                past_tokens = torch.cat((past_tokens, dial_turn_inputs), dim=1)
            past_list.append((dial_turn_inputs,gold_recommended_ids))
        
        else: # System    
                if turn_num == 0:
                    past_tokens = dial_turn_inputs
                    past_list.append((dial_turn_inputs,gold_recommended_ids))
                    continue
                
                title, recommended_item_id = get_recommended_id(past_list,model)

                if gold_recommended_ids!=None:
                    rec_start_token = model.lm_tokenizer(model.REC_TOKEN_STR, return_tensors="pt")["input_ids"].to(model.device)
                    rec_end_token = model.lm_tokenizer(model.REC_END_TOKEN_STR, return_tensors="pt")["input_ids"].to(model.device)
                    
                    past_tokens = torch.cat((past_tokens, rec_start_token, recommended_item_id, rec_end_token), dim=1)
                else:
                    past_tokens = past_tokens

                total_len = past_tokens.shape[1]
                if total_len >= 1024: break


                original_sen = model.lm_tokenizer.decode(dial_turn_inputs[0], skip_special_tokens=True)
                inputs = torch.cat((past_tokens, torch.tensor([[32, 25]]).to(device)), dim=1) 


                if gold_recommended_ids != None:
                    generated = generate_with_constraints(args,inputs,constraints_list, turn_num,eos_ids, bad_words_ids)

                else:
                    generated = generate_without_constraints(inputs)

                final_gen = None
                
                
                if gold_recommended_ids == None:
                    gen_sen = model.lm_tokenizer.decode(generated.sequences[0][past_tokens.shape[1]:], skip_special_tokens=True)
                    final_gen = gen_sen

                else:
                    gen_sen = model.lm_tokenizer.decode(generated[0][past_tokens.shape[1]:], skip_special_tokens=True)
                    final_gen = gen_sen
                    
                final_gen = replace_placeholder(final_gen, [title]).replace("\n\n\n", "")
                tokenized_sen = final_gen.strip().split(' ')
                tokenized_sentences.append(tokenized_sen)
                original_sen = replace_placeholder(original_sen, gold_item_titles).replace("\n\n\n", "")
                original_sentences.append( original_sen.strip().split(' ') )

                writer.write(final_gen.strip()[2:]+'\t'+original_sen.strip()[2:]+'\n')
                    

                if turn_num != 0:
                    past_tokens = torch.cat((past_tokens, dial_turn_inputs), dim=1)

                if gold_recommended_ids== []:
                    past_list.append((dial_turn_inputs,gold_recommended_ids))
                else:
                    past_list.append((None, gold_recommended_ids))
                    past_list.append((dial_turn_inputs, None))
    writer.close()

    return original_sentences, tokenized_sentences



if __name__ == '__main__':
    args = parse_decoder_args()


    
    CKPT = args.pretrained_model
    device = torch.device(0)
    # device = torch.device('cpu')

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



    items_db_path = args.kb_path
    items_db = torch.load(items_db_path)


    ############## Dataset loading ###########
    test_path = args.test_file
    test_dataset = RecDataset(torch.load(test_path), bert_tokenizer, gpt_tokenizer)
    
    
    # Data loader 
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, collate_fn=test_dataset.collate)

    
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
    
    ########## Loading Weights for the Model to generate ###############
    model.to(device)
    model.load_state_dict(torch.load(CKPT,map_location=device)) 
    model.annoy_base_constructor()

    item_id_2_lm_token_id = model.lm_expand_wtes_with_items_annoy_base()

    total_sentences_original = []; total_sentences_generated = []
    
    for batch in tqdm(test_dataloader):
        
        original_sentences, tokenized_sentences = CDecode(model, gpt_tokenizer, device, args, batch[0], item_id_2_lm_token_id)
        total_sentences_original.extend(original_sentences)
        total_sentences_generated.extend(tokenized_sentences) 

    torch.save(total_sentences_generated, 'human_eval/constrained_preds.pt')
    torch.save(total_sentences_original,'human_eval/constrained_labels.pt')               