import torch 
from utilities import past_wtes_constructor, replace_placeholder, get_memory_free_MiB
import numpy as np
import sys
from mese import UniversalCRSModel

class Engine(object):
    def __init__(self,
                device, 
                criterion_language = None, 
                criterion_recall = None,
                criterion_rerank_train = None,
                language_loss_train_coeff = None, 
                recall_loss_train_coeff = None,
                rerank_loss_train_coeff = None, 
                num_samples_recall_train = None, 
                num_samples_rerank_train = None,
                rerank_encoder_chunk_size = None, 
                validation_recall_size = None,
                temperature = None) -> None:
        self.criterion_language = criterion_language
        self.criterion_recall = criterion_recall
        self.criterion_rerank_train = criterion_rerank_train
        self.device = device 
        self.temperature = temperature
        self.language_loss_train_coeff  = language_loss_train_coeff 
        self.recall_loss_train_coeff = recall_loss_train_coeff
        self.rerank_loss_train_coeff = rerank_loss_train_coeff
        self.num_samples_recall_train = num_samples_recall_train
        self.num_samples_rerank_train = num_samples_rerank_train
        self.rerank_encoder_chunk_size = rerank_encoder_chunk_size
        self.validation_recall_size = validation_recall_size
        self.cpu = torch.device('cpu')

    def train_one_iteration(self,batch,model: UniversalCRSModel, scaler):
        role_ids, dialogues = batch
        # print('before training free space: ',get_memory_free_MiB(4))
        dialog_tensors = [torch.LongTensor(utterance).to(model.device) for utterance, _ in dialogues]
        # print('after sending dialogue tensors: ',get_memory_free_MiB(4)) 
        # Negligible GPU Usage at this point
        past_list = []
        ppl_history = []
        # for i,d in enumerate(dialog_tensors):
        #     _, recommended_ids = dialogues[i]
        #     print(model.lm_tokenizer.decode(d[0]),recommended_ids)
        # sys.exit(0)
    #     language_logits, language_targets = [], []
        for turn_num in range(len(role_ids)):
            current_tokens = dialog_tensors[turn_num]
            _, recommended_ids = dialogues[turn_num]

            if past_list == []:
                past_list.append((current_tokens, recommended_ids))
                continue
            
            if recommended_ids == None: # no rec
                if role_ids[turn_num] == 0: # user
                    past_list.append((current_tokens, None))
                else: #system
                    past_wtes = past_wtes_constructor(past_list, model)
                    with torch.cuda.amp.autocast():
                        language_logits, language_targets = model.forward_pure_language_turn(past_wtes, current_tokens)
                        # print('after pure language turn (non recommendation): ',get_memory_free_MiB(4))
                        
                        # loss backward
                        language_targets_mask = torch.ones_like(language_targets).float()
                        loss_ppl = self.criterion_language(language_logits, language_targets, language_targets_mask, label_smoothing=0.02, reduce="batch")
                        loss_ppl = self.language_loss_train_coeff * loss_ppl
                    scaler.scale(loss_ppl).backward()

                    perplexity = np.exp(loss_ppl.item())
                    ppl_history.append(perplexity)
                    # print('after backward call language turn (non recommendation): ',get_memory_free_MiB(0))
                    
                    # append to past list
                    past_list.append((current_tokens, None))
            else: # rec!
                
                if role_ids[turn_num] == 0: #user mentioned
                    past_list.append((current_tokens, recommended_ids))
                    continue
                for recommended_id in recommended_ids:
                    #system recommend turn
                    past_wtes = past_wtes_constructor(past_list, model)
                    # print(model.lm_tokenizer.decode(current_tokens))
                    # recall
                    with torch.cuda.amp.autocast():
                        recall_logits, recall_true_index, all_wte_logits, all_wte_targets = model.forward_recall(
                            past_wtes, 
                            current_tokens, 
                            recommended_id, 
                            self.num_samples_recall_train
                        )
                        # print(all_wte_logits.shape)
                        # print(model.lm_tokenizer.decode(all_wte_targets[0]))
                        # print(model.lm_tokenizer.decode(all_wte_logits.argmax(dim=2)[0]))
                        # sys.exit(0)
                        # print('after forward recall turn (recommendation): ',get_memory_free_MiB(4))
                        
                        # recall items loss
                        recall_targets = torch.LongTensor([recall_true_index]).to(model.device)
                        loss_recall = self.criterion_recall(recall_logits.unsqueeze(0), recall_targets)

                        # language loss in recall turn, REC_TOKEN, Language on conditional generation
                        all_wte_targets_mask = torch.ones_like(all_wte_targets).float()
                        loss_ppl = self.criterion_language(all_wte_logits, all_wte_targets, all_wte_targets_mask, label_smoothing=0.02, reduce="batch")
                        perplexity = np.exp(loss_ppl.item())
                        ppl_history.append(perplexity)
                    # Language turn is negligible gpu
                    # print('after forward recall language turn (recommendation): ',get_memory_free_MiB(4))

                    # combined loss
                    recall_total_loss = loss_recall * self.recall_loss_train_coeff + loss_ppl * self.language_loss_train_coeff
                    scaler.scale(recall_total_loss).backward()

                    # rerank
                    past_wtes = past_wtes_constructor(past_list, model)
                    with torch.cuda.amp.autocast():
                        rerank_logits, rerank_true_index = model.forward_rerank(
                            past_wtes, 
                            recommended_id, 
                            self.num_samples_rerank_train, 
                            self.rerank_encoder_chunk_size
                        )
                        
                        # print('after forward rerank turn (recommendation): ',get_memory_free_MiB(4))
                        rerank_logits /= self.temperature

                        # rerank loss 
                        rerank_targets = torch.LongTensor([rerank_true_index]).to(model.device)
                        loss_rerank = self.criterion_rerank_train(rerank_logits.unsqueeze(0), rerank_targets)
                        loss_rerank *= self.rerank_loss_train_coeff
                    scaler.scale(loss_rerank).backward()

                    # Combined loss backward 
                    # #-----------------ADD---------------------
                    # comb_loss = loss_rerank+recall_total_loss
                    # comb_loss.backward()
                    #-----------------------------------------

                    # print('after backward call  (recommendation): ',get_memory_free_MiB(4))

                past_list.append((None, recommended_ids))
                past_list.append((current_tokens, None))
    
        # Free GPU memory
        # dialog_tensors = [torch.LongTensor(utterance).to(self.cpu) for utterance, _ in dialogues]
        del dialog_tensors
        return np.mean(ppl_history)

    def validate_one_iteration(self, batch, model):
        role_ids, dialogues = batch
        dialog_tensors = [torch.LongTensor(utterance).to(model.device) for utterance, _ in dialogues]

        past_list = []
        ppl_history = []
        recall_loss_history = []
        rerank_loss_history = []
        total = 0
        recall_top100, recall_top300, recall_top500 = 0, 0, 0,
        rerank_top1, rerank_top10, rerank_top50 = 0, 0, 0
        
        for turn_num in range(len(role_ids)):
            current_tokens = dialog_tensors[turn_num]
            _, recommended_ids = dialogues[turn_num]
            
            if past_list == []:
                past_list.append((current_tokens, None))
                continue
            
            if recommended_ids == None: # no rec
                if role_ids[turn_num] == 0: # user
                    past_list.append((current_tokens, None))
                else: #system
                    past_wtes = past_wtes_constructor(past_list, model)
                    language_logits, language_targets = model.forward_pure_language_turn(past_wtes, current_tokens)
                    
                    # loss backward
                    language_targets_mask = torch.ones_like(language_targets).float()
                    loss_ppl = self.criterion_language(language_logits, language_targets, language_targets_mask, label_smoothing=-1, reduce="sentence")
                    perplexity = np.exp(loss_ppl.item())
                    ppl_history.append(perplexity)
                    del loss_ppl
                    
                    # append to past list
                    past_list.append((current_tokens, None))
            else: # rec!
                
                if role_ids[turn_num] == 0: #user mentioned
                    past_list.append((current_tokens, recommended_ids))
                    continue
                
                past_wtes = past_wtes_constructor(past_list, model)
                for recommended_id in recommended_ids:
                    total += 1

                    # recall
                    recall_logits, recall_true_index, all_wte_logits, all_wte_targets = model.forward_recall(
                        past_wtes, 
                        current_tokens, 
                        recommended_id, 
                        self.num_samples_recall_train
                    )

                    
                    # recall items loss
                    recall_targets = torch.LongTensor([recall_true_index]).to(model.device)
                    loss_recall = self.criterion_recall(recall_logits.unsqueeze(0), recall_targets)
                    recall_loss_history.append(loss_recall.item())
                    del loss_recall; del recall_logits; del recall_targets

                    # language loss in recall turn, REC_TOKEN, Language on conditional generation
                    all_wte_targets_mask = torch.ones_like(all_wte_targets).float()
                    loss_ppl = self.criterion_language(all_wte_logits, all_wte_targets, all_wte_targets_mask, label_smoothing=-1, reduce="sentence")
                    perplexity = np.exp(loss_ppl.item())
                    ppl_history.append(perplexity)
                    del loss_ppl; del all_wte_logits; del all_wte_targets

                    recalled_ids = model.validation_perform_recall(past_wtes, self.validation_recall_size)

                    if recommended_id in recalled_ids[:500]:
                        recall_top500 += 1
                    if recommended_id in recalled_ids[:400]:
                        recall_top300 += 1
                    if recommended_id in recalled_ids[:300]:
                        recall_top100 += 1

                    if recommended_id not in recalled_ids:
                        continue # no need to compute rerank since recall is unsuccessful

                    # rerank
                    rerank_true_index = recalled_ids.index(recommended_id)
                    rerank_logits = model.validation_perform_rerank(past_wtes, recalled_ids)
        #             print(rerank_logits)
                    reranks = np.argsort(rerank_logits.cpu().detach().numpy())
                    if rerank_true_index in reranks[-50:]:
                        rerank_top50 += 1
                    if rerank_true_index in reranks[-10:]:
                        rerank_top10 += 1
                    if rerank_true_index in reranks[-1:]:
                        rerank_top1 += 1
                        
                    rerank_targets = torch.LongTensor([rerank_true_index]).to(model.device)
        #             loss_rerank = criterion_rerank(rerank_logits.unsqueeze(0), rerank_targets)
                    rerank_loss_val = torch.nn.CrossEntropyLoss()
                    loss_rerank = rerank_loss_val(rerank_logits.unsqueeze(0), rerank_targets)
                    rerank_loss_history.append(loss_rerank.item())
                    del loss_rerank; del rerank_logits; del rerank_targets
                
                past_list.append((None, recommended_ids))
                past_list.append((current_tokens, None))
        return ppl_history, recall_loss_history, rerank_loss_history, \
                total, recall_top100, recall_top300, recall_top500, \
                rerank_top1, rerank_top10, rerank_top50

    def validate_language_metrics_batch(self, batch, model, item_id_2_lm_token_id):
        role_ids, dialogues = batch
        dialog_tensors = [torch.LongTensor(utterance).to(model.device) for utterance, _ in dialogues]
        
    #     past_list = []
        past_tokens = None
        tokenized_sentences = []
        integration_total, integration_cnt = 0, 0
        
        for turn_num in range(len(role_ids)):
            dial_turn_inputs = dialog_tensors[turn_num]
            _, recommended_ids = dialogues[turn_num]
            
            item_ids = []; 
            if recommended_ids != None:
                for r_id in recommended_ids:
                    item_ids.append(item_id_2_lm_token_id[r_id])
                item_ids = torch.tensor([item_ids]).to(self.device)
            
            if turn_num == 0:
                past_tokens = dial_turn_inputs
            if role_ids[turn_num] == 0:
                if turn_num != 0:
                    past_tokens = torch.cat((past_tokens, dial_turn_inputs), dim=1)
            else:
                if turn_num != 0:
                    if item_ids != []:
                        rec_start_token = model.lm_tokenizer(model.REC_TOKEN_STR, return_tensors="pt")["input_ids"].to(model.device)
                        rec_end_token = model.lm_tokenizer(model.REC_END_TOKEN_STR, return_tensors="pt")["input_ids"].to(model.device)
                        past_tokens = torch.cat((past_tokens, rec_start_token, item_ids, rec_end_token), dim=1)
                    else:
                        past_tokens = past_tokens
                    
                total_len = past_tokens.shape[1]
                if total_len >= 1024: break
    #                 print("Original Rec: " + model.lm_tokenizer.decode(dial_turn_inputs[0], skip_special_tokens=True))
                generated = model.language_model.generate(
                    input_ids= torch.cat((past_tokens, torch.tensor([[32, 25]]).to(self.device)), dim=1),
                    max_length=1024,
                    num_return_sequences=1,
                    do_sample=True,
                    num_beams=5,
                    top_k=50,
                    temperature=1.05,
                    eos_token_id=628,
                    pad_token_id=628,
    #                 no_repeat_ngram_size=3,
    #                         length_penalty=3.0

                )
                generated_sen =  model.lm_tokenizer.decode(generated[0][past_tokens.shape[1]:], skip_special_tokens=True)
    #                 print("Generated Rec: " + generated_sen)
                tokenized_sen = generated_sen.strip().split(' ')
                tokenized_sentences.append(tokenized_sen)
                if recommended_ids != None:
                    integration_total += 1                        
                    if "[MOVIE_ID]" in generated_sen:
                        integration_cnt += 1
                
                if turn_num != 0:
                    past_tokens = torch.cat((past_tokens, dial_turn_inputs), dim=1)
                
        return tokenized_sentences, integration_cnt, integration_total

    def validate_language_metrics_batch2(self, batch, model, item_id_2_lm_token_id):
        with torch.no_grad():
            role_ids, dialogues = batch
            dialog_tensors = [torch.LongTensor(utterance).to(model.device) for utterance, _ in dialogues]
            # print('dialog_tensors :',get_memory_free_MiB(0))
        #     past_list = []
            past_tokens = None
            original_sentences = []
            tokenized_sentences = []
            integration_total, integration_cnt = 0, 0
            valid_gen_selected_cnt = 0; total_gen_cnt = 0; response_with_items = 0; original_response_with_items = 0
            
            for turn_num in range(len(role_ids)):
                dial_turn_inputs = dialog_tensors[turn_num]
                _, recommended_ids = dialogues[turn_num]
                
                item_ids = []; item_titles = []
                if recommended_ids != None:
                    for r_id in recommended_ids:
                        item_ids.append(item_id_2_lm_token_id[r_id])
                        title = model.items_db[r_id]
                        title = title.split('[SEP]')[0].strip()
                        item_titles.append(title)
                    item_ids = torch.tensor([item_ids]).to(self.device)
                
        #         if turn_num == 0:
        #             past_tokens = dial_turn_inputs
                if role_ids[turn_num] == 0: # User
                    if turn_num == 0:
                        past_tokens = dial_turn_inputs
                    elif turn_num != 0:
                        past_tokens = torch.cat((past_tokens, dial_turn_inputs), dim=1)
                else: # System
                    if turn_num != 0:
                        if item_ids != []:
                            rec_start_token = model.lm_tokenizer(model.REC_TOKEN_STR, return_tensors="pt")["input_ids"].to(model.device)
                            rec_end_token = model.lm_tokenizer(model.REC_END_TOKEN_STR, return_tensors="pt")["input_ids"].to(model.device)
                            past_tokens = torch.cat((past_tokens, rec_start_token, item_ids, rec_end_token), dim=1)
                        else:
                            past_tokens = past_tokens
                        
                        total_len = past_tokens.shape[1]
                        if total_len >= 1024: break

                        original_sen = model.lm_tokenizer.decode(dial_turn_inputs[0], skip_special_tokens=True)

                        # print('Before generation: ',get_memory_free_MiB(0))
                        generated = model.language_model.generate(
                            input_ids= torch.cat((past_tokens, torch.tensor([[32, 25]]).to(self.device)), dim=1),
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
                        # print(get_memory_free_MiB(0))
                        # check valid generations, equal num [MOVIE_ID] placeholders
                        total_gen_cnt += 1
                        valid_gens = []; valid_gens_scores = []
                        final_gen = None
                        if len(item_ids) == 0: # no rec items
                            for i in range(len(generated.sequences)):
                                gen_sen = model.lm_tokenizer.decode(generated.sequences[i][past_tokens.shape[1]:], skip_special_tokens=True)
                                if gen_sen.count("[MOVIE_ID]") == 0:
                                    valid_gens.append(gen_sen); valid_gens_scores.append(generated.sequences_scores[i].item())
                            if valid_gens == [] and valid_gens_scores == []: # no valid, pick with highest score
                                i = torch.argmax(generated.sequences_scores).item()
                                final_gen = model.lm_tokenizer.decode(generated.sequences[i][past_tokens.shape[1]:], skip_special_tokens=True)
                            else: # yes valid
                                i = np.argmax(valid_gens_scores)
                                final_gen = valid_gens[i]
                                valid_gen_selected_cnt += 1
                        else:
                            original_response_with_items += 1
                            for i in range(len(generated.sequences)):
                                gen_sen = model.lm_tokenizer.decode(generated.sequences[i][past_tokens.shape[1]:], skip_special_tokens=True)
                                if gen_sen.count("[MOVIE_ID]") == original_sen.count("[MOVIE_ID]"):
                                    valid_gens.append(gen_sen); valid_gens_scores.append(generated.sequences_scores[i].item())
                            if valid_gens == [] and valid_gens_scores == []: # no valid, pick with highest score
                                i = torch.argmax(generated.sequences_scores).item()
                                final_gen = model.lm_tokenizer.decode(generated.sequences[i][past_tokens.shape[1]:], skip_special_tokens=True)
                                if "[MOVIE_ID]" in final_gen:
                                    response_with_items += 1
                                final_gen = replace_placeholder(final_gen, item_titles)
                            else:
                                i = np.argmax(valid_gens_scores)
                                final_gen = valid_gens[i]
                                if "[MOVIE_ID]" in final_gen:
                                    response_with_items += 1
                                final_gen = replace_placeholder(final_gen, item_titles)
                                valid_gen_selected_cnt += 1

            #             generated_sen =  model.lm_tokenizer.decode(generated[0][past_tokens.shape[1]:], skip_special_tokens=True)
            #             print("Generated Rec: " + final_gen)
                        tokenized_sen = final_gen.strip().split(' ')
                        tokenized_sentences.append(tokenized_sen)
                        original_sen = replace_placeholder(original_sen, item_titles).replace("\n\n\n", "")
            #             print("Original Rec: " + original_sen)
                        original_sentences.append( original_sen.strip().split(' ') )
                        if recommended_ids != None:
                            integration_total += 1                        
                            if "[MOVIE_ID]" in final_gen:
                                integration_cnt += 1

                        if turn_num != 0:
                            past_tokens = torch.cat((past_tokens, dial_turn_inputs), dim=1)
                        del generated
                    elif turn_num == 0:
                        original_sen = model.lm_tokenizer.decode(dial_turn_inputs[0], skip_special_tokens=True)


                        generated = model.language_model.generate(
                            input_ids= torch.tensor([[32, 25]]).to(self.device),
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
                        # check valid generations, equal num [MOVIE_ID] placeholders
                        total_gen_cnt += 1
                        valid_gens = []; valid_gens_scores = []
                        final_gen = None
                        if len(item_ids) == 0: # no rec items
                            for i in range(len(generated.sequences)):
                                gen_sen = model.lm_tokenizer.decode(generated.sequences[i], skip_special_tokens=True)
                                if gen_sen.count("[MOVIE_ID]") == 0:
                                    valid_gens.append(gen_sen); valid_gens_scores.append(generated.sequences_scores[i].item())
                            if valid_gens == [] and valid_gens_scores == []: # no valid, pick with highest score
                                i = torch.argmax(generated.sequences_scores).item()
                                final_gen = model.lm_tokenizer.decode(generated.sequences[i], skip_special_tokens=True)
                            else: # yes valid
                                i = np.argmax(valid_gens_scores)
                                final_gen = valid_gens[i]
                                valid_gen_selected_cnt += 1
                        else:
                            original_response_with_items += 1
                            for i in range(len(generated.sequences)):
                                gen_sen = model.lm_tokenizer.decode(generated.sequences[i], skip_special_tokens=True)
                                if gen_sen.count("[MOVIE_ID]") == original_sen.count("[MOVIE_ID]"):
                                    valid_gens.append(gen_sen); valid_gens_scores.append(generated.sequences_scores[i].item())
                            if valid_gens == [] and valid_gens_scores == []: # no valid, pick with highest score
                                i = torch.argmax(generated.sequences_scores).item()
                                final_gen = model.lm_tokenizer.decode(generated.sequences[i], skip_special_tokens=True)
                                if "[MOVIE_ID]" in final_gen:
                                    response_with_items += 1
                                final_gen = replace_placeholder(final_gen, item_titles)
                            else:
                                i = np.argmax(valid_gens_scores)
                                final_gen = valid_gens[i]
                                if "[MOVIE_ID]" in final_gen:
                                    response_with_items += 1
                                final_gen = replace_placeholder(final_gen, item_titles)
                                valid_gen_selected_cnt += 1

            #             generated_sen =  model.lm_tokenizer.decode(generated[0][past_tokens.shape[1]:], skip_special_tokens=True)
            #             print("Generated Rec: " + final_gen)
                        tokenized_sen = final_gen.strip().split(' ')
                        tokenized_sentences.append(tokenized_sen)
                        original_sen = replace_placeholder(original_sen, item_titles).replace("\n\n\n", "")
            #             print("Original Rec: " + original_sen)
                        original_sentences.append( original_sen.strip().split(' ') )
                        if recommended_ids != None:
                            integration_total += 1                        
                            if "[MOVIE_ID]" in final_gen:
                                integration_cnt += 1
                        
                        if turn_num == 0:
                            past_tokens = dial_turn_inputs
                        del generated
            return original_sentences, tokenized_sentences, integration_cnt, integration_total, valid_gen_selected_cnt, total_gen_cnt, response_with_items, original_response_with_items