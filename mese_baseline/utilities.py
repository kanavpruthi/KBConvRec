import random 
import torch 
import pynvml

def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 2

def replace_placeholder(sentence, movie_titles):
    sen = sentence
    for title in movie_titles:
        sen = sen.replace("[MOVIE_ID]", title, 1)
    return sen

def past_wtes_constructor(past_list, model):
    past_wtes = []
    for language_tokens, recommended_ids in past_list:
        if language_tokens == None and recommended_ids != None: # rec turn
            # append REC, gt_item_wte, REC_END
            REC_wte = model.get_rec_token_wtes() # [1, 1, 768]
            gt_item_wte, _ = model.compute_encoded_embeddings_for_items(
                model.item_encoder,
                recommended_ids, 
                model.items_db
            ) # [1, 768]
            gt_item_wte = model.rerank_item_wte_mapper(gt_item_wte)
            
            REC_END_wte = model.get_rec_end_token_wtes() # [1, 1, 768]
            combined_wtes = torch.cat(
                (REC_wte,
                 gt_item_wte.unsqueeze(0), # [1, 1, 768]
                 REC_END_wte
                ), 
                dim=1
            ) # [1, 3, 768]
            past_wtes.append(combined_wtes)
        elif recommended_ids == None and language_tokens != None: # language turn simply append wtes
            wtes = model.language_model.transformer.wte(language_tokens) # [1, len, 768]
            past_wtes.append(wtes)
        elif recommended_ids != None and language_tokens != None: # user mentioned turn
            l_wtes = model.language_model.transformer.wte(language_tokens)
            
            SEP_wte = model.get_sep_token_wtes()
            gt_item_wte, _ = model.compute_encoded_embeddings_for_items(
                model.item_encoder,
                recommended_ids, 
                model.items_db
            ) # [1, 768]
            gt_item_wte = model.rerank_item_wte_mapper(gt_item_wte)
            SEP_wte = model.get_sep_token_wtes()
            combined_wtes = torch.cat(
                (l_wtes,
                 SEP_wte,
                 gt_item_wte.unsqueeze(0), # [1, 1, 768]
                 SEP_wte
                ), 
                dim=1
            )
            past_wtes.append(combined_wtes)
            
    
    past_wtes = torch.cat(past_wtes, dim=1)
    # don't trim since we already dealt with length in model functions
    return past_wtes


def sample_ids_from_db(item_db,
                       gt_id, # ground truth id
                       num_samples, # num samples to return
                       include_gt # if we want gt_id to be included
                      ):
    ids_2_sample_from = list(item_db.keys())
    ids_2_sample_from.remove(gt_id)
    if include_gt:
        results = random.sample(ids_2_sample_from, num_samples-1)
        results.append(gt_id)
    else:
        results = random.sample(ids_2_sample_from, num_samples)
    return results