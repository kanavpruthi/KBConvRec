import torch 
from annoy import AnnoyIndex
import math
import numpy as np
from utilities import sample_ids_from_db, get_memory_free_MiB
from loguru import logger
class C_UniversalCRSModel(torch.nn.Module):
    def __init__(self, 
                 language_model, # backbone of Pretrained LM such as GPT2
                 encoder, # backbone of item encoder such as bert
                 recall_encoder,
                 lm_tokenizer, # language model tokenizer
                 encoder_tokenizer, # item encoder tokenizer
                 device, # Cuda device
                 items_db, # {id:info}, information of all items to be recommended
                 annoy_base_recall=None, # annoy index base of encoded recall embeddings of items
                 annoy_base_rerank=None, # annoy index base of encoded rerank embeddings of items, for validation and inference only
                 recall_item_dim=768, # dimension of each item to be stored in annoy base
                 lm_trim_offset=100, # offset to trim language model wte inputs length = (1024-lm_trim_offset)
                 rec_token_str="[REC]", # special token indicating recommendation and used for recall
                 rec_end_token_str="[REC_END]", # special token indicating recommendation ended, conditional generation starts
                 sep_token_str="[SEP]",
                 placeholder_token_str="[MOVIE_ID]"
                ):
        super(C_UniversalCRSModel, self).__init__()
        
        #models and tokenizers
        self.language_model = language_model
        self.item_encoder = encoder
        self.recall_encoder = recall_encoder
        self.lm_tokenizer = lm_tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.device = device
        
        # item db and annoy index base
        self.items_db = items_db
        self.annoy_base_recall = annoy_base_recall
        self.annoy_base_rerank = annoy_base_rerank
        
        # hyperparameters
        self.recall_item_dim = recall_item_dim
        self.lm_trim_offset = lm_trim_offset
        
        #constants
        self.REC_TOKEN_STR = rec_token_str
        self.REC_END_TOKEN_STR = rec_end_token_str
        self.SEP_TOKEN_STR = sep_token_str
        self.PLACEHOLDER_TOKEN_STR = placeholder_token_str
        
        # map language model hidden states to a vector to query annoy-item-base for recall
        self.recall_lm_query_mapper = torch.nn.Linear(self.language_model.config.n_embd, self.recall_item_dim) # default [768,768]
        # map output of self.item_encoder to vectors to be stored in annoy-item-base 
        self.recall_item_wte_mapper = torch.nn.Linear(self.recall_encoder.config.hidden_size, self.recall_item_dim) # default [768,768]
        # map output of self.item_encoder to a wte of self.language_model
        self.rerank_item_wte_mapper = torch.nn.Linear(self.item_encoder.config.hidden_size, self.language_model.config.n_embd) # default [768,768]
        # map language model hidden states of item wte to a one digit logit for softmax computation
        self.rerank_logits_mapper = torch.nn.Linear(self.language_model.config.n_embd, 1) # default [768,1]
    
    def get_sep_token_wtes(self):
        sep_token_input_ids = self.lm_tokenizer(self.SEP_TOKEN_STR, return_tensors="pt")["input_ids"].to(self.device)
        return self.language_model.transformer.wte(sep_token_input_ids) # [1, 1, self.language_model.config.n_embd]

    def get_rec_token_wtes(self):
        rec_token_input_ids = self.lm_tokenizer(self.REC_TOKEN_STR, return_tensors="pt")["input_ids"].to(self.device)
        return self.language_model.transformer.wte(rec_token_input_ids) # [1, 1, self.language_model.config.n_embd]
    
    def get_rec_end_token_wtes(self):
        rec_end_token_input_ids = self.lm_tokenizer(self.REC_END_TOKEN_STR, return_tensors="pt")["input_ids"].to(self.device)
        return self.language_model.transformer.wte(rec_end_token_input_ids) # [1, 1, self.language_model.config.n_embd]
    
    def get_movie_title(self, m_id):
        title = self.items_db[m_id]
        title = title.split('[SEP]')[0].strip()
        return title
    
    # compute BERT encoded item hidden representation
    # output can be passed to self.recall_item_wte_mapper or self.rerank_item_wte_mapper
    def compute_encoded_embeddings_for_items(self, 
                                             encoder_to_use,
                                             item_ids, # an array of ids, single id should be passed as [id]
                                             items_db_to_use # item databse to use
                                            ):
        chunk_ids = item_ids
        chunk_infos = [items_db_to_use[key] for key in chunk_ids ]
        chunk_tokens = self.encoder_tokenizer(chunk_infos, padding=True, truncation=True, return_tensors="pt")
        chunk_input_ids = chunk_tokens['input_ids'].to(self.device)
        chunk_attention_mask = chunk_tokens['attention_mask'].to(self.device)
        chunk_hiddens = encoder_to_use(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask).last_hidden_state

        # average of non-padding tokens
        expanded_mask_size = list(chunk_attention_mask.size())
        expanded_mask_size.append(encoder_to_use.config.hidden_size)
        expanded_mask = chunk_attention_mask.unsqueeze(-1).expand(expanded_mask_size)
        chunk_masked = torch.mul(chunk_hiddens, expanded_mask) # [num_example, len, 768]
        chunk_pooled = torch.sum(chunk_masked, dim=1) / torch.sum(chunk_attention_mask, dim=1).unsqueeze(-1)
        
        # [len(item_ids), encoder_to_use.config.hidden_size], del chunk_hiddens to free up GPU memory
        return chunk_pooled, chunk_hiddens
    
    # annoy_base_constructor, constructs
    def annoy_base_constructor(self, items_db=None, distance_type='angular', chunk_size=50, n_trees=10):
        items_db_to_use = self.items_db if items_db == None else items_db
        all_item_ids = list(items_db_to_use.keys())
        
        total_pooled = []
        # break into chunks/batches for model concurrency
        num_chunks = math.ceil(len(all_item_ids) / chunk_size)
        for i in range(num_chunks):
            chunk_ids = all_item_ids[i*chunk_size: (i+1)*chunk_size]
            chunk_pooled, chunk_hiddens = self.compute_encoded_embeddings_for_items(self.recall_encoder, chunk_ids, items_db_to_use)
            chunk_pooled = chunk_pooled.cpu().detach().numpy()
            del chunk_hiddens
            total_pooled.append(chunk_pooled)
        total_pooled = np.concatenate(total_pooled, axis=0)
        
        pooled_tensor = torch.tensor(total_pooled).to(self.device)
        
        #build recall annoy index
        annoy_base_recall = AnnoyIndex(self.recall_item_wte_mapper.out_features, distance_type)
        pooled_recall = self.recall_item_wte_mapper(pooled_tensor) # [len(items_db_to_use), self.recall_item_wte_mapper.out_features]
        pooled_recall = pooled_recall.cpu().detach().numpy()
        for i, vector in zip(all_item_ids, pooled_recall):
            annoy_base_recall.add_item(i, vector)
        annoy_base_recall.build(n_trees)
        
        total_pooled = []
        # break into chunks/batches for model concurrency
        num_chunks = math.ceil(len(all_item_ids) / chunk_size)
        for i in range(num_chunks):
            chunk_ids = all_item_ids[i*chunk_size: (i+1)*chunk_size]
            chunk_pooled, chunk_hiddens = self.compute_encoded_embeddings_for_items(self.item_encoder, chunk_ids, items_db_to_use)
            chunk_pooled = chunk_pooled.cpu().detach().numpy()
            del chunk_hiddens
            total_pooled.append(chunk_pooled)
        total_pooled = np.concatenate(total_pooled, axis=0)
        
        pooled_tensor = torch.tensor(total_pooled).to(self.device)
        
        #build rerank annoy index, for validation and inference only
        annoy_base_rerank = AnnoyIndex(self.rerank_item_wte_mapper.out_features, distance_type)
        pooled_rerank = self.rerank_item_wte_mapper(pooled_tensor) # [len(items_db_to_use), self.recall_item_wte_mapper.out_features]
        pooled_rerank = pooled_rerank.cpu().detach().numpy()
        for i, vector in zip(all_item_ids, pooled_rerank):
            annoy_base_rerank.add_item(i, vector)
        annoy_base_rerank.build(n_trees)
        
        del pooled_tensor
        
        self.annoy_base_recall = annoy_base_recall
        self.annoy_base_rerank = annoy_base_rerank
    
    def annoy_loader(self, path, annoy_type, distance_type="angular"):
        if annoy_type == "recall":
            annoy_base = AnnoyIndex(self.recall_item_wte_mapper.out_features, distance_type)
            annoy_base.load(path)
            return annoy_base
        elif annoy_type == "rerank":
            annoy_base = AnnoyIndex(self.rerank_item_wte_mapper.out_features, distance_type)
            annoy_base.load(path)
            return annoy_base
        else:
            return None
    
    def lm_expand_wtes_with_items_annoy_base(self):
        all_item_ids = list(self.items_db.keys())
        total_pooled = []
        for i in all_item_ids:
            total_pooled.append(self.annoy_base_rerank.get_item_vector(i))
        total_pooled = np.asarray(total_pooled) # [len(all_item_ids), 768]
        pooled_tensor = torch.tensor(total_pooled, dtype=torch.float).to(self.device)
        
        old_embeddings = self.language_model.get_input_embeddings()
        item_id_2_lm_token_id = {}
        for k in all_item_ids:
            item_id_2_lm_token_id[k] = len(item_id_2_lm_token_id) + old_embeddings.weight.shape[0]
        new_embeddings = torch.cat([old_embeddings.weight, pooled_tensor], 0)
        new_embeddings = torch.nn.Embedding.from_pretrained(new_embeddings)
        self.language_model.set_input_embeddings(new_embeddings)
        self.language_model.to(self.device)
        return item_id_2_lm_token_id
    
    def lm_restore_wtes(self, original_token_emb_size):
        old_embeddings = self.language_model.get_input_embeddings()
        new_embeddings = torch.nn.Embedding(original_token_emb_size, old_embeddings.weight.size()[1])
        new_embeddings.to(self.device, dtype=old_embeddings.weight.dtype)
        new_embeddings.weight.data[:original_token_emb_size, :] = old_embeddings.weight.data[:original_token_emb_size, :]
        self.language_model.set_input_embeddings(new_embeddings)
        self.language_model.to(self.device)
        assert(self.language_model.get_input_embeddings().weight.shape[0] == original_token_emb_size)
    
    def trim_lm_wtes(self, wtes):
        trimmed_wtes = wtes
        if trimmed_wtes.shape[1] > self.language_model.config.n_positions:
            trimmed_wtes = trimmed_wtes[:,-self.language_model.config.n_positions + self.lm_trim_offset:,:]
        return trimmed_wtes # [batch, self.language_model.config.n_positions - self.lm_trim_offset, self.language_model.config.n_embd]
    
    def trim_positional_ids(self, p_ids, num_items_wtes):
        trimmed_ids = p_ids
        if trimmed_ids.shape[1] > self.language_model.config.n_positions:
            past_ids = trimmed_ids[:,:self.language_model.config.n_positions - self.lm_trim_offset - num_items_wtes]
#             past_ids = trimmed_ids[:, self.lm_trim_offset + num_items_wtes:self.language_model.config.n_positions]
            item_ids = trimmed_ids[:,-num_items_wtes:]
            trimmed_ids = torch.cat((past_ids, item_ids), dim=1)
        return trimmed_ids # [batch, self.language_model.config.n_positions - self.lm_trim_offset]
    
    def compute_inductive_attention_mask(self, length_language, length_rerank_items_wtes):
        total_length = length_language + length_rerank_items_wtes
        language_mask_to_add = torch.zeros((length_language, total_length), dtype=torch.float, device=self.device)
        items_mask_to_add = torch.ones((length_rerank_items_wtes, total_length), dtype=torch.float, device=self.device)
        combined_mask_to_add = torch.cat((language_mask_to_add, items_mask_to_add), dim=0)
        return combined_mask_to_add #[total_length, total_length]
    
    def forward_pure_language_turn(self, 
                                   past_wtes, # past word token embeddings, [1, len, 768]
                                   current_tokens # tokens of current turn conversation, [1, len]
                                  ):
        train_logits, train_targets = None, None
        current_wtes = self.language_model.transformer.wte(current_tokens)
        
        if past_wtes == None:
            lm_outputs = self.language_model(inputs_embeds=current_wtes)
            train_logits = lm_outputs.logits[:, :-1, :]
            train_targets = current_tokens[:,1:]
        else:
            all_wtes = torch.cat((past_wtes, current_wtes), dim=1)
            all_wtes = self.trim_lm_wtes(all_wtes)
            lm_outputs = self.language_model(inputs_embeds=all_wtes)
            train_logits = lm_outputs.logits[:, -current_wtes.shape[1]:-1, :] # skip the last one
            train_targets = current_tokens[:,1:]

        # torch.Size([batch, len_cur, lm_vocab]), torch.Size([batch, len_cur]), torch.Size([batch, len_past+len_cur, lm_emb(768)])
        return train_logits, train_targets
        
    def forward_retrieval(self, 
                       past_wtes, # past word token embeddings, [1, len, 768]
                    #    current_tokens, # tokens of current turn conversation, [1, len]
                       gt_item_id, # id, ex. 0
                       num_samples # num examples to sample for training, including groud truth id
                      ):
        # recall step 1. construct LM sequence output
        # LM input composition: [past_wtes, REC_wtes, gt_item_wte, (gt_item_info_wtes), REC_END_wtes, current_wtes ]
        
        REC_wtes = self.get_rec_token_wtes() # [1, 1, self.language_model.config.n_embd]
        gt_item_wte, _ = self.compute_encoded_embeddings_for_items(self.recall_encoder, [gt_item_id], self.items_db)
        gt_item_wte = self.rerank_item_wte_mapper(gt_item_wte) # [1, self.rerank_item_wte_mapper.out_features]
        
        REC_END_wtes = self.get_rec_end_token_wtes() # [1, 1, self.language_model.config.n_embd]
        # current_wtes = self.language_model.transformer.wte(current_tokens) #[1, current_tokens.shape[1], self.language_model.config.n_embd]
        
        
        # current_wtes_len = current_wtes.shape[1]
        
        # sample num_samples item ids to train recall with "recommendation as classification"
        sampled_item_ids = sample_ids_from_db(self.items_db, gt_item_id, num_samples, include_gt=True)
        gt_item_id_index = sampled_item_ids.index(gt_item_id)
         # [num_samples, self.item_encoder.config.hidden_size]
        encoded_items_wte, _ = self.compute_encoded_embeddings_for_items(self.recall_encoder, sampled_item_ids, self.items_db)
        encoded_items_wte = self.recall_item_wte_mapper(encoded_items_wte)
        
        REC_wtes_len = REC_wtes.shape[1] # 1 by default
        encoded_items_wte_len = encoded_items_wte.shape[0] # Num Samples Randomly Retrieved
        REC_END_wtes_len = REC_END_wtes.shape[1] # 1 by default

        
        lm_wte_inputs = torch.cat(
            (past_wtes, # [batch (1), len, self.language_model.config.n_embd]
             REC_wtes,
             encoded_items_wte.unsqueeze(0), # reshape to [1,1,self.rerank_item_wte_mapper.out_features]
             REC_END_wtes,
            #  current_wtes # [batch (1), len, self.language_model.config.n_embd]
            ),
            dim=1
        )
        lm_wte_inputs = self.trim_lm_wtes(lm_wte_inputs) # trim for len > self.language_model.config.n_positions
        
        # recall step 2. get gpt output logits and hidden states
        lm_outputs = self.language_model(inputs_embeds=lm_wte_inputs, output_hidden_states=True)
        
        # recall step 3. pull logits (recall, rec_token and language logits of current turn) and compute
        
        # recall logit(s)
        rec_token_start_index = -REC_END_wtes_len-encoded_items_wte_len-REC_wtes_len
        rec_token_end_index = -REC_END_wtes_len-encoded_items_wte_len
        enc_items_start_index = -REC_END_wtes_len-encoded_items_wte_len
        enc_items_end_index = -REC_END_wtes_len
        # [batch (1), REC_wtes_len, self.language_model.config.n_embd]
        rec_token_hidden = lm_outputs.hidden_states[-1][:, rec_token_start_index:rec_token_end_index, :]
        enc_items_hidden = lm_outputs.hidden_states[-1][:, enc_items_start_index:enc_items_end_index, :]
        # [batch (1), self.recall_lm_query_mapper.out_features]
        rec_query_vector = self.recall_lm_query_mapper(rec_token_hidden).squeeze(1)
        
        rerank_logits = self.rerank_logits_mapper(enc_items_hidden)
       
        # to compute dot product with rec_query_vector
        items_key_vectors = self.recall_item_wte_mapper(encoded_items_wte) # [num_samples, self.recall_item_wte_mapper.out_features]
        # Expanded Query Vector => Dr
        expanded_rec_query_vector = rec_query_vector.expand(items_key_vectors.shape[0], rec_query_vector.shape[1]) # [num_samples, self.recall_item_wte_mapper.out_features]
        select_logits = torch.sum(expanded_rec_query_vector * items_key_vectors, dim=1) # torch.size([num_samples])
        

        # REC_TOKEN prediction and future sentence prediction
        # hidden rep of the token that's right before REC_TOKEN
        token_before_REC_logits = lm_outputs.logits[:, rec_token_start_index-1:rec_token_end_index-1, :]
        REC_targets = self.lm_tokenizer(self.REC_TOKEN_STR, return_tensors="pt")['input_ids'].to(self.device) # [1, 1]
        
        #
        # torch.size([num_samples]), id, [batch, current_wtes_len+REC_wtes_len, lm_vocab], [current_wtes_len+REC_wtes_len, lm_vocab]
        return select_logits, gt_item_id_index, rerank_logits
        
    def forward_response_generation(self,
                       past_wtes, # past word token embeddings, [1, len, 768]
                       current_tokens, # Current Utterance
                       gt_item_id, # tokens of current turn conversation, [1, len]
                      ):
        REC_wtes = self.get_rec_token_wtes() # [1, 1, self.language_model.config.n_embd]
        gt_item_wte, _ = self.compute_encoded_embeddings_for_items(self.recall_encoder, [gt_item_id], self.items_db)
        gt_item_wte = self.rerank_item_wte_mapper(gt_item_wte) # [1, self.rerank_item_wte_mapper.out_features]
        
        REC_END_wtes = self.get_rec_end_token_wtes() # [1, 1, self.language_model.config.n_embd]
        current_wtes = self.language_model.transformer.wte(current_tokens) #[1, current_tokens.shape[1], self.language_model.config.n_embd]
        
        REC_wtes_len = REC_wtes.shape[1] # 1 by default
        gt_item_wte_len = gt_item_wte.shape[0] # 1 by default
        REC_END_wtes_len = REC_END_wtes.shape[1] # 1 by default
        current_wtes_len = current_wtes.shape[1]
        
        lm_wte_inputs = torch.cat(
            (past_wtes, # [batch (1), len, self.language_model.config.n_embd]
             REC_wtes,
             gt_item_wte.unsqueeze(0), # reshape to [1,1,self.rerank_item_wte_mapper.out_features]
             REC_END_wtes,
             current_wtes # [batch (1), len, self.language_model.config.n_embd]
            ),
            dim=1
        )
        lm_wte_inputs = self.trim_lm_wtes(lm_wte_inputs) # trim for len > self.language_model.config.n_positions
        
        # recall step 2. get gpt output logits and hidden states
        lm_outputs = self.language_model(inputs_embeds=lm_wte_inputs, output_hidden_states=True)
        
        # recall step 3. pull logits (recall, rec_token and language logits of current turn) and compute
        
        rec_token_start_index = -current_wtes_len-REC_END_wtes_len-gt_item_wte_len-REC_wtes_len
        rec_token_end_index = -current_wtes_len-REC_END_wtes_len-gt_item_wte_len
        
        # REC_TOKEN prediction and future sentence prediction
        # hidden rep of the token that's right before REC_TOKEN
        token_before_REC_logits = lm_outputs.logits[:, rec_token_start_index-1:rec_token_end_index-1, :]
        REC_targets = self.lm_tokenizer(self.REC_TOKEN_STR, return_tensors="pt")['input_ids'].to(self.device) # [1, 1]
        
        #language logits and targets
        current_language_logits = lm_outputs.logits[:, -current_wtes_len:-1, :]
        current_language_targets = current_tokens[:,1:]
        
        # REC token and language, their logits and targets
        # [batch, current_wtes_len+REC_wtes_len, lm_vocab]
        all_wte_logits = torch.cat((token_before_REC_logits, current_language_logits), dim=1)
        # [current_wtes_len+REC_wtes_len, lm_vocab]
        all_wte_targets = torch.cat((REC_targets, current_language_targets), dim=1)
        
        # torch.size([num_samples]), id, [batch, current_wtes_len+REC_wtes_len, lm_vocab], [current_wtes_len+REC_wtes_len, lm_vocab]
        return all_wte_logits, all_wte_targets
    
    def validation_perform_recall(self, past_wtes, topk):
        REC_wtes = self.get_rec_token_wtes()
        lm_wte_inputs = torch.cat(
            (past_wtes, # [batch (1), len, self.language_model.config.n_embd]
             REC_wtes # [1, 1, self.language_model.config.n_embd]
            ),
            dim=1
        )
        lm_wte_inputs = self.trim_lm_wtes(lm_wte_inputs) # trim for len > self.language_model.config.n_positions
        lm_outputs = self.language_model(inputs_embeds=lm_wte_inputs, output_hidden_states=True)
        
        rec_token_hidden = lm_outputs.hidden_states[-1][:, -1, :]
        # [batch (1), self.recall_lm_query_mapper.out_features]
        rec_query_vector = self.recall_lm_query_mapper(rec_token_hidden).squeeze(0) # [768]
        rec_query_vector = rec_query_vector.cpu().detach().numpy()
        recall_results = self.annoy_base_recall.get_nns_by_vector(rec_query_vector, topk)
        return recall_results
    
    def validation_perform_rerank(self, past_wtes, recalled_ids):
        REC_wtes = self.get_rec_token_wtes()
        
        total_wtes = [ self.annoy_base_rerank.get_item_vector(r_id) for r_id in recalled_ids]
        total_wtes = [ torch.tensor(wte).reshape(-1, self.language_model.config.n_embd).to(self.device) for wte in total_wtes]
        total_wtes = torch.cat(total_wtes, dim=0) # [len(recalled_ids), 768]
        
        past_wtes_len = past_wtes.shape[1]
        REC_wtes_len = REC_wtes.shape[1] # 1 by default
        total_wtes_len = total_wtes.shape[0]
        
        # compute positional ids, all rerank item wte should have the same positional encoding id 0
        position_ids = torch.arange(0, past_wtes_len + REC_wtes_len, dtype=torch.long, device=self.device)
        items_position_ids = torch.zeros(total_wtes.shape[0], dtype=torch.long, device=self.device)
        combined_position_ids = torch.cat((position_ids, items_position_ids), dim=0)
        combined_position_ids = combined_position_ids.unsqueeze(0) # [1, past_wtes_len+REC_wtes_len+total_wtes_len]
        
        # compute concatenated lm wtes
        lm_wte_inputs = torch.cat(
            (past_wtes, # [batch (1), len, self.language_model.config.n_embd]
             REC_wtes, # [batch (1), 1, self.language_model.config.n_embd]
             total_wtes.unsqueeze(0), # [1, num_samples, self.language_model.config.n_embd]
            ),
            dim=1
        ) # [1, past_len + REC_wtes_len + num_samples, self.language_model.config.n_embd]

        # trim sequence to smaller length (len < self.language_model.config.n_positions-self.lm_trim_offset)
        combined_position_ids_trimmed = self.trim_positional_ids(combined_position_ids, total_wtes_len) # [1, len]
        lm_wte_inputs_trimmed = self.trim_lm_wtes(lm_wte_inputs) # [1, len, self.language_model.config.n_embd]
        assert(combined_position_ids.shape[1] == lm_wte_inputs.shape[1])
        
        inductive_attention_mask = self.compute_inductive_attention_mask(
            lm_wte_inputs_trimmed.shape[1]-total_wtes.shape[0], 
            total_wtes.shape[0]
        )
        rerank_lm_outputs = self.language_model(inputs_embeds=lm_wte_inputs_trimmed,
                  inductive_attention_mask=inductive_attention_mask,
                  position_ids=combined_position_ids_trimmed,
                  output_hidden_states=True)
        
        rerank_lm_hidden = rerank_lm_outputs.hidden_states[-1][:, -total_wtes.shape[0]:, :]
        rerank_logits = self.rerank_logits_mapper(rerank_lm_hidden).squeeze()
        
        return rerank_logits
    
    def validation_generate_sentence(self, past_wtes, recommended_id):
        if recommended_id == None: #  pure language
            lm_wte_inputs = self.trim_lm_wtes(past_wtes)
            lm_outputs = self.language_model(inputs_embeds=lm_wte_inputs, output_hidden_states=True)
            generated = self.language_model.generate(
                past_key_values=lm_outputs.past_key_values, 
                max_length=50, 
                num_return_sequences=1, 
                do_sample=True, 
                eos_token_id=198,
                pad_token_id=198
            )
            generated_sen = self.lm_tokenizer.decode(generated[0], skip_special_tokens=True)
            return generated_sen, None
        else:
            REC_wtes = self.get_rec_token_wtes() # [1, 1, self.language_model.config.n_embd]
            gt_item_wte, _ = self.compute_encoded_embeddings_for_items([recommended_id], self.items_db)
            gt_item_wte = self.rerank_item_wte_mapper(gt_item_wte) # [1, self.rerank_item_wte_mapper.out_features]
            REC_END_wtes = self.get_rec_end_token_wtes() # [1, 1, self.language_model.config.n_embd]

            REC_wtes_len = REC_wtes.shape[1] # 1 by default
            gt_item_wte_len = gt_item_wte.shape[0] # 1 by default
            REC_END_wtes_len = REC_END_wtes.shape[1] # 1 by default

            lm_wte_inputs = torch.cat(
                (past_wtes, # [batch (1), len, self.language_model.config.n_embd]
                 REC_wtes,
                 gt_item_wte.unsqueeze(0), # reshape to [1,1,self.rerank_item_wte_mapper.out_features]
                 REC_END_wtes
                ),
                dim=1
            )
            lm_wte_inputs = self.trim_lm_wtes(lm_wte_inputs) # trim for len > self.language_model.config.n_positions
            lm_outputs = self.language_model(inputs_embeds=lm_wte_inputs, output_hidden_states=True)
            generated = self.language_model.generate(
                past_key_values=lm_outputs.past_key_values, 
                max_length=50, 
                num_return_sequences=1, 
                do_sample=True, 
                eos_token_id=198,
                pad_token_id=198
            )
            generated_sen = self.lm_tokenizer.decode(generated[0], skip_special_tokens=True)
            return generated_sen, self.get_movie_title(recommended_id)