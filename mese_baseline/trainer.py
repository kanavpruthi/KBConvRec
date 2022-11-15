from engine import Engine
import time 
import torch
from metrics import distinct_metrics
from mese import UniversalCRSModel
import numpy as np
from utilities import get_memory_free_MiB

class Trainer(object):
    def __init__(self, model:UniversalCRSModel, engine: Engine, train_dataloader, test_dataloader, optimizer, scheduler, progress_bar) -> None:
        self.model = model
        self.engine = engine
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer 
        self.scheduler = scheduler
        self.progress_bar = progress_bar
        self.original_token_emb_size = model.language_model.get_input_embeddings().weight.shape[0]
        self.update_count = 0

    def train(self, num_epochs, num_gradients_accumulation, batch_size, output_file_path, model_saved_path):
        start = time.time()
        for ep in range(num_epochs):
            #"Training"
            pbar = self.progress_bar(self.train_dataloader)
            self.model.train()
            for batch in pbar:
                # batch size of train_dataloader is 1
                avg_ppl = self.engine.train_one_iteration(batch[0], self.model)
                self.update_count +=1
                if self.update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
                    
                    # update for gradient accumulation
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # speed measure
                    end = time.time()
                    speed = batch_size * num_gradients_accumulation / (end - start)
                    start = end
                    
                    # show progress
                    pbar.set_postfix(ppl=avg_ppl, speed=speed)
                    
            self.model.eval()

            with torch.no_grad():
                self.model.annoy_base_constructor()
                pbar = self.progress_bar(self.test_dataloader)
                ppls, recall_losses, rerank_losses = [],[],[]
                total_val, recall_top100_val, recall_top300_val, recall_top500_val, \
                    rerank_top1_val, rerank_top10_val, rerank_top50_val = 0,0,0,0,0,0,0
                for batch in pbar:
                    ppl_history, recall_loss_history, rerank_loss_history, \
                    total, recall_top100, recall_top300, recall_top500, \
                    rerank_top1, rerank_top10, rerank_top50 = self.engine.validate_one_iteration(batch[0], self.model)
                    ppls += ppl_history; recall_losses += recall_loss_history; rerank_losses += rerank_loss_history
                    total_val += total; 
                    recall_top100_val += recall_top100; recall_top300_val += recall_top300; recall_top500_val += recall_top500
                    rerank_top1_val += rerank_top1; rerank_top10_val += rerank_top10; rerank_top50_val += rerank_top50
                
                item_id_2_lm_token_id = self.model.lm_expand_wtes_with_items_annoy_base()
                pbar = self.progress_bar(self.test_dataloader)
                total_sentences = []
                integration_cnt, total_int_cnt = 0, 0
                for batch in pbar:
                    sentences, ic, tc = self.engine.validate_language_metrics_batch(batch[0], self.model, item_id_2_lm_token_id)
                    for s in sentences:
                        total_sentences.append(s)

                    integration_cnt += ic; total_int_cnt += tc
                integration_ratio = integration_cnt / total_int_cnt
                dist1, dist2, dist3, dist4 = distinct_metrics(total_sentences)
                self.model.lm_restore_wtes(self.original_token_emb_size)
                
            output_file = open(output_file_path, 'w')
            output_file.writelines([f"Epoch {ep} ppl: {np.mean(ppls)}, recall_loss: {np.mean(recall_losses)}, rerank_loss: {np.mean(rerank_losses)}"])
            output_file.write('\n')
            output_file.writelines([f"recall top100: {recall_top100_val/total_val}, top300: {recall_top300_val/total_val}, top500: {recall_top500_val/total_val}"])
            output_file.write('\n')
            output_file.writelines([f"rerank top1: {rerank_top1_val/total_val}, top10: {rerank_top10_val/total_val}, top50: {rerank_top50_val/total_val}"])
            output_file.write('\n')
            output_file.writelines([f"Integration Ratio: {integration_ratio}"])
            output_file.write('\n')
            output_file.writelines([f"Dist1: {dist1}, Dist2: {dist2}, Dist3: {dist3}, Dist4: {dist4}"])
            output_file.write('\n\n')
            output_file.close()
            
            torch.save(self.model.state_dict(), model_saved_path + str(ep) +".pt")

    def generate(self):
        self.model.eval()

        with torch.no_grad():
            self.model.annoy_base_constructor()


            item_id_2_lm_token_id = self.model.lm_expand_wtes_with_items_annoy_base()
            pbar = self.progress_bar(self.test_dataloader)
            total_sentences_original = []; total_sentences_generated = []
            integration_cnt, total_int_cnt = 0, 0
            valid_cnt, total_gen_cnt, response_with_items = 0, 0, 0
            for batch in pbar:
                original_sens, sentences, ic, tc, vc, tgc, rwi, group = self.engine.validate_language_metrics_batch2(batch[0], self.model, item_id_2_lm_token_id)
                total_sentences_original.append(original_sens)
                total_sentences_generated.append(sentences)
                
                integration_cnt += ic; total_int_cnt += tc
                valid_cnt += vc; total_gen_cnt += tgc; response_with_items += rwi
            integration_ratio = integration_cnt / total_int_cnt
            valid_gen_ratio = valid_cnt / total_gen_cnt
            self.model.lm_restore_wtes(self.original_token_emb_size)

            return total_sentences_original, total_sentences_generated, (valid_cnt, response_with_items, total_gen_cnt)
