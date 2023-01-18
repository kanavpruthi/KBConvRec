from corrected_engine import C_Engine
from engine import Engine
from corrected_engine import C_Engine
import time 
import torch
from metrics import distinct_metrics
from corrected_mese import C_UniversalCRSModel
from mese import UniversalCRSModel
from typing import Union
import numpy as np
from loguru import logger 
from utilities import get_memory_free_MiB
from sklearn.metrics import confusion_matrix

class Trainer(object):
    def __init__(self, model: Union[ UniversalCRSModel, C_UniversalCRSModel], engine: Union[ Engine, C_Engine], train_dataloader = None, test_dataloader = None, optimizer = None, scheduler = None, scaler = None, progress_bar = None, writer = None) -> None:
        self.model = model
        self.engine = engine
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer 
        self.scheduler = scheduler
        self.scaler = scaler
        self.progress_bar = progress_bar
        self.writer = writer
        self.original_token_emb_size = model.language_model.get_input_embeddings().weight.shape[0]
        self.update_count = 0

    def train(self, num_epochs, num_gradients_accumulation, batch_size, output_file_path, model_saved_path):
        start = time.time()
        step_train = 0
        step_valid = 0
        for ep in range(num_epochs):
            #"Training"
            pbar = self.progress_bar(self.train_dataloader)
            self.model.train()
            for batch in pbar:
                # batch size of train_dataloader is 1
                self.optimizer.zero_grad()
                if isinstance(self.engine,C_Engine):
                    avg_ppl, avg_ce_loss = self.engine.train_one_iteration(batch[0], self.model, self.scaler)
                    self.writer.add_scalar("Loss/train/CE_Loss", avg_ce_loss, step_train)
                    self.writer.add_scalar("Loss/train/PPL_Loss", avg_ppl, step_train)
                else:
                    avg_ppl = self.engine.train_one_iteration(batch[0], self.model, self.scaler)
                    self.writer.add_scalar("Loss/train/PPL_Loss", avg_ppl, step_train)
                step_train += 1
                self.update_count +=1
                if self.update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
                    
                    # update for gradient accumulation
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    
                    # speed measure
                    end = time.time()
                    speed = batch_size * num_gradients_accumulation / (end - start)
                    start = end
                    
                    # show progress
                    pbar.set_postfix(ppl=avg_ppl, speed=speed)
            
            output_file = open(output_file_path, 'a')
            output_file.write('Training Stage:\n')
            if isinstance(self.engine,C_Engine):
                output_file.writelines([f"Epoch {ep}: ppl: {np.mean(avg_ppl)}, CE Loss: {np.mean(avg_ce_loss)}"])
            else:
                output_file.writelines([f"Epoch {ep}: ppl: {np.mean(avg_ppl)}"])

            output_file.write('\n')

            self.model.eval()

            with torch.no_grad():
                self.model.annoy_base_constructor()
                pbar = self.progress_bar(self.test_dataloader)
                ppls, recall_losses, rerank_losses, true_goals, pred_goals = [],[],[],[],[]
                total_val, recall_top100_val, recall_top300_val, recall_top500_val, \
                    rerank_top1_val, rerank_top10_val, rerank_top50_val = 0,0,0,0,0,0,0
                for batch in pbar:
                    if isinstance(self.engine,C_Engine):
                        ppl_history, ce_history, recall_loss_history, rerank_loss_history, \
                        total, recall_top100, recall_top300, recall_top500, \
                        rerank_top1, rerank_top10, rerank_top50, \
                        y_true, y_pred     = self.engine.validate_one_iteration(batch[0], self.model)
                        true_goals.extend(y_true)
                        pred_goals.extend(y_pred)
                        self.writer.add_scalar("Loss/dev/CE_Loss", np.mean(ce_history), step_valid)
                    else:
                        ppl_history, recall_loss_history, rerank_loss_history, \
                        total, recall_top100, recall_top300, recall_top500, \
                        rerank_top1, rerank_top10, rerank_top50 = self.engine.validate_one_iteration(batch[0], self.model)
                    
                    self.writer.add_scalar("Loss/dev/PPL_Loss", np.mean(ppl_history), step_valid)    
                    step_valid += 1
                    ppls += ppl_history; recall_losses += recall_loss_history; rerank_losses += rerank_loss_history
                    total_val += total; 
                    recall_top100_val += recall_top100; recall_top300_val += recall_top300; recall_top500_val += recall_top500
                    rerank_top1_val += rerank_top1; rerank_top10_val += rerank_top10; rerank_top50_val += rerank_top50
                
                # item_id_2_lm_token_id = self.model.lm_expand_wtes_with_items_annoy_base()
                # pbar = self.progress_bar(self.test_dataloader)
                # total_sentences = []
                # integration_cnt, total_int_cnt = 0, 0
                # for batch in pbar:
                #     sentences, ic, tc = self.engine.validate_language_metrics_batch(batch[0], self.model, item_id_2_lm_token_id)
                #     for s in sentences:
                #         total_sentences.append(s)

                #     integration_cnt += ic; total_int_cnt += tc
                # integration_ratio = integration_cnt / total_int_cnt
                # dist1, dist2, dist3, dist4 = distinct_metrics(total_sentences)
                # self.model.lm_restore_wtes(self.original_token_emb_size)
                
            # output_file = open(output_file_path, 'a')
            output_file.write('Validation Stage:\n')
            output_file.writelines([f"Epoch {ep} ppl: {np.mean(ppls)}, recall_loss: {np.mean(recall_losses)}, rerank_loss: {np.mean(rerank_losses)}"])
            output_file.write('\n')
            output_file.writelines([f"recall top100: {recall_top100_val/total_val}, top300: {recall_top300_val/total_val}, top500: {recall_top500_val/total_val}"])
            output_file.write('\n')
            output_file.writelines([f"rerank top1: {rerank_top1_val/total_val}, top10: {rerank_top10_val/total_val}, top50: {rerank_top50_val/total_val}"])
            output_file.write('\n')

            if isinstance(self.engine,C_Engine):
                tn, fp, fn, tp = confusion_matrix(true_goals,pred_goals).ravel()

                output_file.writelines([f"True Positive: {tp}, False Positive: {fp}, False Negative: {fn}, True Negative: {tn}"])
                output_file.write('\n')
                output_file.writelines([f"Total Recommendations = {len(true_goals)}"])
                output_file.write('\n')
                acc = (tp+tn)/(tp+tn+fp+fn)
                p = tp/(tp+fp)
                r = tp/(tp+fn)
                f1 = 2*p*r/(p+r)
                output_file.writelines([f"Accuracy: {acc}, Precision: {p}, Recall: {r}, F1: {f1}"])

                #################### Tensorboard Visualisation ####################
                self.writer.add_scalar("Scores/dev/Accuracy", acc, ep)
                self.writer.add_scalar("Scores/dev/Precision", p, ep)
                self.writer.add_scalar("Scores/dev/Recall", r, ep)
                self.writer.add_scalar("Scores/dev/F1", f1, ep)
                #################### Tensorboard Visualisation #####################

            output_file.write('\n\n')
            output_file.close()
            
            torch.save(self.model.state_dict(), model_saved_path + str(ep) +".pt")


    def train_binary(self, num_epochs, num_gradients_accumulation, batch_size, output_file_path, model_saved_path):
        start = time.time()
        step = 1
        for ep in range(num_epochs):
            #"Training"
            true_goals, pred_goals  = [], []
            loss_history = []
            pbar = self.progress_bar(self.train_dataloader)
            self.model.train()
            for batch in pbar:
                # batch size of train_dataloader is 1
                self.optimizer.zero_grad()
                avg_ce_loss, y_true, y_pred = self.engine.train_one_iteration_binary(batch[0], self.model, self.scaler)
                self.writer.add_scalar("Loss/train", avg_ce_loss, step)
                step += 1
                loss_history.append(avg_ce_loss)
                true_goals.extend(y_true)
                pred_goals.extend(y_pred)
                self.update_count +=1
                if self.update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
                    
                    # update for gradient accumulation
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    
                    # speed measure
                    end = time.time()
                    speed = batch_size * num_gradients_accumulation / (end - start)
                    start = end
                    
                    # show progress
                    pbar.set_postfix(avg_ce_loss=avg_ce_loss, speed=speed)
            
            tn, fp, fn, tp = confusion_matrix(true_goals,pred_goals).ravel()

            output_file = open(output_file_path, 'a')
            output_file.writelines([f"Epoch {ep} Training Cross Entropy Loss: {np.mean(loss_history)}"])
            output_file.write('\n')
            output_file.writelines([f"True Positive: {tp}, False Positive: {fp}, False Negative: {fn}, True Negative: {tn}"])
            output_file.write('\n')
            output_file.writelines([f"Total Recommendations = {len(true_goals)}"])
            output_file.write('\n')
            acc = (tp+tn)/(tp+tn+fp+fn)
            p = tp/(tp+fp)
            r = tp/(tp+fn)
            f1 = 2*p*r/(p+r)
            output_file.writelines([f"Accuracy: {acc}, Precision: {p}, Recall: {r}, F1: {f1}"])
            
            #################### Tensorboard Visualisation ####################
            self.writer.add_scalar("Scores/train/Accuracy", acc, ep)
            self.writer.add_scalar("Scores/train/Precision", p, ep)
            self.writer.add_scalar("Scores/train/Recall", r, ep)
            self.writer.add_scalar("Scores/train/F1", f1, ep)
            #################### Tensorboard Visualisation #####################
            
            output_file.write('\n\n')
            output_file.close()
            
            # torch.save(self.model.state_dict(), model_saved_path + str(ep) +".pt")


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
                if isinstance(self.engine,C_Engine):
                    original_sens, sentences, ic, tc, vc, tgc, rwi, group = self.engine.generation_with_gold_labels_and_gold_recs(batch[0], self.model, item_id_2_lm_token_id)
                else:
                    original_sens, sentences, ic, tc, vc, tgc, rwi, group = self.engine.generate_with_gold_labels(batch[0], self.model, item_id_2_lm_token_id)
                total_sentences_original.append(original_sens)
                total_sentences_generated.append(sentences)
                
                integration_cnt += ic; total_int_cnt += tc
                valid_cnt += vc; total_gen_cnt += tgc; response_with_items += rwi
            integration_ratio = integration_cnt / total_int_cnt
            valid_gen_ratio = valid_cnt / total_gen_cnt
            self.model.lm_restore_wtes(self.original_token_emb_size)

            return total_sentences_original, total_sentences_generated, (valid_cnt, response_with_items, total_gen_cnt)

    def validate(self, output_file_path):
        self.model.eval()
        
        with torch.no_grad():

            self.model.annoy_base_constructor()
            pbar = self.progress_bar(self.test_dataloader)
            mrrs, ppls, recall_losses, rerank_losses, true_goals, pred_goals = [],[],[],[],[], []
            total_val, recall_top100_val, recall_top300_val, recall_top500_val, \
                rerank_top1_val, rerank_top10_val, rerank_top50_val = 0,0,0,0,0,0,0
            for batch in pbar:
                ppl_history, ce_history, recall_loss_history, rerank_loss_history, mrr_history,\
                total, recall_top100, recall_top300, recall_top500, \
                rerank_top1, rerank_top10, rerank_top50, \
                y_true, y_pred                               = self.engine.validate_one_iteration(batch[0], self.model)
                ppls += ppl_history; recall_losses += recall_loss_history; rerank_losses += rerank_loss_history
                total_val += total; 
                recall_top100_val += recall_top100; recall_top300_val += recall_top300; recall_top500_val += recall_top500
                rerank_top1_val += rerank_top1; rerank_top10_val += rerank_top10; rerank_top50_val += rerank_top50
                true_goals.extend(y_true)
                pred_goals.extend(y_pred)
                if len(mrr_history)!=0:
                    mrrs.append(np.mean(mrr_history))
            
            tn, fp, fn, tp = confusion_matrix(true_goals,pred_goals).ravel()

            output_file = open(output_file_path, 'a')
            output_file.writelines([f"ppl: {np.mean(ppls)}, recall_loss: {np.mean(recall_losses)}, rerank_loss: {np.mean(rerank_losses)}"])
            output_file.write('\n')
            output_file.writelines([f"recall top100: {recall_top100_val/total_val}, top300: {recall_top300_val/total_val}, top500: {recall_top500_val/total_val}"])
            output_file.write('\n')
            output_file.writelines([f"rerank top1: {rerank_top1_val/total_val}, top10: {rerank_top10_val/total_val}, top50: {rerank_top50_val/total_val}"])
            output_file.write('\n')
            output_file.writelines([f"Mean Reciprocal Rank: {np.mean(mrrs)}"])
            output_file.write('\n')
            output_file.writelines([f"True Positive: {tp}, False Positive: {fp}, False Negative: {fn}, True Negative: {tn}"])
            output_file.write('\n')
            output_file.writelines([f"Total Recommendations = {len(true_goals)}"])
            output_file.write('\n')
            acc = (tp+tn)/(tp+tn+fp+fn)
            p = tp/(tp+fp)
            r = tp/(tp+fn)
            f1 = 2*p*r/(p+r)
            output_file.writelines([f"Accuracy: {acc}, Precision: {p}, Recall: {r}, F1: {f1}"])
            output_file.close()
            