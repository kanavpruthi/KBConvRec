import torch 
from torch.utils.data import Dataset
import numpy as np

class MovieRecDataset(Dataset):
    def __init__(self, data, bert_tok, gpt2_tok):
        self.data = data
        self.bert_tok = bert_tok
        self.gpt2_tok = gpt2_tok
        self.turn_ending = torch.tensor([[628, 198]]) # end of turn, '\n\n\n'
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        dialogue = self.data[index]
        
        dialogue_tokens = []
        
        for utterance, gt_ind, _ in dialogue:
            utt_tokens = self.gpt2_tok(utterance, return_tensors="pt")['input_ids']
            dialogue_tokens.append( ( torch.cat( (utt_tokens, self.turn_ending), dim=1), gt_ind) )
            
        role_ids = None
        previous_role_ids = None
        if role_ids == None:
            role_ids = [ 0 if item[0] == 'B' else 1 for item, _, _ in dialogue]
            previous_role_ids = role_ids
        else:
            role_ids = [ 0 if item[0] == 'B' else 1 for item, _, _ in dialogue]
            if not np.array_equal(role_ids, previous_role_ids):
                raise Exception("Role ids dont match between languages")
            previous_role_ids = role_ids
        
        return role_ids, dialogue_tokens
    
    def collate(self, unpacked_data):
        return unpacked_data


class RecDataset(Dataset):
    def __init__(self, data, bert_tok, gpt2_tok):
        self.data = data
        self.bert_tok = bert_tok
        self.gpt2_tok = gpt2_tok
        self.turn_ending = torch.tensor([[628, 198]]) # end of turn, '\n\n\n'
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        dialogue = self.data[index]
        
        dialogue_tokens = []
        
        for utterance, gt_ind, rec_or_not_target in dialogue:
            utt_tokens = self.gpt2_tok(utterance, return_tensors="pt")['input_ids']
            dialogue_tokens.append( ( torch.cat( (utt_tokens, self.turn_ending), dim=1), gt_ind, int(rec_or_not_target)) )
            
        role_ids = None
        previous_role_ids = None
        if role_ids == None:
            role_ids = [ 0 if item[0] == 'B' else 1 for item, _ , _ in dialogue]
            previous_role_ids = role_ids
        else:
            role_ids = [ 0 if item[0] == 'B' else 1 for item, _ , _ in dialogue]
            if not np.array_equal(role_ids, previous_role_ids):
                raise Exception("Role ids dont match between languages")
            previous_role_ids = role_ids
        
        return role_ids, dialogue_tokens
    
    def collate(self, unpacked_data):
        return unpacked_data