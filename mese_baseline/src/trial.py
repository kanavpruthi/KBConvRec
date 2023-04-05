from customPEgpt2 import CTModel
from transformers import GPT2Tokenizer

gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = CTModel.from_pretrained("gpt2")

sent = "BLUE RED GREEN TV <sep> Hello ! Whats up boys? I [ENTITY_ID] was watching reddit today"
sent_tokens = gpt_tokenizer(sent, return_tensors="pt")
ids = sent_tokens['input_ids'].cpu().numpy().tolist()[0]
split_index = ids.index(60)


sent_vector = model(**sent_tokens, bifurcation_index = split_index,context_length = 5,output_hidden_states=True)
