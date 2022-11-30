import json 

FILE_PATH = ['annotated_en_train.txt','annotated_en_dev.txt','annotated_en_test.txt']

singers = set()

for path in FILE_PATH:
    file = open(path,'r')
    for data_sample in file:
        data_sample = json.loads(data_sample)
        kg = data_sample['knowledge']
        for fact in kg:
            if len(fact) == 0:
                continue
            if fact[1]=='Sings':
                singers.add(fact[0])

print(singers)