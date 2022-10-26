from io import TextIOWrapper
import requests
import json
from nltk.tokenize import sent_tokenize
import sys 
from tqdm import tqdm 

headers = {
    'Accept': 'application/json',
}
FILENAME = 'en_train.txt'


def annotate(toRead, toWrite:TextIOWrapper):   
    file = open(toRead,'r')
    lines = file.readlines()
    tot_data = []
    
    for line in tqdm(lines):
        dic = json.loads(line)
        complete_ents = []
        complete_tech_ents = []
        for utterance in dic['conversation']:
            data = {'text': utterance}
            resp = requests.post('https://api.dbpedia-spotlight.org/en/annotate', 
                                    data=data, headers=headers)
            ents = []
            tec_ents = []
            if resp is not None:
                response_object = json.loads(resp.text)
                
                try:
                    for linker in response_object["Resources"]:
                        normal = linker['@surfaceForm']
                        dbpedia = linker['@URI']
                        ents.append(normal)
                        tec_ents.append(dbpedia)
                except:
                    pass 

            complete_ents.append(ents)
            complete_tech_ents.append(tec_ents)
        
        dic['entities'] = complete_ents
        dic['dbpedia_entities'] = complete_tech_ents

        toWrite.write(json.dumps(dic,ensure_ascii=False)+'\n')
            


f2w = open('preprocessed_en_train.txt','w')
annotate(FILENAME,f2w)
