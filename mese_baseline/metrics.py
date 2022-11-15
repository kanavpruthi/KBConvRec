from nltk.translate import bleu_score
from nltk.translate.bleu_score import sentence_bleu
import string 

def distinct_metrics(outs):
    # outputs is a list which contains several sentences, each sentence contains several words
    unigram_count = 0
    bigram_count = 0
    trigram_count=0
    quagram_count=0
    unigram_set = set()
    bigram_set = set()
    trigram_set=set()
    quagram_set=set()
    for sen in outs:
        for word in sen:
            unigram_count += 1
            unigram_set.add(word)
        for start in range(len(sen) - 1):
            bg = str(sen[start]) + ' ' + str(sen[start + 1])
            bigram_count += 1
            bigram_set.add(bg)
        for start in range(len(sen)-2):
            trg=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
            trigram_count+=1
            trigram_set.add(trg)
        for start in range(len(sen)-3):
            quag=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
            quagram_count+=1
            quagram_set.add(quag)
    dis1 = len(unigram_set) / len(outs)#unigram_count
    dis2 = len(bigram_set) / len(outs)#bigram_count
    dis3 = len(trigram_set)/len(outs)#trigram_count
    dis4 = len(quagram_set)/len(outs)#quagram_count
    return dis1, dis2, dis3, dis4

def bleu_calc_one(ref, hyp):
    for i in range(len(ref)):
        ref[i] = ref[i].lower()
    for i in range(len(hyp)):
        hyp[i] = hyp[i].lower()
    bleu1 = sentence_bleu([ref], hyp, weights=(1, 0, 0, 0), smoothing_function=bleu_score.SmoothingFunction(epsilon=1e-12).method7)
    bleu2 = sentence_bleu([ref], hyp, weights=(1/2, 1/2, 0, 0), smoothing_function=bleu_score.SmoothingFunction(epsilon=1e-12).method7)
    bleu3 = sentence_bleu([ref], hyp, weights=(1/3, 1/3, 1/3, 0), smoothing_function=bleu_score.SmoothingFunction(epsilon=1e-12).method7)
    bleu4 = sentence_bleu([ref], hyp, weights=(1/4, 1/4, 1/4, 1/4), smoothing_function=bleu_score.SmoothingFunction(epsilon=1e-12).method7)
    return bleu1, bleu2, bleu3, bleu4

def bleu_calc_all(originals, generated):
    bleu1_total, bleu2_total, bleu3_total, bleu4_total = 0, 0, 0, 0
    total = 0
    for o, g in zip(originals, generated):
        r = [ i.translate(str.maketrans('', '', string.punctuation)) for i in o][1:]
        h = [ i.translate(str.maketrans('', '', string.punctuation)) for i in g][1:]
        if '[MOVIE_ID]' in r: continue
#         if len(g) >= 500: continue
        if len(g) >= 100: continue
        bleu1, bleu2, bleu3, bleu4 = bleu_calc_one(r, h)
        bleu1_total += bleu1; bleu2_total += bleu2; bleu3_total += bleu3; bleu4_total += bleu4;
        total += 1
    return bleu1_total / total, bleu2_total / total, bleu3_total / total, bleu4_total / total