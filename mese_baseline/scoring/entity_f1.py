import pickle 
import sys 

def get_entities_in_utt(utt, entities):
	seen = []
	for entity in entities:
		if entity in utt:
			seen.append(entity)
	return seen


def compute_micro_f1(pr_list, re_list):

	#average of F1s
	#arithmetic mean over harmonic mean
	total_f1 = 0
	total_pr = 0
	total_re = 0
	for pr, re in zip(pr_list, re_list):
		if pr+re > 0:
			f1=(2*pr*re)/(pr+re)
			total_f1 += f1
		total_pr += pr
		total_re += re

	return total_pr/float(len(pr_list)), total_re/float(len(pr_list)), total_f1/float(len(pr_list))
		

def compute_micro_f1(pr_list, re_list):

	#average of F1s
	#arithmetic mean over harmonic mean
	total_f1 = 0
	total_pr = 0
	total_re = 0
	for pr, re in zip(pr_list, re_list):
		if pr+re > 0:
			f1=(2*pr*re)/(pr+re)
			total_f1 += f1
		total_pr += pr
		total_re += re

	return total_pr/float(len(pr_list)), total_re/float(len(pr_list)), total_f1/float(len(pr_list))
		
def compute_macro_f1(pr_list, re_list):
	average_conv_len = 9
	conv_pr_list, conv_re_list, conv_f1_list = [], [], []
	#average of F1s
	#arithmetic mean over harmonic mean
	for idx in range(0,len(pr_list),average_conv_len):
		p,r,f1 =  compute_micro_f1(pr_list[idx:idx+average_conv_len], re_list[idx:idx+average_conv_len])
		conv_pr_list.append(p)
		conv_re_list.append(r)
		conv_f1_list.append(f1)

	return sum(conv_pr_list)/float(len(conv_pr_list)), sum(conv_re_list)/float(len(conv_re_list)), sum(conv_f1_list)/float(len(conv_f1_list))
	


def f1(preds, golds, movie_list):
	
	pr_list = []
	re_list = []


	for i, (pred, gold) in enumerate(zip(preds, golds)):
		
		if "for you" in gold:
			continue
		
		entities_in_gold = set(get_entities_in_utt(gold,movie_list))
		entities_in_pred = set(get_entities_in_utt(pred,movie_list))
		common = float(len(entities_in_gold.intersection(entities_in_pred)))


		if len(entities_in_gold) == 0:
			continue
		else:
			if entities_in_gold != entities_in_pred:
				pass
			# print(entities_in_gold)
			# print(entities_in_pred)
			# print(f'{gold}{pred}')
			# print()
			# print()
			recall = common/len(entities_in_gold)
			re_list.append(recall)

		
		
		if len(entities_in_pred) == 0:
			pr_list.append(0)
			precision = 0
		else:
			precision = common/len(entities_in_pred)
			pr_list.append(precision)

		if precision != recall:
			print('Gold:',entities_in_gold)
			print('Predicted:',entities_in_pred)
		
	micro_pr, micro_re, micro_f1 = compute_micro_f1(pr_list, re_list)

	macro_pr, macro_re, macro_f1 = compute_macro_f1(pr_list, re_list)

	return (macro_pr, macro_re, macro_f1)


file_path = sys.argv[1]

MOVIE_DB = '../auxiliary/movies.pkl'
movies = pickle.load(open(MOVIE_DB,'rb'))

FOOD_DB = '../auxiliary/food.pkl'
food = pickle.load(open(FOOD_DB,'rb'))

POI_DB = '../auxiliary/poi.pkl'
poi = pickle.load(open(POI_DB,'rb'))

MUSIC_DB = '../auxiliary/music.pkl'
music = pickle.load(open(MUSIC_DB,'rb'))

movies.extend(food)
movies.extend(poi)
movies.extend(music)

file = open(file_path,'r')

golds = [];preds = []
for line_no,line in enumerate(file):
	# print(line_no)
	pred, gold = line.split('\t')
	preds.append(pred)
	golds.append(gold)

print(f1(preds,golds,movies))