def f1(preds, golds, stories, entities, dids, answers, ordered_oovs, word_map, encoder_word_map, db_engine):
	
	punc = ['.', ',', '!', '\'', '\"', '-', '?']

	'''
	punc = ['.', ',', '!', '\'', '\"', '-', '?']
	for entity_idxs, gold in zip(entities, golds):
		processed_gold = [x for x in gold if x != EOS_INDEX and x != PAD_INDEX]
		print("gold", gold)
		print("processed_gold", processed_gold)
		print("entity_idxs", entity_idxs)
		for e in entity_idxs:
			if e < len(processed_gold):
				e_surface_form = word_map[processed_gold[e]]
				if e_surface_form not in punc:
					print("\t", e_surface_form)
		print("----------------------")
	'''

	pr_list = []
	re_list = []

	pr_infor_list = []
	re_infor_list = []

	pr_req_list = []
	re_req_list = []


	for i, (pred, gold, entity, did, answer, story) in enumerate(zip(preds, golds, entities, dids, answers, stories)):

		entities_in_pred = set([])
		entities_in_gold = set([])

		informable_entities_in_pred = set([])
		informable_entities_in_gold = set([])

		'''
		print("did", did)
		print("gold", gold, get_tokenized_response_from_padded_vector(gold, word_map, ordered_oovs[did]))
		print("pred", pred, get_tokenized_response_from_padded_vector(pred, word_map, ordered_oovs[did]))
		print("entity", entity)
		for e in entity:
			e_surface =  word_map[gold[e]]
			if e_surface not in punc:
				entities_in_gold.add(e_surface)
				if db_engine.is_informable_field_value(e_surface):
					informable_entities_in_gold.add(e_surface)
		'''
		
		story_entities = set([])
		for story_line in story:
			story_line_str = " ".join(story_line)
			story_line_str = story_line_str.strip()
			if story_line_str.strip() != "" and "$db" not in story_line_str:
				line_entities, _ = db_engine.get_entities_in_utt(story_line_str)
				for e in line_entities:
					story_entities.add(e)

		entities_in_gold, _ = db_engine.get_entities_in_utt(answer)
		story_entities_in_gold = entities_in_gold.intersection(story_entities)

		if len(entities_in_gold) == 0:
			continue

		entities_in_pred, _ = db_engine.get_entities_in_utt(get_tokenized_response_from_padded_vector(pred, word_map, ordered_oovs[did]))
		story_entities_in_pred = entities_in_pred.intersection(story_entities)

		common = float(len(entities_in_gold.intersection(entities_in_pred)))
		re_list.append(common/len(entities_in_gold))
		if len(entities_in_pred) == 0:
			pr_list.append(0)
		else:
			pr_list.append(common/len(entities_in_pred))
		
		if len(story_entities_in_gold) > 0:
			common = float(len(story_entities_in_gold.intersection(story_entities_in_pred)))
			if len(story_entities_in_pred) == 0:
				pr_infor_list.append(0)
			else:
				pr_infor_list.append(common/len(story_entities_in_pred))
			re_infor_list.append(common/len(story_entities_in_gold))
		#else:
		#	pr_infor_list.append(0)
		#	re_infor_list.append(0)

		req_in_gold = entities_in_gold.difference(story_entities_in_gold)
		if len(req_in_gold) > 0:
			req_in_pred = entities_in_pred.difference(story_entities_in_pred)
			common = float(len(req_in_gold.intersection(req_in_pred)))
			if len(req_in_pred) == 0:
				pr_req_list.append(0)
			else:
				pr_req_list.append(common/len(req_in_pred))
			re_req_list.append(common/len(req_in_gold))
		#else:
		#	pr_req_list.append(0)
		#	re_req_list.append(0)

	macro_pr, macro_re, macro_f1 = compute_macro_f1(pr_list, re_list)

	macro_infor_pr, macro_infor_re, macro_infor_f1 = compute_macro_f1(pr_infor_list, re_infor_list)
	
	macro_req_pr, macro_req_re, macro_req_f1 = compute_macro_f1(pr_req_list, re_req_list)

	return ((macro_pr, macro_re, macro_f1), (macro_infor_pr, macro_infor_re, macro_infor_f1), (macro_req_pr, macro_req_re, macro_req_f1))
