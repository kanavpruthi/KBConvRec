CUDA_VISIBLE_DEVICES=0 python decode_e2e.py \
  --output_file a_star_decode.out \
  --batch_size 1 --beam_size 5 --max_tgt_length 1024 --min_tgt_length 8 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 500000 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0.175 --look_ahead_width 1 