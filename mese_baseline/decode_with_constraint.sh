CUDA_VISIBLE_DEVICES=0 python src/decode_e2e.py \
  --kb_path data/kb/durecdial2_db \
  --pretrained_model runs/new_model_rec.pt \
  --test_file data/processed/durecdial2/durecdial2_all_dev_placeholder_updated \
  --output_file decode.out \
  --batch_size 1 --beam_size 5 --max_tgt_length 1024 --min_tgt_length 8 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 500000 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0.175 --look_ahead_width 1 