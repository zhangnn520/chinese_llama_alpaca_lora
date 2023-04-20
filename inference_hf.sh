python inference_hf.py \
  --lora_model /root/autodl-fs/Chinese-LLaMA-Alpaca-main/output \
  --base_model /root/autodl-fs/Chinese-LLaMA-Alpaca-main/alpaca_combined_hf \
  --data_file /root/autodl-fs/Chinese-LLaMA-Alpaca-main/data/enoch_fine_tune_dev.json \
  --tokenizer_path /root/autodl-fs/Chinese-LLaMA-Alpaca-main/scripts/merged_tokenizer_hf \