python3 ../Megatron-DeepSpeed/tools/preprocess_data.py \
    --input corpus.jsonl \
    --output-prefix my-gpt \
    --vocab-file gpt2-vocab.json \
    --merge-file gpt2-merges.txt \
    --tokenizer-type GPT2BPETokenizer \
    --workers 1
