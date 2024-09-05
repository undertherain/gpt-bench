python3 ./tools/preprocess_data.py \
    --input ./data/corpus.jsonl \
    --output-prefix ./data/my-gpt \
    --vocab-file ./data/gpt2-vocab.json \
    --merge-file ./data/gpt2-merges.txt \
    --tokenizer-type GPT2BPETokenizer \
    --workers 1
