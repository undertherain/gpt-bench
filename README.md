# Large Language Modeling Benchmark

This benchmark measures throughput of a causal language model training in tokens per second.
The model is implemented following LLaMa architecture (e.g. using rotary embeddings) but parameterized to contain 1.46 billion paramters, similarly to GPT-2 XL (1.5B).

The model definition is based on the HuggingFace transforemrs library code, stripped of all external dependencies.

The model contains 12 self-attention layers, 
each with 32 attention heads, embedding size of 3200 and mixing layer of size 6400.

The model is trained on synthetic sequences, each 256 tokens long.

# Preparing data
 
 dump corus to jsonl format by running `python3 dump_wiki.py` in `data` folder

 download vocabulary file with `bash download_vocab.sh`

 