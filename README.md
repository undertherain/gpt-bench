# Large Language Modeling Benchmark

This is a benchmark for a causal language model, 
following LLaMa architecture (e.g. using rotary embeddings) but parameterized to 
contain XXXX paramters, similarly to GPT-2 (1.5B) or GPT-J (6B)

# Running

All hyperparameters are fixed, except for the batch size.
This is done in order to accomodate for devices with smaller available memory.
Specify batch size as `--batch_size=XX`. Users can report best peroformance achieved under any batch size.
