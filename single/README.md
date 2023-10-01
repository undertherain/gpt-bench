# Large Language Modeling Benchmark

This benchmark measures throughput of a causal language model training in tokens per second.
The model is implemented following LLaMa architecture (e.g. using rotary embeddings) but parameterized to contain 1.46 billion paramters, similarly to GPT-2 XL (1.5B).

The model definition is based on the HuggingFace transforemrs library code, stripped of all external dependencies.

The model contains 12 self-attention layers, 
each with 32 attention heads, embedding size of 3200 and mixing layer of size 6400.

The model is trained on synthetic sequences, each 256 tokens long.

# Dependencies

The benchmark requires Python interpreter verion 3.8 or above.

The only Python module needed outside of those contained in standard Python distribution is `torch`.

# Running

Run the module as follows:

```
python3 -m bench \
    --batch-size=XX \
    --precision=XXXX \
    --device=XX
```    



# Debugging

Additionally, for debugging purpose, the fillowing options can be specified to make model or input smaller:

- `--sequence-length`
- `--hidden-size`
- `--intermediate-size`
- `--num-hidden-layers`
- `--num-attention-heads`

Changing any of these parameters from their defaults makes benchmark results not valid. 
Note that changing batch size is the only option to increase or decrease the amount of computation done by the benchmark and have alid results.

# Modification

We realize that novel hardware support can require specific modification to the benchmark code. The changes shold be submitted along with benchmark results, and satisfy the following criteria:

## What can not be changed:

 - the model should be run from the Python interpreter.
 - The model definition should stay as it is.
 - The optimizer should be `AdamW` imported from `torch.optim` module.

## What can be changed:

- The model can be post-processed after being instantiated in any way that: 
    - Preserves the original fuctionality.
    - Allows the model to be trained in the same training loop (i.e. `opimizer.step` being called from Python interpreter).
    - Done within the same script.
- Any additional set-up steps to PyTorch framework etc. E.g. `torch.backends.my_backend.enabled = True`
- New numeric format can be added.

All modification should be in principle contained within `set_environment` and `set_model_and_data` methods.
If you need to modify any other part of code (e.g. PyTorch API changed etc) - please contanct benchmark authors.

## Reporting changes:

Submit modified code along with results

For questions contact RIKEN CCS [HPAIS Team](https://www.r-ccs.riken.jp/en/research/labs/hpaisrt/)
