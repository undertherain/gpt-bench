# Large Language Modeling Benchmark

This benchmark measures throughput of a causal language model training in tokens per second.
The model is implemented following LLaMa architecture (e.g. using rotary embeddings) but parameterized to contain 1.4 buiillion paramters, similarly to GPT-2 XL (1.5B).
The model definition is based on the HuggingFace transforemrs library code, stripped of most of external dependencies.

# Correctness

In order to be able to execute in relatively short time, the benchmark does not check for convergence to a given findelity metric etc. 
We expect that all operations are performed correctly in corresponding numeric formats, according to model definition.

# Running

Run the module as follows:

```
python3 -m bench \
    --batch-size=XX \
    --precision=XXXX \
    --device=XX
```    

Batch size can be set to any integer number in order to accomodate devices with different amounts of memory. User can report the best throughput achieved under any batch size. 

Precision can be one of "FP32", "TF32", "FP16", "BF16".
User should report all supported precions along with achieved throughput number. 
If a combination of several numeric formats is used, e.g. FP32 and accumulation is done in FP16 in hardware - the whole run is considrered to be done in a precision with the lowest number of bits.  

Device can be any string that can be interpreted in `torch.to(device)` call.

# Debugging

Additionally, for debugging purpose, the fillowing options can be specified:

- --sequence-length
- --hidden-size
- --intermediate-size
- --num-hidden-layers
- --num-attention-heads

# Modification

We realize that novel harware can require specific modification to benchmark code. The changes shold be submitted along with bechmark results, and satisfy the following criteria:

## What can not be changed:

 - the model should be run from Python interpreter.
 - model definition should stay as it is.
 - optimizer should be `AdamW` imported from `torch.optim` module.

## What can be changed:

model can be post-processed after being instanciated in any way that 
- preserves original fuctionality
- allows model to be trained in the same training loop (i.e. opimizer.step being called from Python interpreter)
- done withing the same script

any additional set-up steps to PyTorch framework etc. E.g. `torch.backends.my_backend.enabled = True`

