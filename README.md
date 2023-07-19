# Large Language Modeling Benchmark

This benchmark measures throughput of a causal language model training in tokens per second.
The model is implemented following LLaMa architecture (e.g. using rotary embeddings) but parameterized to contain 1.46 billion paramters, similarly to GPT-2 XL (1.5B).

The model definition is based on the HuggingFace transforemrs library code, stripped of most of external dependencies.

The model contains 12 self-attention layers, 
each with 32 attention heads, embedding size of 3200 and mixing layer of size 6400.

The model is trained on synthetic sequences, each 256 tokens long.

Expected minimal memory footprint for the model is 


todo memory footprint
todo dependencies

# Correctness

In order to be able to execute in relatively short time, the benchmark does not check for convergence to a given fidelity metric etc. 
We expect that all operations are performed correctly in corresponding numeric formats, according to the model definition.

# Running

Run the module as follows:

```
python3 -m bench \
    --batch-size=XX \
    --precision=XXXX \
    --device=XX
```    

Batch size can be set to any integer number in order to accomodate devices with different amounts of memory. The user can report the best throughput achieved under any batch size. 
NOTE THAT THE ONY ACCETABLE WAY TO adjust amount of work is by changing the batch size.

Precisions can be one of "FP32", "TF32", "FP16", "BF16".
The user should report all supported precisions along with the achieved throughput number for each precision. 
If a combination of several numeric formats is used, e.g. FP32 and accumulation is done in FP16 in hardware - the whole run is considrered to be done in a precision with the lowest number of bits.  

Device can be any string that can be interpreted in `torch.to(device)` call.


todo: explicitly state that test runs can't be counted

# Debugging

Additionally, for debugging purpose, the fillowing options can be specified:

- --sequence-length
- --hidden-size
- --intermediate-size
- --num-hidden-layers
- --num-attention-heads

# Modification

We realize that novel hardware can require specific modification to the benchmark code. The changes shold be submitted along with benchmark results, and satisfy the following criteria:

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

ADDING NEW PRECISION(S)
point to which lines of code can be modified

## Reporting changes:

Submit modified code along with results
PR to a corresponding branch,
contact if any isses though githhub issues

For questions contact RIKEN CCS <a href="[https://www.example.com/my great page](https://www.r-ccs.riken.jp/en/research/labs/hpaisrt/)https://www.r-ccs.riken.jp/en/research/labs/hpaisrt/">HPAIS</a> team.
