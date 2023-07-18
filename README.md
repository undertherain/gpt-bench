# Large Language Modeling Benchmark

This benchmark measures throughput of a causal language model training.
The model is implemented following LLaMa architecture (e.g. using rotary embeddings) but parameterized to contain 1.4 buiillion paramters, similarly to GPT-2 (1.5B) or GPT-J (6B)

# Running

Run the module as follows:

```
python3 -m bench \
    --batch-size=XX \
    --precision=XXXX \
    --device=XX
```    

Batch size can be set to any integer number in order to accomodate devices with different amounts of memory. User can report the best throughput acheved under any batch size. 

Precision can be one of "FP32", "TF32", "FP16", "BF16".
User should report all supported precions along with achieved throughput number. 
If a combination of several numeric formats is used, e.g. FP32 and accumulation is done in FP16 in hardware - the whole run is considrered to be done in a precision with the lowest number of bits.  

Device can be any string that can be interpreted in `torch.to(device)` call.
