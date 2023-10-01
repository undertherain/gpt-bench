# Large Language Modeling Benchmark

This benchmark measures throughput of a causal language model training, measured in tokens per second.

The benchmark comes in two variants: single-node and distributed.
These two variants are not mutually replaceable, but rather complimentary.

Single node benchmark is designed to have minimal external dependencies, the model is implemented in pure PyTorch. Additionally this version has stricter conditions on what can be modified - the model can only be optimized in post-processing step etc. Refer to [correspondign README file](/single/README.md) for more details.

## Correctness

In order to be able to execute in relatively short time, the benchmark does not check for convergence to a given fidelity metric etc.
We expect that all operations are performed correctly in corresponding numeric formats, according to the model definition.
Correctness might be additionally checked by code review and test runs outside of benchmark.

## Numeric precision

Precisions can be one of "FP32", "TF32", "FP16", "BF16".
The user should report all supported precisions along with the achieved throughput number for each precision.

If a combination of several numeric formats is used, e.g. values are stotered in FP32 and accumulation is done in FP16 in hardware - the **whole run is considrered to be done in a precision with the lowest number of bits.**

## Other parameters

Batch size can be set to any integer number in order to accomodate devices with different amounts of memory / core count etc. The user can report the best throughput achieved under any batch size.

Device can be any string that can be interpreted in `torch.to(device)` call.

## Distributed benchmark

Distributed version benchmark is formulated as training of 175B parameters GPT-like model. The actual parallelization scheme is not fixed in order to accomodate for different network topologies. The model size is set, however, to be prohibitively large for a typical stand-alone accelerator.

Reference implementation is based on Megatron-LM codebase and supports "tensor" and pipeline parallelizm.

The model should be initialized to correspond to the GPT atchitecture with the following configuration:

## Preparing data

Download vocabulary file with `bash download_vocab.sh`

Dump corpus to jsonl format by running `python3 dump_wiki.py` in `data` folder and tokenize by running `preprocess.sh` script.
