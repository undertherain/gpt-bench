import torch
from megatron import get_num_microbatches, mpu


class PerfMonitor():
    def __init__(self):
        self.tokens_per_second = []

    def update_stats(args, iteration_time, total_iterations):
        gpus_per_model = torch.distributed.get_world_size(group=mpu.get_model_parallel_group())
        batch_size = args.micro_batch_size * get_num_microbatches() * args.data_parallel_size
        #samples_per_model = batch_size * args.seq_length
        #model_replica_count = torch.distributed.get_world_size() / gpus_per_model
        # approx_parameters_in_billions = None if (model is None) else get_parameters_in_billions(model)
        elapsed_time_per_iter = iteration_time / total_iterations
        samples_per_second = batch_size / elapsed_time_per_iter

        #flops calculator
        #hidden_size = args.hidden_size
        #num_layers = args.num_layers
        #vocab_size = args.padded_vocab_size

        # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
        # https://arxiv.org/pdf/2104.04473.pdf).
        # The factor of 4 is when used with activation check-pointing,
        # otherwise it will be 3.
        #checkpoint_activations_factor = 3
        #if hasattr(args, 'checkpoint_activations') and args.checkpoint_activations:
            #checkpoint_activations_factor = 4
        #if hasattr(args, 'recompute_granularity') and (args.recompute_granularity == 'selective' or args.recompute_granularity == 'full'):
            #checkpoint_activations_factor = 4
        #seq_len = args.seq_length
        #if hasattr(args, 'actual_seq_length'):
            #seq_len = args.actual_seq_length
        #flops_per_iteration = (24 * checkpoint_activations_factor * batch_size * seq_len * num_layers * (hidden_size**2)) * (1. + (seq_len / (6. * hidden_size)) + (vocab_size / (16. * num_layers * hidden_size)))
        #tflops = flops_per_iteration / (elapsed_time_per_iter * args.world_size * (10**12))
        #return samples_per_second, tflops, approx_parameters_in_billions
        print("samples per second:", samples_per_second)


perf_monitor = PerfMonitor()
