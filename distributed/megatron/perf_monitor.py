import torch
from megatron import get_num_microbatches  # , mpu


class PerfMonitor():
    def __init__(self):
        self.tokens_per_second = []

    def get_tokens_per_second(self):
        local_perf = sum(self.tokens_per_second) / len(self.tokens_per_second)
        return torch.distributed.all_reduce(torch.tensor(local_perf)) / torch.distributed.get_world_size()

    def update_stats(self, args, elapsed_time, total_iterations):
        # gpus_per_model = torch.distributed.get_world_size(group=mpu.get_model_parallel_group())
        batch_size = args.micro_batch_size * get_num_microbatches() * args.data_parallel_size
        #samples_per_model = batch_size * args.seq_length
        #model_replica_count = torch.distributed.get_world_size() / gpus_per_model
        # approx_parameters_in_billions = None if (model is None) else get_parameters_in_billions(model)
        elapsed_time_per_iter = elapsed_time / total_iterations
        samples_per_second = batch_size / elapsed_time_per_iter
        # seq_len = args.actual_seq_length
        # TODO: what is actual seq length ???
        tokens_per_second = samples_per_second * args.seq_length
        self.tokens_per_second.append(tokens_per_second)
        #flops calculator
        #hidden_size = args.hidden_size
        #num_layers = args.num_layers
        #vocab_size = args.padded_vocab_size
        print("###### UPD samples per second:", samples_per_second)


perf_monitor = PerfMonitor()
