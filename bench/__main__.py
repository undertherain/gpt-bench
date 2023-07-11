from timeit import default_timer as timer

import torch

from .models.llama.configuration_llama import LlamaConfig
from .models.llama.modeling_llama import LlamaForCausalLM


def describe_model(net):
    print(net)
    cnt_params = 0
    for p in net.parameters():
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        cnt_params += nn
    print("cnt params:", cnt_params)


def get_train_data(config):
    batch_size = 2
    data = torch.randint(low=0, high=config.vocab_size, size=(batch_size, 256))
    # TODO: add labels, wrap in named tuple
    return data


def train(net, config):
    data = get_train_data(config)
    # TODO: do preheat
    # TODO: specify cnt repeats so that at least N samples are seen
    cnt_batches = 10
    # TODO: so far this is inference
    time_start = timer()
    for i in range(cnt_batches):
        res = net(data)
    time_end = timer()
    print(res.logits.shape)
    elapsed_time = time_end - time_start
    print("done in", elapsed_time)


def main():
    config = LlamaConfig()
    config.num_hidden_layers = 6
    config.hidden_size = 1024  # 2048
    config.intermediate_size = 2048  # 5504
    config.num_attention_heads = 16
    # TODO: create configs for testing and production runs
    print(config)
    net = LlamaForCausalLM(config)
    describe_model(net)
    # TODO: configure device
    train(net, config)
    # model_size = sum(t.numel() for t in model.parameters())


if __name__ == "__main__":
    main()
