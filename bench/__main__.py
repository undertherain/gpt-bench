from timeit import default_timer as timer

import torch

from .models.llama.configuration_llama import LlamaConfig
from .models.llama.modeling_llama import LlamaForCausalLM

LEN_SEQUENCE = 256
# TODO: make batch size config
batch_size = 2
device = "cuda"


def describe_model(net):
    print(net)
    cnt_params = 0
    for p in net.parameters():
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        cnt_params += nn
    print(f"cnt params: {cnt_params} ({cnt_params/10**9:0.2F}B)")


def get_train_data(config):
    data = torch.randint(low=0, high=config.vocab_size, size=(batch_size, LEN_SEQUENCE))
    # TODO: add labels, wrap in named tuple
    return data.to(device)


def train(net, config):
    data = get_train_data(config)
    # TODO: do preheat
    # TODO: specify cnt repeats so that at least N samples are seen
    cnt_batches = 1
    # TODO: so far this is inference
    param_optimizer = [param for param in net.named_parameters() if param[1].requires_grad]
    params_without_weight_decay = ["bias", "gamma", "beta", "LayerNorm", "layer_norm"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in params_without_weight_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in params_without_weight_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=0.0001,
        eps=1e-06,
        weight_decay=0.01,
        betas=(0.9, 0.999))
    time_start = timer()
    for i in range(cnt_batches):
        net.zero_grad()
        batch = {"input_ids": data, "labels": data}
        res = net(**batch)
        loss = res.loss
        loss.backward()
        optimizer.step()
    time_end = timer()
    print(res.logits.shape)
    # print("loss:", res.loss.item()  )
    elapsed_time = time_end - time_start
    cnt_tokens = LEN_SEQUENCE * cnt_batches * batch_size
    tokens_per_scond = cnt_tokens / elapsed_time
    print("done in", elapsed_time)
    print("tokens per second", tokens_per_scond)


def main():
    config = LlamaConfig()
    config.num_hidden_layers = 6
    config.hidden_size = 1024  # 2048
    config.intermediate_size = 2048  # 5504
    config.num_attention_heads = 16
    # TODO: create configs for testing and production runs
    print(config)
    net = LlamaForCausalLM(config)
    net.to(device)
    describe_model(net)
    # TODO: configure device
    train(net, config)
    # model_size = sum(t.numel() for t in model.parameters())


if __name__ == "__main__":
    main()
