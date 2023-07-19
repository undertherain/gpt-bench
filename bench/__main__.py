import argparse
from timeit import default_timer as timer

import torch

from .models.llama.configuration_llama import LlamaConfig
from .models.llama.modeling_llama import LlamaForCausalLM


def describe_model(net):
    # print(net)
    cnt_params = sum(t.numel() for t in net.parameters())
    print(f"cnt params: {cnt_params} ({cnt_params/10**9:0.2F}B)")


def get_train_data(config):
    data = torch.randint(low=0, high=config.vocab_size, size=(config.batch_size, config.sequence_length))
    # TODO: add labels, wrap in named tuple
    return data.to(config.device)


def set_opimizer(net):
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
        lr=0.00001,
        eps=1e-06,
        weight_decay=0.01,
        betas=(0.9, 0.999))
    return optimizer


def set_environment():
    # To the extend possible, do all environment setting operations here
    pass


class Trainer:
    def __init__(self, net, config):
        self.net = net
        self.config = config

    def set_precision(self):
        prec = self.config.precision.lower()
        if prec == "fp32":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        elif prec == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif prec == "fp16":
            self.net.half()
        elif prec == "bf16":
            self.net.bfloat16()
        # Add support for new numeric formats here if needed
        else:
            raise RuntimeError(f"can't set precision to {self.config.precision}")

    def post_process_model(self):
        # To the extend possible implement all post-processing of the model within this method.
        pass

    def train(self):
        data = get_train_data(self.config)
        self.set_precision()
        self.post_process_model()
        self.net.to(self.config.device)
        # TODO: specify cnt repeats so that at least N samples are seen
        cnt_batches = 50
        optimizer = set_opimizer(self.net)
        # Preheat
        self.net.zero_grad()
        batch = {"input_ids": data, "labels": data}
        res = self.net(**batch)
        loss = res.loss
        loss.backward()
        optimizer.step()
        time_start = timer()
        for i in range(cnt_batches):
            self.net.zero_grad()
            batch = {"input_ids": data, "labels": data}
            res = self.net(**batch)
            loss = res.loss
            loss.backward()
            optimizer.step()
        time_end = timer()
        print(res.logits.shape)
        # print("loss:", res.loss.item()  )
        elapsed_time = time_end - time_start
        cnt_tokens = self.config.sequence_length * cnt_batches * self.config.batch_size
        tokens_per_scond = cnt_tokens / elapsed_time
        print("done in", elapsed_time)
        print("tokens per second", tokens_per_scond)


def check_defaults(config, defaults):
    for k, v in defaults.items():
        if v is None:
            continue
        if config[k] != v:
            print(f"{k} is not at default value of {v}, DO NOT CONSIDER THIS A PRODUCTION RUN")


def main():
    parser = argparse.ArgumentParser(prog='LLM Benchmark')
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--precision")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=3200)
    parser.add_argument("--intermediate-size", type=int, default=6400)
    parser.add_argument("--num-hidden-layers", type=int, default=12)
    parser.add_argument("--num-attention-heads", type=int, default=32)
    args = parser.parse_args()
    config = LlamaConfig()
    config.batch_size = args.batch_size
    config.sequence_length = args.sequence_length
    config.device = args.device
    config.precision = args.precision
    config.num_hidden_layers = args.num_hidden_layers
    config.hidden_size = args.hidden_size
    config.intermediate_size = args.intermediate_size
    config.num_attention_heads = args.num_attention_heads
    #print(vars(config)["num_attention_heads"])
    check_defaults(vars(config), vars(parser.parse_args([])))
    set_environment()
    net = LlamaForCausalLM(config)
    print("model created")
    describe_model(net)
    trainer = Trainer(net, config)
    trainer.train()


if __name__ == "__main__":
    main()
