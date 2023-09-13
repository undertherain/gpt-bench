#export WORLD_SIZE=$(($CPUS_PER_NODE*$NNODES))
#export MASTER_ADDR=localhost
#export MASTER_PORT=$((10000 + ($PJM_JOBID % 50000)))
#export TIMER="timer.30b_pp48_tp6_dp2_pytorch1.13_10itr"
CHECKPOINT_PATH=../checkpoints/
#INPUT_PREFIX=dataset
#VOCAB_FILE=gpt2-vocab.json
#MERGE_FILE=gpt2-merges.txt
DATA_PATH=/home/blackbird/Projects_heavy/performance/LLMs/llm-bench/data
#TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"

#DATA_PARALLEL_SIZE=2
#PIPELINE_MODEL_PARALLEL_SIZE=48
#TENSOR_MODEL_PARALLEL_SIZE=6
#PIPELINE_PARALLEL_ARGS="--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE"
#MODEL_PARALLEL_ARGS="--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE"
#PARALLEL_ARGS="$MODEL_PARALLEL_ARGS $PIPELINE_PARALLEL_ARGS"

#sed -i -e 's/^torch.set_flush_denormal(True)//' pretrain_gpt.py
#sed -i -e '/__main__/a\    torch.set_flush_denormal(True)' pretrain_gpt.py
#sed -i -e 's/with profile/#with profile/g' -e 's/    pretrain(train_/pretrain(train_/g' -e 's/print(prof/#print(prof/g' pretrain_gpt.py
#sed -i -e '/from torch.nn.parallel.distributed.*torchDDP/a from torch.profiler import profile, record_function, ProfilerActivity' -e '/while iteration < args.train_iters and/i\    with profile(activities=[ProfilerActivity.CPU], schedule=torch.profiler.schedule(wait=1,warmup=1,active=1), record_shapes=True) as prof:' -e 's/while iteration < args.train_iters and/  while iteration < args.train_iters and/' -e '/return iteration/i\        prof.step()\n    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=100))' megatron/training.py

#export OMP_NUM_THREADS=48
#export LD_PRELOAD=/local/fcc/inst/other/lib/libtcmalloc.so
#export OMP_WAIT_POLICY=ACTIVE

python3 DeepSpeedFugaku/pretrain_gpt.py \
    --num-layers 2 \
    --hidden-size 512 \
    --num-attention-heads 8 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --seq-length 256 \
    --max-position-embeddings 256 \
    --train-iters 3 \
    --lr-decay-iters 320000 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --vocab-file /home/blackbird/Projects_heavy/performance/LLMs/llm-bench/data/gpt2-vocab.json \
    --merge-file /home/blackbird/Projects_heavy/performance/LLMs/llm-bench/data/gpt2-merges.txt \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend gloo \
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --checkpoint-activations \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 0 \
    --use-cpu-initialization \
    --num-workers 0 \
    --no-load-rng \
    $PARALLEL_ARGS \
    $TENSORBOARD_ARGS

#--hidden-size 7176 \
#--num-attention-heads 78 \
#PIPELINE_MODEL_PARALLEL_SIZE=48
#    --no-cuda \

rm -rf checkpoints/
