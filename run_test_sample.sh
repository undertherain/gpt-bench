python3 -m bench \
	--batch-size=4 \
	--precision=fp32 \
	--device=cuda \
	--num-hidden-layers=2 \
	--num-attention-heads=4 \
	--hidden-size=512 \
	--intermediate-size=1024
