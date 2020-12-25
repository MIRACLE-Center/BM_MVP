#!/bin/bash
if [ "$1" == "train" ]
then
	CUDA_VISIBLE_DEVICES=0,1,2 python tools/train_net_step.py --dataset lesion_train --cfg configs/lesion_baselines/multi_windows_3_slices.yaml --start_step 2400 --bs 3 --nw 16 --use_tfboard\
	--load_ckpt ./model_step43856.pth
elif [ "$1" == "test" ]
then
	CUDA_VISIBLE_DEVICES=3 python tools/test_net.py --dataset lesion_test --cfg configs/lesion_baselines/multi_windows_3_slices.yaml\
		--load_ckpt ./model_step43856.pth #--multi-gpu-testing
else
	echo "choose from [train,test]"
fi
