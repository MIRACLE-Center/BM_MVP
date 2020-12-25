#!/bin/bash
if [ "$1" == "train" ]
then
	CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py --dataset lesion_train --cfg configs/lesion_baselines/multi_windows_9_slices.yaml --bs 1 --nw 4 --use_tfboard --start_step 2500 --load_ckpt /home1/hli/MVP-Net/Outputs/multi_windows_9_slices/Jul15-00-46-06_ubuntu_step/ckpt/model_step5799.pth
elif [ "$1" == "test" ]
then
	CUDA_VISIBLE_DEVICES=3 python tools/test_net.py --dataset lesion_test --cfg configs/lesion_baselines/multi_windows_9_slices.yaml \
		--load_ckpt /home1/hli/MVP-Net/Outputs/multi_windows_9_slices/Jul15-02-05-47_ubuntu_step/ckpt/model_step6099.pth # 1215 new with pos, 9*3 slices --multi-gpu-testing
else
	echo "choose from [train,test]"
fi
