
#!/bin/bash

#module load python/3.5.3 
#module load cuda/8.0 

#module load python/2.7.3

#export PYTHONPATH=/ifs/home/zz1409/.local/lib/python2.7/site-packages:/local/apps/python/2.7.3/lib/python2.7/site-packages

cd $HOME/AutomatedSurgeons/code

#python3 preprocess_data.py --max_len 3000 --model cnn --d chf


#python3 train.py --model cnn --d chf --kernels 3 --batch_norm --dropout .2 --h 256  --message "CNN BatchNorm + Dropout + More Hidden Units" --max_len 3000
python3 train.py --model cnn_non_neg --d chf --kernels 3 --batch_norm --dropout .2 --h 256  --message "CNN BatchNorm + Dropout + More Hidden Units" --max_len 3000


#python3 trainV2.py --modelName CNN_Text --d kf --nK  3 --batch_norm --p_dropOut .2 --filters 256   --message "CNN Kernel 3" --max_len 3000 --n_iter 6 --batchSize 64 --flg_cuda
#python3 trainV2.py --modelName CNN_Text --d str --nK  3 --batch_norm --p_dropOut .2 --filters 256   --message "CNN Kernel 3" --max_len 3000 --n_iter 6 --batchSize 64 --flg_cuda


#python3 trainV2.py --modelName CNN_Text --d kf --nK  3 --batch_norm --p_dropOut .2 --filters 256   --message "CNN Kernel 3" --max_len 3000 --n_iter 6 --batchSize 64 --flg_cuda
#python3 trainV2.py --modelName CNN_Text --d str --nK  3 --batch_norm --p_dropOut .2 --filters 256   --message "CNN Kernel 3" --max_len 3000 --n_iter 6 --batchSize 64 --flg_cuda

