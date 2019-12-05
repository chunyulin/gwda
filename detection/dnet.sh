AMP="1.8 1.6 1.4 1.3 1.2 1.1 1.0 0.9 0.8 0.7 0.6 0.5"
AMP="1.8 1.6 1.4 1.2 1.1 1.0 0.9"
export CUDA_VISIBLE_DEVICES=1

### My previous run with ccn3/nc=5 seems produce large FAR.
### First train by increasing nc=10
#AMP="1.8 1.6 1.4 1.2 1.1 1.0 0.9"
#python train_detectnet.py -i white_h_8192_dm2.h5 -a ${AMP} -o ./cnn3 -it 2000 -e 1e-8 -nc 10

AMP="1.4 1.2 1.1 1.0 0.9 0.8 0.7"
python train_detectnet.py -i white_h_8192_dm2.h5 -a ${AMP} -o ./cnn3 -it 2000 -e 1e-9 -nc 10 -ck ./cnn3/model_0.90.ckpt


## --train_from ./cnn3/model_1.60.ckpt
      
