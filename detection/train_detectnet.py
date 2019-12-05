from __future__ import print_function

import time
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

### default values ==================================================================
H5_FILE    = "./white_h_8192_dm2.h5"
MODEL_PATH = "./model"
CKPT       = ""
BATCH      = 128
MAX_EPOCHS = 2000
LR_INIT   = 1e-4
LR_DECAYR = 0.95
LR_DECAYS = 1000
DROPOUT   = 0
NOISE_COPY = 10

TRAIN_AMP = [1.8, 1.6]
SEED = 123

MONITOR = 20
PATIENCE  = 4
TOLERANCE = 1.e-8
#------------------------------------------------------------------------------
import argparse
def get_arguments():
    parser = argparse.ArgumentParser(description='DetectNet')
    parser.add_argument('-i', '--input', type=str, default=H5_FILE, help='GW template in HDF5 file', required=False)
    parser.add_argument('-o', '--savepath', type=str, default=MODEL_PATH, help='Model save path')
    parser.add_argument('-a', '--train_amp', type=float, nargs='+',  default=TRAIN_AMP, help='List of pretrain amplitude')
    parser.add_argument('-ck', '--train_from', type=str, default=CKPT, help='Model save path')
    parser.add_argument('-bs', '--batch_size', type=int, default=BATCH, help='Batch size during training per GPU')
    parser.add_argument('-it', '--max_epochs', type=int, default=MAX_EPOCHS, help='Max epoches for each training')
    parser.add_argument('-lr', '--lr',    type=float, default=LR_INIT, help='Init learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=LR_DECAYR, help='Learning rate decay rate')
    parser.add_argument('--lr_decay_step', type=int, default=LR_DECAYS, help='Learning rate decay step')
    parser.add_argument('-p', '--dropout_p', type=float, default=DROPOUT, help='Dropout (0 to turn off), not used yet')
    parser.add_argument('-s', '--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument('-e', '--tolerance', type=float, default=TOLERANCE, help='Loss tolerance')
    parser.add_argument('-nc','--noise_copy', type=int, default=NOISE_COPY, help='# noise copy for training')
    return parser.parse_args()

args = get_arguments()
# fix random seed for reproducibility
if SEED > 0:
    rng = np.random.RandomState(args.seed)
    tf.set_random_seed(args.seed)

H5_FILE = args.input
MODEL_PATH = args.savepath
CKPT =  args.train_from
BATCH   = args.batch_size
MAX_EPOCHS  = args.max_epochs
LR_INIT = args.lr
LR_DECAYR = args.lr_decay_rate
LR_DECAYS = args.lr_decay_step
DROPOUT   = args.dropout_p
SEED = args.seed
TRAIN_AMP = args.train_amp
TOLERANCE= args.tolerance
NOISE_COPY= args.noise_copy

print ("###### [ Input params ]")
print ("Input      : ", H5_FILE)
print ("Model path : ", MODEL_PATH)
print ("Train list : ", TRAIN_AMP)
print ("Batch      : ", BATCH)
print ("Max epoch  : ", MAX_EPOCHS)
print ("Tolerance  : ", TOLERANCE)
print ("LR/decay/step  : ", LR_INIT, LR_DECAYR, LR_DECAYS )
print ("Pretrain model : ", CKPT)
print ("# noise copy   : ", NOISE_COPY)

### Load data =======================================================================
import GWDA.loader
GWdata = GWDA.loader.GWInject(H5_FILE)
RATE   = GWdata.srate
print("Sampling rate : ", RATE)


### Assemble model ========================================================
import GWDA.model

tf.reset_default_graph()
DIM = RATE

keep_prob = tf.placeholder(tf.float32)   ##  for dropout, not used.

x = tf.placeholder(tf.float32, [None,DIM])
y = tf.placeholder(tf.float32, [None,1])

logits = GWDA.model.CNN_3(x, keep_prob, DIM)
#logits = GWDA.model.CNN_4(x, keep_prob, DIM)

predict_prob = tf.sigmoid(logits, name="sigmoid_tensor")
predict_op   = tf.cast( tf.round(predict_prob), tf.int32 )
loss_op      = tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=y)
## cf. tf.nn.softmax_cross_entropy_with_logits_v2 

counter  = tf.Variable(0, trainable=False)
add_counter   = counter.assign_add(1)
with tf.name_scope('optimizer'):
    learning_rate = tf.train.exponential_decay(LR_INIT, global_step=counter,
                    decay_steps=LR_DECAYS, decay_rate=LR_DECAYR, staircase=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_op) ##, global_step = counter)
    
_, accuracy    = tf.metrics.accuracy(labels=y, predictions=predict_op  )
_, sensitivity = tf.metrics.recall(labels=y, predictions=predict_op)
#_, sensitivity = tf.metrics.sensitivity_at_specificity(labels=y, predictions=predict_op, specificity=0.005 )

_, fp = tf.metrics.false_positives(labels=y, predictions=predict_op  )
_, fn = tf.metrics.false_negatives(labels=y, predictions=predict_op  )
_, tp = tf.metrics.true_positives(labels=y, predictions=predict_op  )
_, tn = tf.metrics.true_negatives(labels=y, predictions=predict_op  )

#tf.summary.histogram('loss', loss_op)
#tf.summary.scalar('loss', loss_op)
#tf.summary.scalar('accuracy', accuracy)

import GWDA.utils
GWDA.utils.report_vars()


### Training with fixed template ===================================================================================
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())   ### init w, bias to gaussian noise
    merged = tf.summary.merge_all()   ## call merged to do every summary
    saver = tf.train.Saver(max_to_keep=50)
    
    if CKPT != '':
        saver.restore(sess, CKPT )
    
    fig_loss = plt.figure()
    
    for amp in TRAIN_AMP:
        ## summarize to a new folder
        train_writer = tf.summary.FileWriter("%s/train_%4.2f" % (MODEL_PATH, amp ) )
        #train_writer.add_graph(tf.get_default_graph())
        #print('Saving graph to: %s' % MODEL_PATH)
    
        sess.run(tf.local_variables_initializer())  ###  only local vars like TP, TN, FP, FN to be init
        
        time0 = time.time()
        X0, Y0 = GWdata.get_train_set(A=amp, nc=NOISE_COPY)
        X1, Y1 = GWdata.get_val_set(A=amp, nc=1)
        X = np.vstack( (X0,X1) ) 
        Y = np.vstack( (Y0,Y1) ) 
        

        print ("###### Trainning for A = %f with template size %d" % (amp, len(X) ) )
        patience = 0
        sess.run(counter.assign(-1))  # reset lr
        loss_in_epoches = []
        time0 = time.time()
        for e in range(MAX_EPOCHS):
            Xt, Xv, Yt, Yv = train_test_split(X, Y, test_size=0.25, shuffle=True, random_state=None)
            #Xt, _, Yt, _ = train_test_split(X, Y, test_size=0.01, shuffle=True, random_state=None)
            
            STEPS   = int(len(Xt) / BATCH) 
            for i in range(STEPS):
                xbatch = Xt[i*BATCH:(i+1)*BATCH, :]
                ybatch = Yt[i*BATCH:(i+1)*BATCH, :]
                #_, loss = sess.run( [optimizer, loss_op], feed_dict={ x:xbatch, y:ybatch }   ) 
                #if (i%10==0): print('    Loss: %f', loss) 
                
                _ = sess.run( [add_counter, optimizer], feed_dict={ x:xbatch, y:ybatch }   ) 
                #_, summary = sess.run( [optimizer, merged], feed_dict={ x:xbatch, y:ybatch }   ) 
                #train_writer.add_summary(summary, global_step=e)

            ### evaluate    
            loss, acc, sen = sess.run( [loss_op, accuracy, sensitivity],   feed_dict={x:Xv, y:Yv} )
            loss_in_epoches.append(loss)

            if e % MONITOR == 0:
                duration = time.time() - time0
                speed = STEPS * BATCH * (e+1) / duration
                print('  Epoch: %4d, loss: %10.3e a/s: %4.2f %4.2f @ %6.1fs speed: %4.0f wf/s lr=%.4e'
                      % (e, loss, acc, sen, duration, speed, sess.run(learning_rate) ) )
            if loss < TOLERANCE:
                if patience > PATIENCE: 
                    print('  Epoch: %4d, loss: %10.3e a/s: %4.2f %4.2f @ %6.1fs speed %4.0f wf/s lr=%.4e'
                          % (e, loss, acc, sen, duration, speed, sess.run(learning_rate) ) )
                    break
                patience += 1
            else:
                patience = 0
        
        save_path = saver.save(sess, "%s/model_%4.2f.ckpt" % (MODEL_PATH, amp ) )
        print("Model saved at %s" % save_path)
        
        ### plot in each epoch
        plt.semilogy(loss_in_epoches, lw=1, label="A=%s"%amp)
        
    plt.legend()
    fig_loss.savefig("loss_history.png")

print("###### Done.")
