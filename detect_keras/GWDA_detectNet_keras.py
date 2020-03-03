#!/usr/bin/env python
# coding: utf-8
# %%
from __future__ import print_function
import os
import time
import h5py as h5
import numpy as np
import sklearn
import tensorflow as tf
from tensorflow import keras
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

print ("Numpy      ver. ", np.__version__)
print ("H5Py       ver. ", h5.__version__)
print ("SKLearn    ver. ", sklearn.__version__)
print ("TensorFlow ver. ", tf.__version__)
print ("Keras      ver. ", keras.__version__)

#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#np.random.seed(777)
#tf.set_random_seed(777)

import matplotlib
import matplotlib.pyplot as plt

# #%matplotlib inline
matplotlib.use('agg')

# %%
H5_FILE="../white_h_4096_dm2.h5"
H5_FILE="../white_h_8192_dm1.h5"
H5_FILE="../white_h_8192_dm2.h5"

# #!wget http://grqc.ncts.ncku.edu.tw/~lincy/GWDA/white_h_8192_dm2.h5
# #!ls /tmp/tf_tmp -al

# Load data
import GWDA.loader
GWdata = GWDA.loader.GWInject(H5_FILE, plot=0)
RATE   = GWdata.srate
print("Sampling rate : ", RATE)

#X0, Y0 = GWdata.get_train_set(nc=5)
#GWDA.loader.plot_template(H5_FILE)


# %%
###
###  Construct TF graph
###
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#ROOT_FOLDER = './model'
ROOT_FOLDER = '/tmp/tf_model'
ROOT_FOLDER = '/home/p00lcy01/TWCC_con/save/kcnn3_relu_tmp'
ROOT_FOLDER = '/home/p00lcy01/TWCC_con/save/kcnn3_tanh_v2'
ROOT_FOLDER = '/home/p00lcy01/TWCC_con/save/kcnn3_elu'
act = tf.nn.relu
act = tf.nn.tanh
act = tf.nn.elu

tf.reset_default_graph()
DIM   = RATE
LRATE        = 1e-4   ###1e-5 for relu   ##-4
LRATE_MIN    = 1e-6
LRATE_DECAY  = 0.85
LRATE_WIDTH  = 500
LRATE_FACTOR = 0.75

#keep_prob = tf.placeholder(tf.float32)   ##  for dropout, not used.
#lrate     = tf.placeholder(tf.float32)

##########################################
##import GWDA.model
###########################################
F = [16,32,64]
K = [16, 8, 8]
D = [ 1, 1, 1]
S = [ 1, 1, 1]
PO= [ 4, 4, 4]
PS= [ 4, 4, 4]
args0 = { "padding":'valid', "data_format":"channels_last" }
ki  = keras.initializers.TruncatedNormal() ##(seed=2)
ki0 = keras.initializers.Zeros()
input_data = keras.layers.Input(shape=(RATE,1))
a1=keras.layers.Conv1D(F[0], K[0], S[0], dilation_rate=D[0], kernel_initializer=ki, bias_initializer=ki0, activation=act, **args0)(input_data)
a1=keras.layers.MaxPooling1D(PO[0], PS[0], **args0)(a1)
a1=keras.layers.Conv1D(F[1], K[1], S[1], dilation_rate=D[1], kernel_initializer=ki, bias_initializer=ki0, activation=act, **args0)(a1)
a1=keras.layers.MaxPooling1D(PO[1], PS[1], **args0)(a1)
a1=keras.layers.Conv1D(F[2], K[2], S[2], dilation_rate=D[2], kernel_initializer=ki, bias_initializer=ki0, activation=act, **args0)(a1)
a1=keras.layers.MaxPooling1D(PO[2], PS[2], **args0)(a1)
a1=keras.layers.Flatten()(a1)
a2=keras.layers.Dense(64, kernel_initializer=ki, bias_initializer=ki0, activation=act, name='fc1')(a1)
a2=keras.layers.Dense(1, activation=None, name='prediction')(a2)
model = keras.models.Model(input_data, a2)

model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(lr=LRATE), 
              metrics=['accuracy', 
                       keras.metrics.Recall(), 
                       keras.metrics.TruePositives(name='tp'),
                       keras.metrics.FalsePositives(name='fp'),
                       keras.metrics.TrueNegatives(name='tn'),
                       keras.metrics.FalseNegatives(name='fn'),
                       keras.metrics.AUC(num_thresholds=200, name='auc', curve='PR',)
                                          ])

model.summary()
print(model.metrics_names)

#### Save model
if not os.path.exists(ROOT_FOLDER):
    os.makedirs(ROOT_FOLDER)
model_json = model.to_json()
modelcpt = "%s/model.json" % (ROOT_FOLDER) 
with open(modelcpt, "w") as json_file:
    json_file.write(model_json)

#### Load model
# load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)


# %%
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope("summaries_%s"% var.name.replace("/", "_").replace(":", "_")):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

    
### check all vari\ables
tf.global_variables_initializer()
if 1:
    vars = 0
    for v in tf.global_variables():
        print (v)
        vars += np.prod(v.get_shape().as_list())
    print("Whole size: %.3f MB | Var # : %d" % (8*vars/(1024**2), len(tf.global_variables()) ) )

    vars = 0
    for v in tf.trainable_variables():
        print (v)
        #variable_summaries(v)
        vars += np.prod(v.get_shape().as_list())
    print("Model size: %.3f MB | Var # : %d" % (8*vars/(1024**2), len(tf.trainable_variables()) ) )

    vars = 0
    for v in tf.local_variables():
        print (v)
        vars += np.prod(v.get_shape().as_list())
    print("Local var size: %.3f Bytes | Var # : %d" % (8*vars, len(tf.local_variables()) ) )


# %%
BATCH = 2048
EPOCHS = 5000
MONITOR = 10

PATIENCE    = 8
PATIENCE_PT = 2

#TOLLERENCE = 1.e-4
TOLLERENCE = 1.e-9


# %%
##
##  Training with fixed template ....
##
NOISE_COPY = 10

#TEST_LIST = [1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
#TRAIN_A =   [1.4,1.2,1.0,0.8]
TRAIN_A =   [1.8, 1.6, 1.4, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

def lr_scheduler(e):
    if e < 100: return LRATE
    lr = LRATE * LRATE_DECAY**(e/float(LRATE_WIDTH))
    if lr < LRATE: 
        return LRATE
    else:
        return lr
    

###=================================================
plt.figure()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    #merged = tf.summary.merge_all()   ## call merged to do every summary

    try:
        CKPT = "%s/model_%4.2f.h5" % ("/home/p00lcy01/save/kcnn3_tanh", 3.0 )
        model.load_weights(CKPT)
        print("== Load weight from ", CKPT)
    except:
        print("== Trainning for scratch in : ", ROOT_FOLDER  )

        
    for amp in TRAIN_A:

        time0 = time.time()
        X0, Y0 = GWdata.get_train_set(A=amp, nc=NOISE_COPY)
        X1, Y1 = GWdata.get_val_set(A=amp, nc=NOISE_COPY)
        X = np.vstack( (X0,X1) ).astype('float32')
        Y = np.vstack( (Y0,Y1) ).astype('float32') 
        X = np.expand_dims(X, axis=-1)  

        print("== Amp: %f with template size %d. Stage-in time: %f s"% (amp, len(X), time.time()-time0  ))
        
        patience    = 0
        patience_pt = 0
        time0 = time.time()

        el_in_epoches = []
        ea_in_epoches = []
        er_in_epoches = []

        tl_in_epoches = []
        ta_in_epoches = []
        tr_in_epoches = []

        loss0 = 0
        for e in range(EPOCHS):

            ## reset LR
            lr = lr_scheduler(e)
            keras.backend.set_value(model.optimizer.lr, lr )

            Xt, Xv, Yt, Yv = train_test_split(X, Y, test_size=0.22, shuffle=True, random_state=777)   ##None

            STEPS = int(len(Xt) / BATCH) 
            tl=ta=0
            tf.local_variables_initializer().run()
            for i in range(STEPS):
                xbatch = Xt[i*BATCH:(i+1)*BATCH]
                ybatch = Yt[i*BATCH:(i+1)*BATCH]
                tl, ta, tr, _,_,_,_,_ =  model.train_on_batch(xbatch, ybatch, reset_metrics=False)   ## toa: train output array
            
            tl_in_epoches.append(tl)
            ta_in_epoches.append(ta)
            tr_in_epoches.append(tr)

            ###
            ### evaluate    
            ###
            #el, ea, er, etp, efp, etn, efn, eauc = model.evaluate(Xv, Yv, verbose=0)   #, batch_size=128
            el, ea, er, _,_,_,_,_ = model.evaluate(Xv, Yv, verbose=0)   #, batch_size=128
            el_in_epoches.append(el)
            ea_in_epoches.append(ea)
            er_in_epoches.append(er)
            
            if e % MONITOR == 0:
                duration = time.time() - time0
                speed = STEPS * BATCH * (e+1) / duration
                print('  Epoch: %3d, loss: %10.3e %10.3e a/s: %4.2g %4.2f speed: %5.0f wf/s lr=%.4e' 
                      % (e, tl, el, ea, er, speed, keras.backend.get_value(model.optimizer.lr)) )
            
            ## Check convergence
            if el < TOLLERENCE:
                if patience > PATIENCE: break
                patience += 1
            else:
                patience = 0

            # ReduceLROnPlateau    
            #if np.abs(el/loss0-1) < 0.001:
            #    loss0 = el
            #    if patience_pt > PATIENCE_PT: 
            #        lr_ = keras.backend.get_value(model.optimizer.lr)
            #        if lr_ > LRATE_MIN: keras.backend.set_value(model.optimizer.lr, lr_ * LRATE_FACTOR)
            #        continue
            #    patience_pt += 1
            #else:
            #    patience_pt = 0
        
        ### 
        model.save_weights("%s/model_%4.2f.h5" % (ROOT_FOLDER, amp ))
        print("Saved weight with A = ", amp)

        ### dump loss history
        with open("%s/his_loss.log" % (ROOT_FOLDER), 'a') as f:
            f.write("### A=%4.2f ## tl/el/ta/ea/tr/er \n" % (amp)  )
            for i in range( len(tl_in_epoches) ):
                f.write("%5d %e %e %5.3f %5.3f  %5.3f %5.3f\n" % ( i, tl_in_epoches[i], el_in_epoches[i], ta_in_epoches[i], ea_in_epoches[i], tr_in_epoches[i], er_in_epoches[i])  )
            f.write("\n\n")

        ### plot in each epoch
        plt.clf()
        plt.semilogy(el_in_epoches,      lw=1, label="A=%s"%amp)
        plt.semilogy(tl_in_epoches, '.', lw=1)
        plt.legend()
        plt.savefig("%s/his_loss%f.png" % (ROOT_FOLDER, amp))
        
        plt.clf()
        plt.plot(ea_in_epoches, 'k',  lw=1, label="A=%s acc"%amp)
        plt.plot(ta_in_epoches, 'k.', lw=1)
        plt.plot(er_in_epoches, 'r',  lw=1, label="A=%s recall"%amp)
        plt.plot(tr_in_epoches, 'r.', lw=1)
        plt.legend()
        plt.savefig("%s/his_acc%f.png" % (ROOT_FOLDER, amp))
        


# %%
###
###  Testing...
###
import GWDA.loader

NOISE_COPY = 2
H5_FILE="../white_h_8192_dm2.h5"
GWdata = GWDA.loader.GWInject(H5_FILE, plot=0)
RATE   = GWdata.srate
print("Sampling rate : ", RATE)

BATCH = 8192
TRAIN_A =   [1.8, 1.6, 1.4, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
TEST_LIST = np.linspace(1.4, 0.01, 32)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    #fig1 = plt.figure()
    #fig2 = plt.figure()
    
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    
    for m in TRAIN_A:
        CKPT = "%s/model_%4.2f.h5" % (ROOT_FOLDER, m )
        try:
            model.load_weights(CKPT)
            print("== Load : ", CKPT)
        except: 
            print("== Found no file ", CKPT)
            continue
        
        gacc=[]
        gsen=[]
        groc=[]
        gpr=[]

        for amp in TEST_LIST:

            X, Y  = GWdata.get_test_set(A=amp, nc=NOISE_COPY)
            X = np.expand_dims(X, axis=-1)  
            Xts, _, Yts, _ = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=222)

            time0 = time.time()

            #i = 1 ##int(np.random.random()* int(len(Xts)/BATCH) )
            #xbatch = Xts[i*BATCH:(i+1)*BATCH, :]
            #ybatch = Yts[i*BATCH:(i+1)*BATCH, :]
            #el, ea, er, etp, efp, etn, efn, eauc = model.test_on_batch(xbatch, ybatch, reset_metrics=True) 
            
            el, ea, er, etp, efp, etn, efn, eroc, epr = model.evaluate(Xts, Yts, verbose=0, batch_size=512)
            
            gacc.append(ea)
            gsen.append(er)
            groc.append(eroc)
            gpr.append(epr)
            print(" A= %4.2f : a/s: %7.3f %7.3f, TP/FP/TN/FN: %5d %5d %5d %5d roc/pr: %7.3f %7.3f speed: %5d wf/s" 
                  % (amp, ea, er, etp, efp, etn, efn, eroc, epr, len(Yts)/(time.time()-time0) ) )

            ###
            #yprob = model.predict(Xts, verbose=0, batch_size=512)
            #ff = roc_curve(Yts, yprob)
            #print(ff)
            #print(Yts)
            
        ax1.plot(TEST_LIST, gsen, label="m:%f"%m)
        ax2.plot(TEST_LIST, groc, label="m:%f"%m)
        ax3.plot(TEST_LIST, gpr, label="m:%f"%m)
    
    FIGNAME = "%s/infer.png" % (ROOT_FOLDER)
    ax1.set_xlabel("Amplitude of injected template")
    ax1.set_ylabel("Sensitivity")
    ax1.legend()
    fig1.savefig(FIGNAME)
    FIGNAME_ROC = "%s/infer_roc.png" % (ROOT_FOLDER)
    ax2.set_xlabel("Amplitude of injected template")
    ax2.set_ylabel("ROC")
    ax2.legend()
    fig2.savefig(FIGNAME_ROC)
    FIGNAME_PR = "%s/infer_pr.png" % (ROOT_FOLDER)
    ax3.set_xlabel("Amplitude of injected template")
    ax3.set_ylabel("PR")
    ax3.legend()
    fig3.savefig(FIGNAME_PR)
