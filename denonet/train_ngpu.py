import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from toydata import SinGauss
from wavenet import WaveNet

physical_devices = tf.config.list_physical_devices('GPU')
NGPU = len(physical_devices)
mirrored_strategy = tf.distribute.MirroredStrategy()


RATE=8192

def train_step(model, x, y, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        y_hat = model(x)
        y = tf.expand_dims(y, -1)
        loss = loss_fn(y, y_hat) + tf.reduce_sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
    return loss

@tf.function
def distributed_train_step(model, x, y, loss_fn, optimizer):
  per_replica_losses = mirrored_strategy.run(train_step, args=(model, x, y, loss_fn, optimizer))
  return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


import time
def map_data(x,y):
    xn = x + tf.random.normal([RATE])
    mean = tf.reduce_mean(xn)
    max  = tf.reduce_max(tf.abs(xn))
    return (xn-mean)/max, (x-mean)/max
def mseloss(target_y, predicted_y):
      return tf.reduce_mean(tf.square(target_y - predicted_y))

def train():
    tf.compat.v1.reset_default_graph()

    ROOT = "/work/p00lcy01/saced_model"
    REGU  = 0.001
    BATCH = 128*NGPU
    EPOCHS = 30
    NSTEP  = 2000
    
    NUM_THREADS = 4
    OUT_N_STEP = 200
    sgdata = SinGauss()
    ds = tf.data.Dataset.from_generator(sgdata.gen, output_types=(tf.float32, tf.float32))
    #ds = ds.map(lambda x,y : ( (x + np.random.normal(size=8192)), x), num_parallel_calls=NUM_THREADS ) \
    ds = ds.map(map_data, num_parallel_calls=NUM_THREADS ) \
             .repeat().prefetch(buffer_size=2500).batch(BATCH)
    train_ds = mirrored_strategy.experimental_distribute_dataset(ds)

    with mirrored_strategy.scope():
        model = WaveNet()

        try: 
            chk = "saved.h5"
            model.load_weights(chk)
            print("== Train from load weight...", chk)
        except:
            print("== Train from scratch...")

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        #loss_fn = tf.keras.losses.MeanSquaredError()
        #loss_fn = tf.keras.losses.CategoricalCrossentropy()
        loss_fn = mseloss

        for epoch in range(EPOCHS):
            print("Epoch %d" % (epoch))

            # Iterate over the batches of the dataset.
            t0 = time.perf_counter()
            for (step, (x,y)) in enumerate(train_ds):
                loss = distributed_train_step(model, x, y, loss_fn, optimizer)
                
                
                if step % OUT_N_STEP == 0:
                    tps = (time.perf_counter() - t0) / ( OUT_N_STEP*BATCH ) * 1e6
                    print("step {:3d}: mean loss = {:10.4g}, time per step = {:10.3f} us".format(step, loss, tps) )
                    t0 = time.perf_counter()
                if step >= NSTEP: break    

            if (epoch+1)%5 == 0:
                #model.save_weights("/work/p00lcy01/saced_model/wnet_{:03d}.h5".format(epoch))
                save_file = "{}/wnet_{:03d}.h5".format(ROOT,epoch+1)
                model.save_weights(save_file)
                print("Model saved at {}".format(save_file)

# +
##==== main()

train()
# -


