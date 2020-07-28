import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class SinGauss(object):
    def __init__(self, f0=10, f1=50, w=1, rate=8192):
        self.RATE = rate
        self.f0=f0
        self.f1=f1
        self.w=w
        self.N = 50
    def gen(self):
        #for _ in range(10):
        while True:
            x = np.linspace(0,2*np.pi,self.RATE)
            y = np.random.uniform(self.f0,self.f1, self.N)
            xx, yy = np.meshgrid(x, y, sparse=True)
            _ , ww = np.meshgrid(x, np.random.uniform(self.w*0.4,self.w*1.5, self.N ), sparse=True)
            _ , aa = np.meshgrid(x, np.random.uniform(0.5,1.5, self.N ), sparse=True)
            _ , dd = np.meshgrid(x, np.random.uniform(0.5,1.5, self.N ), sparse=True)
            X1 = aa * (np.sin(yy*xx)*np.exp(-((xx-dd*np.pi)/ww)**2))
            #X2 = A*(np.cos(yy*xx)*np.exp(-((xx-1.5*np.pi)/ww)**2))     #.astype(np.float32)
            #ynor = ( (y-np.min(y)) / (np.max(y)-np.min(y)) - 0.5 ) *2   ## normalize
            #Y  = np.vstack(( ynor )).astype(np.float32)
            for i in range(self.N):
                yield X1[i,:], y[i]

# +
@tf.function
def map_data(x,y):
    xn = x + tf.random.normal([np.shape(x)[0]])
    mean = tf.reduce_mean(xn)
    max  = tf.reduce_max(tf.abs(xn))
    
    return (xn-mean)/max, (x-mean)/max


def test():
    sgdata = SinGauss()
    train_ds = tf.data.Dataset.from_generator(sgdata.gen, output_types=(tf.float32, tf.float32))
    train_ds = train_ds.map(map_data).repeat().prefetch(buffer_size=1000).batch(1)
    #train_ds = train_ds.map(lambda x,y : ( (x + np.random.normal(size=8192)), x)) \
    #    .repeat().prefetch(buffer_size=1000).batch(1)

    it = train_ds.__iter__()
    x, y = next(it)
    x=np.squeeze(x)
    y=np.squeeze(y)
    plt.plot(x)    
    plt.plot(y)

# +
#test()
# -


