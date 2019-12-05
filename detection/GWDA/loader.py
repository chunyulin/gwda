###
###  Prepare data: extract, transform, load (ETL)
###
import numpy as np
import h5py as h5
#import matplotlib
#matplotlib.use('agg')             ## Need for CMD
import matplotlib.pyplot as plt

class GWInject():
    _NOISE_COPY_ = 1
    def _add_noise(self, tag, A, nc):
        var = self.f[tag]
        NX = len(var)
        NN = nc * NX
        noise = np.random.normal(0,1,(NN,self.srate))    
        X     = np.random.normal(0,1,(NX,self.srate))  + A * var[:NX,:]      
        if self.plot:
            plt.figure(figsize=(18,6))
            for i in range(len(X)):
                plt.subplot(3,5,i+1)         
                plt.plot(X[i,:])
                plt.plot(A * var[i,:])
                #plt.title(" )
                if (i > 13): break
            plt.show()

        X = np.vstack( (noise, X )  ).astype(np.float32)
        Y = np.array([0]*NN + [1]*NX).astype(np.float32).reshape(-1,1)
        #return (X, Y, random_state=0)
        return X, Y
    
    def __init__(self, fname, plot=0):
        self.fname = fname
        self.plot = plot
        self.f = h5.File(fname, "r")
        self.srate = self.f.attrs.get('srate')   ##4096/ 8192
        #print(self.srate, self.f.attrs.get('merger_idx'))
        
    def __exit__(self):
        self.f.close()
        
    def get_train_set(self, A=1.0, nc = _NOISE_COPY_):
        X, Y = self._add_noise('/train_hp', A, nc = nc)
        return X, Y
    def get_val_set(self, A=1.0, nc = _NOISE_COPY_):
        X, Y = self._add_noise('/val_hp', A, nc = nc)
        return X, Y
    def get_test_set(self, A=1.0, nc = _NOISE_COPY_):
        X, Y = self._add_noise('/test_hp', A, nc = nc)
        return X, Y
   
def GWInject_test():
    tmp = GWInject(H5_FILE, plot=0)
    RATE = tmp.srate
    print (RATE)
    X, Y = tmp.get_test_set(A=1, nc=3)
    
def plot_template(H5_FILE):
    f = h5.File(H5_FILE,'r')
    m1t = f['train_m1']
    m2t = f['train_m2']
    m1v = f['val_m1']
    m2v = f['val_m2']
    m1s = f['test_m1']
    m2s = f['test_m2']
    plt.figure(figsize=(8,8))
    plt.plot(m2t, m1t, 'r.', markersize=2, label="Training")
    plt.plot(m2v, m1v, 'b.', markersize=2, label="Validation")
    plt.plot(m2s, m1s, 'g.', markersize=2, label="Test")
    plt.axes().set_aspect('equal')
    plt.legend()
    #plt.show()
    plt.savefig("template.png")
    print ("# of whiten waveform for each set: ", len(f['train_hp']))

    f.close()