import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D, Add, ReLU
from tensorflow.keras import regularizers


class ResBlk_R(tf.keras.layers.Layer):
    def __init__(self, res_ch, gate_ch, ks, skip_ch=None, dilation=1, lid=0, regu=0, **kwargs):
        super(ResBlk_R,self).__init__()
        self.res_ch = res_ch
        if skip_ch is None: skip_ch = res_ch

        self.tanh = Conv1D(gate_ch, kernel_size=ks, padding='causal', dilation_rate=dilation, activation='tanh',    name='tanh',
                           kernel_regularizer=regularizers.l2(regu))
        self.gate = Conv1D(gate_ch, kernel_size=ks, padding='causal', dilation_rate=dilation, activation='sigmoid', name='gate',
                          kernel_regularizer=regularizers.l2(regu))
        self.conv_s = Conv1D(skip_ch, kernel_size=1, padding='causal', name='convs',
                            kernel_regularizer=regularizers.l2(regu))
        self.conv_r = Conv1D(res_ch,  kernel_size=1, padding='causal', name='convr',
                            kernel_regularizer=regularizers.l2(regu))
        
    def call(self, inputs):
        gated = self.tanh(inputs) * self.gate(inputs)
        s = self.conv_s(gated)
        x = self.conv_r(gated) + inputs
        return x, s

class WaveNet_R(tf.keras.Model):
    def __init__(self, regu=0, rl=10, rc=32, gc=16, sc=64, quan=256):
        super(WaveNet_R,self).__init__()
        self.RL = rl
        self.RC = rc
        self.GC = gc
        self.SC = sc
        self.QUANTIZE = quan
        #self.causal = Conv1D(RC, kernel_size=2, padding='causal', kernel_initializer=tf.keras.initializers.Ones() )
        self.causal = Conv1D(self.RC, kernel_size=2, padding='causal', name='causal',
                            kernel_regularizer=regularizers.l2(regu))

        self.repeat = []
        for i in range(self.RL):
            self.repeat.append( ResBlk_R(res_ch=self.RC, gate_ch=self.GC, ks=2, skip_ch=self.SC, dilation=2**i, lid=i, regu=regu) )

        self.post = [
            Conv1D(self.SC, kernel_size=1, padding='causal', activation='relu', name='post1',
                  kernel_regularizer=regularizers.l2(regu) ),
            Conv1D(1, kernel_size=1, padding='causal', name='post2',
                  kernel_regularizer=regularizers.l2(regu) )
        ]
        
    def call(self, inputs):
        x = tf.expand_dims(inputs, -1)
        x = self.causal(x)
        sh = tf.shape(x)
        skips = tf.zeros([sh[0],sh[1],self.SC])
        for block in self.repeat:
            x, h = block(x)
            skips = skips + h

        x = ReLU()(skips)
        for layer in self.post:  x = layer(x)
        
        return x


class ResBlk(tf.keras.layers.Layer):
    def __init__(self, res_ch, gate_ch, ks, skip_ch=None, dilation=1, pad='same', **kwargs):
        super(ResBlk,self).__init__()
        self.res_ch = res_ch
        if skip_ch is None: skip_ch = res_ch

        self.tanh = Conv1D(gate_ch, kernel_size=ks, padding=pad, dilation_rate=dilation, activation='tanh',    name='tanh')
        self.gate = Conv1D(gate_ch, kernel_size=ks, padding=pad, dilation_rate=dilation, activation='sigmoid', name='gate')
        self.conv_s = Conv1D(skip_ch, kernel_size=1, padding=pad, name='convs')
        self.conv_r = Conv1D(res_ch,  kernel_size=1, padding=pad, name='convr')
        
    def call(self, inputs):
        gated = self.tanh(inputs) * self.gate(inputs)
        s = self.conv_s(gated)
        x = self.conv_r(gated) + inputs
        return x, s

class WaveNet(tf.keras.Model):
    def __init__(self, ks=3,rl=8, rc=128, gc=128, sc=128, pc=64, quan=256):
        super(WaveNet,self).__init__()
        self.KS = ks
        self.RL = rl
        self.RC = rc
        self.GC = gc
        self.SC = sc
        self.PC = pc
        self.QUANTIZE = quan

        self.Dilations = [2**i  for i in range(self.RL)]
        pad = 'same'
        #self.causal = Conv1D(RC, kernel_size=2, padding='causal', kernel_initializer=tf.keras.initializers.Ones() )
        self.causal = Conv1D(self.RC, kernel_size=self.KS, padding=pad, name='causal')

        self.repeat = []
        for i in range(self.RL):
            self.repeat.append( ResBlk(res_ch=self.RC, gate_ch=self.GC, ks=self.KS, skip_ch=self.SC, dilation=self.Dilations[i], pad=pad) )

        self.post = [
            Conv1D(self.PC, kernel_size=1, padding='same', activation='relu', name='post1'),
            Conv1D(1,       kernel_size=1, padding='same', name='post2')
        ]
        
    def receptive_field(self):
        rf = self.KS
        for d in  self.Dilations:
            rf += d*(self.KS-1)
        return rf
        
    def call(self, inputs):
        x = tf.expand_dims(inputs, -1)
        x = self.causal(x)
        sh = tf.shape(x)
        skips = tf.zeros([sh[0],sh[1],self.SC])
        for block in self.repeat:
            x, h = block(x)
            skips = skips + h

        x = ReLU()(skips)
        for layer in self.post:  x = layer(x)
        
        return x



