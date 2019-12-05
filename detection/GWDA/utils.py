from __future__ import print_function
import numpy as np
import tensorflow as tf

def variable_summaries(var):
  """ for TensorBoard visualization """
  with tf.name_scope("summaries_%s"% var.name.replace("/", "_").replace(":", "_")):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def report_vars(summ=False):
    print ("###### [ Variables briefing ]")
    vars = 0
    for v in tf.global_variables():
        #print (v)
        vars += np.prod(v.get_shape().as_list())
    print("Global var size: %10.3f MB | Var # : %d" % (8*vars/(1024**2), len(tf.global_variables()) ) )

    vars = 0
    for v in tf.trainable_variables():
        #print (v)
        if summ: 
            variable_summaries(v)
        vars += np.prod(v.get_shape().as_list())
    print("Train  var size: %10.3f MB | Var # : %d" % (8*vars/(1024**2), len(tf.trainable_variables()) ) )

    vars = 0
    for v in tf.local_variables():
        #print (v)
        vars += np.prod(v.get_shape().as_list())
    print("Local  var size: %10.3f B  | Var # : %d" % (8*vars, len(tf.local_variables()) ) )
