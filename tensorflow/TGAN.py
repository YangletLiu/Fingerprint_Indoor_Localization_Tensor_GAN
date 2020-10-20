# Tensorflow 

import numpy as np
import sys, os
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import transform

if os.getcwd().endswith("Tensorflow TGAN"):
    root_dir = os.getcwd()
else:
    root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)

from Model import model_base

learning_rate = 1e-3
LAMBDA = 10

def lrelu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)

def relu(x):
    return tf.nn.relu(x)

def elu(x):
    return tf.nn.elu(x)

def xavier_init(size):
    input_dim = size[0]
    stddev = 1. / tf.sqrt(input_dim / 2.)
    return tf.random_normal(shape=size, stddev=stddev)

def he_init(size, stride):
    input_dim = size[2]
    output_dim = size[3]
    filter_size = size[0]

    fan_in = input_dim * filter_size ** 2
    fan_out = output_dim * filter_size ** 2 / (stride ** 2)
    stddev = tf.sqrt(4. / (fan_in + fan_out))
    minval = -stddev * np.sqrt(3)
    maxval = stddev * np.sqrt(3)
    return tf.random_uniform(shape=size, minval=minval, maxval=maxval)

class Network(object):
    def __init__(self):
        self.layer_num = 0
        self.weights = []
        self.biases = []
        
    def dense(self, input, output_dim):
        with tf.variable_scope('dense' + str(self.layer_num)):
            input_dim = input.get_shape().as_list()[1]

            init_w = xavier_init([input_dim, output_dim])
            weight = tf.get_variable('weight', initializer=init_w)

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable('bias', initializer=init_b)

            output = tf.add(tf.matmul(input, weight), bias)

            self.layer_num += 1
            self.weights.append(weight)
            self.biases.append(bias)

        return output
    
    def conv2d(self, input, input_dim, output_dim, filter_size, stride, padding='SAME'):
        with tf.variable_scope('conv' + str(self.layer_num)):
            init_w = he_init([filter_size, filter_size, input_dim, output_dim], stride)
            weight = tf.get_variable(
                'weight',
                initializer=init_w
            )

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable(
                'bias',
                initializer=init_b
            )

            output = tf.add(tf.nn.conv2d(
                input,
                weight,
                strides=[1, stride, stride, 1],
                padding=padding
            ), bias)

            self.layer_num += 1
            self.weights.append(weight)
            self.biases.append(bias)

        return output

    def deconv2d(self, input, output_dim, filter_size, stride, padding='SAME'):
        with tf.variable_scope('deconv' + str(self.layer_num)):
            input_shape = input.get_shape().as_list()
            init_w = he_init([filter_size, filter_size, output_dim, input_shape[3]], stride)
            weight = tf.get_variable(
                'weight',
                initializer=init_w
            )

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable(
                'bias',
                initializer=init_b
            )

            output = tf.add(tf.nn.conv2d_transpose(
                value=input,
                filter=weight,
                output_shape=[
                    tf.shape(input)[0],
                    input_shape[1] * stride,
                    input_shape[2] * stride,
                    output_dim
                ],
                strides=[1, stride, stride, 1],
                padding=padding
            ), bias)
            output = tf.reshape(output,
                                [tf.shape(input)[0], input_shape[1] * stride, input_shape[2] * stride, output_dim])

            self.layer_num += 1
            self.weights.append(weight)
            self.biases.append(bias)

        return output
    
    def batch_norm(self, input, scale=False):
        ''' batch normalization
        ArXiv 1502.03167v3 '''
        with tf.variable_scope('batch_norm' + str(self.layer_num)):
            output = tf.contrib.layers.batch_norm(input, scale=scale)
            self.layer_num += 1

        return output
    
class TGAN(object):
    def __init__(self, T_shape, latent_dim):
        self.T_shape = T_shape
        self.latent_dim = latent_dim
        # self.learning_rate = learning_rate
        # self.vgg = VGG19(None, None, None)

        self.G_params = []
        self.D_params = []
        
        self.y = tf.placeholder(
            tf.float32,
            [None, self.T_shape[0]*2, self.T_shape[1]*2, self.T_shape[2]],
            name='x'
        )
        self.z = tf.placeholder(tf.float32, [None, self.latent_dim], name='z')
        self.x = self.downscale(self.y, 2)

        with tf.variable_scope('generator'):
            self.g = self.generator(self.z)
        with tf.variable_scope('discriminator') as scope:
            self.D_real = self.discriminator(self.x)
            scope.reuse_variables()
            self.D_fake = self.discriminator(self.g)

        disc_loss = -tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake)
        gen_loss = -tf.reduce_mean(self.D_fake)

        alpha = tf.random_uniform(
            # shape=[self.batch_size, 1],
            shape=(tf.shape(self.y)[0], 1),
            minval=0.,
            maxval=1.
        )

        x_ = tf.reshape(self.x, [-1, np.prod(self.T_shape)])
        g_ = tf.reshape(self.g, [-1, np.prod(self.T_shape)])

        differences = x_ - g_
        interpolates = x_ + alpha * differences
        interpolates = tf.reshape(interpolates, (-1, self.T_shape[0], self.T_shape[1], self.T_shape[2]))
        gradients = tf.gradients(self.discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        self.D_loss = disc_loss + LAMBDA * gradient_penalty
        # self.G_loss = content_loss + self.SIGMA * gen_loss
        self.G_loss = gen_loss

        self.D_opt = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.5,
            beta2=0.9
        ).minimize(self.D_loss, var_list=self.D_params)
        self.G_opt = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.5,
            beta2=0.9
        ).minimize(self.G_loss, var_list=self.G_params)
                
            
     def discriminator(self, x):
        D = Network()
        # Network.conv2d(input, output_dim, filter_size, stride, padding='SAME')
        h = D.dense(x, self.T_shape[2], 32, 1, 1). # Use 1 x 1 kernel and stride 1 
        h = lrelu(h)

        # h = D.conv2d(h, 64, 64, 3, 1)
        # h = lrelu(h)
        # h = D.batch_norm(h)

        map_nums = [32, 64]

        for i in range(len(map_nums) - 1):
            h = D.conv2d(h, map_nums[i], map_nums[i + 1], 1, 1)  # Use 1 x 1 kernel and stride 1 
            h = lrelu(h)
            h = D.batch_norm(h)

        h_shape = h.get_shape().as_list()
        h = tf.reshape(h, [-1, h_shape[1] * h_shape[2] * h_shape[3]])
        h = D.dense(h, 1024)
        h = lrelu(h)

        h = D.dense(h, 1)

        self.D_params = D.weights + D.biases

        return h


    def regressor(self, x):
        G = Network()
        # Network.deconv2d(input, input_shape, output_dim, filter_size, stride)
        # h = G.dense(z, np.prod((self.T_shape[0], self.T_shape[1], 64)))
        h = lrelu(G.dense(x, np.prod((4, 4, 256))))
        # h = tf.reshape(h, (tf.shape(h)[0], self.T_shape[0], self.T_shape[1], 64))
        h = tf.reshape(h, (tf.shape(h)[0], 4, 4, 256))
        h = lrelu(G.deconv2d(h, 128, 5, 2))
        h = h[:,:7,:7,:]
        h = lrelu(G.deconv2d(h, 64, 5, 2))

        # h = G.residual_block(h, 64, 3, 2)

        h = G.deconv2d(h, self.T_shape[2], 3, 1)
        h = tf.nn.sigmoid(h)

        self.G_params = G.weights + G.biases

        return h



    def forward_pass(self, z_g, y_g, x_l_c, y_l_c, x_l_d, y_l_d, x_u_d, x_u_c, train):
        
        # output of G
        G = self.good_generator(z_g, y_g, reuse = False)

        # output of D for true fingerprints
        C_real_logits, _ = self.regressor(x_l_c, train, reuse = False)

        # output of R for unlabel fingerprints)
        C_unl_logits, _ = self.regressor(x_u_c, train, reuse = True)
        C_unl_hard = tf.argmax(C_unl_logits, axis = 1)

        # output of R for unlabel fingprintes
        C_unl_d_logits, _ = self.regressor(x_u_d, train, reuse=True)
        C_unl_d_hard = tf.argmax(C_unl_d_logits, axis = 1)

        # output of G for recovered fingerprints
        C_fake_logits, _ = self.regressor(G, train, reuse = True)

        # output of D for positive examples
        X_P = tf.concat([x_l_d, x_u_d], axis = 0)
        Y_P = tf.concat([y_l_d, tf.one_hot(C_unl_d_hard, depth = self.config.NUM_CLASSES)], axis = 0)
        X_P.set_shape([self.config.BATCH_SIZE_L_D + self.config.BATCH_SIZE_U_D] + self.config.IMAGE_DIM)
        Y_P.set_shape([self.config.BATCH_SIZE_L_D + self.config.BATCH_SIZE_U_D, self.config.NUM_CLASSES])

        D_real, D_real_logits = self.discriminator(X_P, Y_P, reuse = False)

        # output of D for generated examples
        D_fake, D_fake_logits = self.discriminator(G, y_g, reuse = True)

        # output of D for unlabeled examples (negative example)
        D_unl, D_unl_logits = self.discriminator(x_u_c, tf.one_hot(C_unl_hard, depth = self.config.NUM_CLASSES),\
                                                         reuse = True)

        return [G, [D_real, D_real_logits, D_fake, D_fake_logits, D_unl, D_unl_logits],
                [C_real_logits, C_unl_logits, C_unl_d_logits, C_fake_logits]]

    def forward_pass_CGAN(self, z, tensor, y):
        G = self.good_generator(z, y, reuse = False)
        D_real, D_real_logits = self.discriminator(tensor, y, reuse = False)
        D_fake, D_fake_logits = self.discriminator(G, y, reuse = True)
        D = [D_real, D_real_logits, D_fake, D_fake_logits]
        return G, D

if __name__ == "__main__":
        
    batch_size = 32 #64
    step_num = 5000 #3000
    latent_dim = 64 #128
    
    #from tensorflow.examples.tutorials.mnist import input_data
    
    # Load mat file, refer: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html
    from os.path import dirname, join as pjoin
    import scipy.io as sio
    
    data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data'). # check your own path
    mat_fname = pjoin(data_dir, 'tensor.mat')
    
    tensor = sio.loadmat(mat_fname)
    
    g = GAN([21, 1, 1], latent_dim)
    
    if not os.path.exists('./backup/'):
        os.mkdir('./backup/')
    if not os.path.exists('./out/'):
        os.mkdir('./out/')

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    if tf.train.get_checkpoint_state('./backup/'):
        saver.restore(sess, './backup/')
        print('********Restore the latest trained parameters.********')

    for step in range(step_num):
        for _ in range(5):
            xs, _ = data.train.next_batch(batch_size)
            xs = np.reshape(xs, (-1, 21, 1, 1))  # Use 1 x 1 kernel and stride 1 
            zs = np.random.uniform(size=[batch_size, latent_dim])
            # xs = np.expand_dims(xs, axis=-1) 
            _, dloss = sess.run([g.D_opt, g.D_loss], feed_dict={g.z:zs, g.y:xs})

        zs = np.random.uniform(size=[batch_size, latent_dim])
        xs, _ = data.train.next_batch(batch_size)
        xs = np.reshape(xs, (-1, 21, 1, 1))   # Use 1 x 1 kernel and stride 1 
        # xs = np.expand_dims(xs, axis=-1)
        _, gloss = sess.run([g.G_opt, g.G_loss], feed_dict={g.z:zs, g.y:xs})

        if step % 100 == 0:  #output every 100 steps
            saver.save(sess, './backup/', write_meta_graph=False)
            zs = np.random.uniform(size=[3, latent_dim])
            gs = sess.run(g.g, feed_dict={g.z:zs})
            show_result(gs[0], gs[1], gs[2])
            print('step: {}, D_loss: {}, G_loss:{}'.format(step, dloss, gloss))
