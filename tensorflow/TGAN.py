# Tensorflow 

import sys, os
if os.getcwd().endswith("Tensorflow TGAN"):
    root_dir = os.getcwd()
else:
    root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
import tensorflow as tf
from Model import model_base

class TGAN(model_base.NN_Base):
    def __init__(self, config):
        super(TGAN, self).__init__(config.BATCH_NORM_DECAY, config.BATCH_NORM_EPSILON)
        self.config = config

    # Define the discriminator   
    def discriminator(self, tensor, y, reuse = False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            if self.config.DATA_NAME == "rf_tensor":
                image = tf.reshape(tensor, [-1, 21])
                image = self._add_noise(tensor, stddev=0.2)
                image = tf.concat([tensor, y], 1)

                h0 = self._WN_dense(tensor, 1000, 'd_h0_wndense0', init = False)
                h0 = tf.nn.leaky_relu(h0)
                h0 = self._add_noise(h0, stddev=0.2)
                h0 = tf.concat([h0, y], 1)

                h1 = self._WN_dense(h0, 500, 'd_h1_wndense0', init=False)
                h1 = tf.nn.leaky_relu(h1)
                h1 = self._add_noise(h1, stddev=0.2)
                h1 = tf.concat([h1, y], 1)

                h3 = self._WN_dense(h4, 1, 'd_h5_wndense0', init=False)
                return tf.nn.sigmoid(h5), h5
            
             else:
                raise ValueError("The specified dataset is not yet implemented!")


    def regressor(self, tensor, train_ph, reuse = False):
        with tf.variable_scope("Regressor") as scope:
            if reuse:
                scope.reuse_variables()
            if self.config.DATA_NAME == "mnist":
                image = tf.reshape(image, [-1, 28, 28, 1])
                image = self._add_noise(image, stddev=0.3)

                h0 = self._conv2d(image, 32, k_h=3, k_w=3, d_h=1, d_w=1, name='c_h0_conv0')
                h0 = tf.nn.leaky_relu(h0)
                h0 = self._batch_norm_contrib(h0, name='c_h0_bn0', train=train_ph)
                h0 = tf.layers.max_pooling2d(h0, 2, 2)
                h0 = self._drop_out(h0, 0.5, train_ph)

                h1 = self._conv2d(h0, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='c_h1_conv0')
                h1 = tf.nn.leaky_relu(h1)
                h1 = self._batch_norm_contrib(h1, name='c_h1_bn0', train=train_ph)

                h1 = self._conv2d(h1, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='c_h1_conv1')
                h1 = tf.nn.leaky_relu(h1)
                h1 = self._batch_norm_contrib(h1, name='c_h1_bn1', train=train_ph)
                h1 = tf.layers.max_pooling2d(h1, 2, 2)
                h1 = self._drop_out(h1, 0.5, train_ph)

                h2 = tf.reduce_mean(h2, axis=[1, 2])  # Global pooling
                fm = h2
                h2 = self._linear_fc(h2, self.config.NUM_CLASSES, 'c_h2_lin')
                h2 = self._batch_norm_contrib(h2, name='c_h3_bn0', train=train_ph)
                return h2, fm

            else:
                raise ValueError("The specified dataset is not yet implemented!")

    def sampler(self, z, y, reuse = True):
        with tf.variable_scope("good_generator", reuse = reuse):
            if self.config.DATA_NAME == "rf_tensor":
                z = tf.concat([z, y], 1)
                z = self._linear_fc(z, 500, 'gg_h0_lin')
                h0 = tf.nn.softplus(z, 'gg_sp0')
                h0 = self._batch_norm_contrib(h0, 'gg_bn0', train=True)

                h1 = tf.concat([h0, y], 1)
                h1 = self._linear_fc(h1, 500, 'gg_h1_lin')
                h1 = tf.nn.softplus(h1, 'gg_sp1')
                h1 = self._batch_norm_contrib(h1, 'gg_bn1', train=True)

                h2 = tf.concat([h1, y], 1)
                h2 = self._WN_dense(h2, 28 * 28, 'gg_h2_lin')
                h2 = tf.nn.sigmoid(h2, 'gg_sp1')
                return h2     

            elif self.config.DATA_NAME == "prostate":
                return
            else:
                raise ValueError("The specified dataset is not yet implemented!")

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
    from config import Config

    class TempConfig(Config):
        DATA_NAME = "rf_tensor"
        BATCH_SIZE = 64
        NUM_CLASSES = 10
        MINIBATCH_DIS = False


    tmpconfig = TempConfig()

    tf.reset_default_graph()

    tensor = tf.ones((64, 32, 32, 3))
    y = tf.ones((64, 10))
    z = tf.ones((64, 21))
    model = TGAN(tmpconfig)

    h = model.good_generator(z, y)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        h_o = sess.run(h)

    print(h_o.shape)
