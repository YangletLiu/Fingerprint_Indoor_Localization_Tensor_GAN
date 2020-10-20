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


    def discriminator(self, image, y, reuse = False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            if self.config.DATA_NAME == "mnist":
                image = tf.reshape(image, [-1, 28*28])
                image = self._add_noise(image, stddev=0.2)
                image = tf.concat([image, y], 1)

                h0 = self._WN_dense(image, 1000, 'd_h0_wndense0', init = False)
                h0 = tf.nn.leaky_relu(h0)
                h0 = self._add_noise(h0, stddev=0.2)
                h0 = tf.concat([h0, y], 1)

                h1 = self._WN_dense(h0, 500, 'd_h1_wndense0', init=False)
                h1 = tf.nn.leaky_relu(h1)
                h1 = self._add_noise(h1, stddev=0.2)
                h1 = tf.concat([h1, y], 1)

                h2 = self._WN_dense(h1, 250, 'd_h2_wndense0', init=False)
                h2 = tf.nn.leaky_relu(h2)
                h2 = self._add_noise(h2, stddev=0.2)
                h2 = tf.concat([h2, y], 1)

                h3 = self._WN_dense(h2, 250, 'd_h3_wndense0', init=False)
                h3 = tf.nn.leaky_relu(h3)
                h3 = self._add_noise(h3, stddev=0.2)
                h3 = tf.concat([h3, y], 1)

                h4 = self._WN_dense(h3, 250, 'd_h4_wndense0', init=False)
                h4 = tf.nn.leaky_relu(h4)
                h4 = self._add_noise(h4, stddev=0.2)
                h4 = tf.concat([h4, y], 1)

                h5 = self._WN_dense(h4, 1, 'd_h5_wndense0', init=False)
                return tf.nn.sigmoid(h5), h5

            elif self.config.DATA_NAME == "svhn":
                image = self._drop_out(image, 0.2, True)
                yb = tf.reshape(y, [tf.shape(image)[0], 1, 1, self.config.NUM_CLASSES])
                image = self._conv_cond_concat(image, yb)

                h0 = self._WN_conv2d(image, 32, k_h=3, k_w=3, d_h=1, d_w=1, init = False, name = "d_h0_wnconv0")
                h0 = tf.nn.leaky_relu(h0)
                h0 = self._conv_cond_concat(h0, yb)

                h0 = self._WN_conv2d(h0, 32, k_h=3, k_w=3, d_h=2, d_w=2, init = False, name="d_h0_wnconv1")
                h0 = tf.nn.leaky_relu(h0)
                h0 = self._drop_out(h0, 0.2, True) # [16, 16]

                h1 = self._conv_cond_concat(h0, yb)
                h1 = self._WN_conv2d(h1, 64, k_h=3, k_w=3, d_h=1, d_w=1, init = False, name="d_h1_wnconv0")
                h1 = tf.nn.leaky_relu(h1)
                h1 = self._conv_cond_concat(h1, yb)

                h1 = self._WN_conv2d(h1, 64, k_h=3, k_w=3, d_h=2, d_w=2, init = False, name="d_h1_wnconv1")
                h1 = tf.nn.leaky_relu(h1)
                h1 = self._drop_out(h1, 0.2, True) # [8, 8]

                h2 = self._conv_cond_concat(h1, yb)
                h2 = self._WN_conv2d(h2, 128, k_h=3, k_w=3, d_h=1, d_w=1, init = False, name="d_h2_wnconv0")
                h2 = tf.nn.leaky_relu(h2)
                h2 = self._conv_cond_concat(h2, yb)

                h2 = self._conv_cond_concat(h2, yb)
                h2 = self._WN_conv2d(h2, 128, k_h=3, k_w=3, d_h=1, d_w=1, init = False, name="d_h2_wnconv1")
                h2 = tf.nn.leaky_relu(h2)

                h3 = tf.reduce_mean(h2, axis = [1, 2])
                h3 = tf.concat([h3, y], 1)
                if self.config.MINIBATCH_DIS:
                    f = self._minibatch_discrimination(h3, 100)
                    h3 = tf.concat([h3, f], 1)
                    h3 = self._linear_fc(h3, 1, 'd_h3_lin')
                else:
                    h3 = self._WN_dense(h3, 1, 'd_h3_wndense')
                return tf.nn.sigmoid(h3), h3

            elif self.config.DATA_NAME == "cifar10":
                image = self._drop_out(image, 0.2, True)
                yb = tf.reshape(y, [tf.shape(image)[0], 1, 1, self.config.NUM_CLASSES])
                image = self._conv_cond_concat(image, yb)

                h0 = self._WN_conv2d(image, 32, k_h=3, k_w=3, d_h=1, d_w=1, init=False, name="d_h0_wnconv0")
                h0 = tf.nn.leaky_relu(h0)
                h0 = self._conv_cond_concat(h0, yb)

                h0 = self._WN_conv2d(h0, 32, k_h=3, k_w=3, d_h=2, d_w=2, init=False, name="d_h0_wnconv1")
                h0 = tf.nn.leaky_relu(h0)
                h0 = self._drop_out(h0, 0.2, True)  # [16, 16]

                h1 = self._conv_cond_concat(h0, yb)
                h1 = self._WN_conv2d(h1, 64, k_h=3, k_w=3, d_h=1, d_w=1, init=False, name="d_h1_wnconv0")
                h1 = tf.nn.leaky_relu(h1)
                h1 = self._conv_cond_concat(h1, yb)

                h1 = self._WN_conv2d(h1, 64, k_h=3, k_w=3, d_h=2, d_w=2, init=False, name="d_h1_wnconv1")
                h1 = tf.nn.leaky_relu(h1)
                h1 = self._drop_out(h1, 0.2, True)  # [8, 8]

                h2 = self._conv_cond_concat(h1, yb)
                h2 = self._WN_conv2d(h2, 128, k_h=3, k_w=3, d_h=1, d_w=1, init=False, name="d_h2_wnconv0")
                h2 = tf.nn.leaky_relu(h2)
                h2 = self._conv_cond_concat(h2, yb)

                h2 = self._conv_cond_concat(h2, yb)
                h2 = self._WN_conv2d(h2, 128, k_h=3, k_w=3, d_h=1, d_w=1, init=False, name="d_h2_wnconv1")
                h2 = tf.nn.leaky_relu(h2)

                h3 = tf.reduce_mean(h2, axis=[1, 2])
                h3 = tf.concat([h3, y], 1)
                if self.config.MINIBATCH_DIS:
                    f = self._minibatch_discrimination(h3, 100)
                    h3 = tf.concat([h3, f], 1)
                    h3 = self._linear_fc(h3, 1, 'd_h3_lin')
                else:
                    h3 = self._WN_dense(h3, 1, 'd_h3_wndense')
                return tf.nn.sigmoid(h3), h3
            elif self.config.DATA_NAME == "prostate":
                return
            else:
                raise ValueError("The specified dataset is not yet implemented!")

    def classifier(self, image, train_ph, reuse = False):
        with tf.variable_scope("classifier") as scope:
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

                h2 = self._conv2d(h1, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='c_h2_conv0')
                h2 = tf.nn.leaky_relu(h2)
                h2 = self._batch_norm_contrib(h2, name='c_h2_bn0', train=train_ph)
                h2 = self._conv2d(h2, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='c_h2_conv1')
                h2 = tf.nn.leaky_relu(h2)
                h2 = self._batch_norm_contrib(h2, name='c_h2_bn1', train=train_ph)

                h2 = tf.reduce_mean(h2, axis=[1, 2])  # Global pooling
                fm = h2
                h2 = self._linear_fc(h2, self.config.NUM_CLASSES, 'c_h2_lin')
                h2 = self._batch_norm_contrib(h2, name='c_h3_bn0', train=train_ph)
                return h2, fm

            elif self.config.DATA_NAME == "svhn":
                image = self._drop_out(image, 0.2, train_ph)

                h0 = self._conv2d(image, 128, k_h = 3, k_w = 3, d_h = 1, d_w = 1, name = 'c_h0_conv0')
                h0 = tf.nn.leaky_relu(h0)
                h0 = self._batch_norm_contrib(h0, name = 'c_h0_bn0', train = train_ph)

                h0 = self._conv2d(h0, 128, k_h = 3, k_w = 3, d_h = 1, d_w = 1, name='c_h0_conv1')
                h0 = tf.nn.leaky_relu(h0)
                h0 = self._batch_norm_contrib(h0, name='c_h0_bn1', train = train_ph)

                h0 = self._conv2d(h0, 128, k_h = 3, k_w = 3, d_h = 1, d_w = 1, name='c_h0_conv2')
                h0 = tf.nn.leaky_relu(h0)
                h0 = self._batch_norm_contrib(h0, name='c_h0_bn2', train = train_ph)

                h0 = tf.layers.max_pooling2d(h0, 2, 2)
                h0 = self._drop_out(h0, 0.5, train_ph)

                h1 = self._conv2d(h0, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='c_h1_conv0')
                h1 = tf.nn.leaky_relu(h1)
                h1= self._batch_norm_contrib(h1, name='c_h1_bn0', train=train_ph)

                h1 = self._conv2d(h1, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='c_h1_conv1')
                h1 = tf.nn.leaky_relu(h1)
                h1 = self._batch_norm_contrib(h1, name='c_h1_bn1', train=train_ph)

                h1 = self._conv2d(h1, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='c_h1_conv2')
                h1 = tf.nn.leaky_relu(h1)
                h1 = self._batch_norm_contrib(h1, name='c_h1_bn2', train=train_ph)

                h1 = tf.layers.max_pooling2d(h1, 2, 2)
                h1 = self._drop_out(h1, 0.5, train_ph)

                h2 = self._conv2d(h1, 512, k_h=3, k_w=3, d_h=1, d_w=1, name='c_h2_conv0')
                h2 = tf.nn.leaky_relu(h2)
                h2 = self._batch_norm_contrib(h2, name='c_h2_bn0', train=train_ph)

                h2 = self._nin(h2, 256, name='c_h2_nin0')
                h2 = tf.nn.leaky_relu(h2)
                h2 = self._batch_norm_contrib(h2, name='c_h2_bn1', train=train_ph)

                h2 = self._nin(h2, 128, name='c_h2_nin1')
                h2 = tf.nn.leaky_relu(h2)
                h2 = self._batch_norm_contrib(h2, name='c_h2_bn2', train=train_ph)


                h2 = tf.reduce_mean(h2, axis = [1, 2]) # Global pooling
                fm = h2
                h2 = self._linear_fc(h2, self.config.NUM_CLASSES, 'c_h2_lin')
                h2 = self._batch_norm_contrib(h2, name='c_h3_bn0', train=train_ph)
                return h2, fm

            elif self.config.DATA_NAME == "cifar10":
                image = self._drop_out(image, 0.2, train_ph)

                h0 = self._conv2d(image, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='c_h0_conv0')
                h0 = tf.nn.leaky_relu(h0)
                h0 = self._batch_norm_contrib(h0, name='c_h0_bn0', train=train_ph)

                h0 = self._conv2d(h0, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='c_h0_conv1')
                h0 = tf.nn.leaky_relu(h0)
                h0 = self._batch_norm_contrib(h0, name='c_h0_bn1', train=train_ph)

                h0 = self._conv2d(h0, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='c_h0_conv2')
                h0 = tf.nn.leaky_relu(h0)
                h0 = self._batch_norm_contrib(h0, name='c_h0_bn2', train=train_ph)

                h0 = tf.layers.max_pooling2d(h0, 2, 2)
                h0 = self._drop_out(h0, 0.5, train_ph)

                h1 = self._conv2d(h0, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='c_h1_conv0')
                h1 = tf.nn.leaky_relu(h1)
                h1 = self._batch_norm_contrib(h1, name='c_h1_bn0', train=train_ph)

                h1 = self._conv2d(h1, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='c_h1_conv1')
                h1 = tf.nn.leaky_relu(h1)
                h1 = self._batch_norm_contrib(h1, name='c_h1_bn1', train=train_ph)

                h1 = self._conv2d(h1, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='c_h1_conv2')
                h1 = tf.nn.leaky_relu(h1)
                h1 = self._batch_norm_contrib(h1, name='c_h1_bn2', train=train_ph)

                h1 = tf.layers.max_pooling2d(h1, 2, 2)
                h1 = self._drop_out(h1, 0.5, train_ph)

                h2 = self._conv2d(h1, 512, k_h=3, k_w=3, d_h=1, d_w=1, name='c_h2_conv0')
                h2 = tf.nn.leaky_relu(h2)
                h2 = self._batch_norm_contrib(h2, name='c_h2_bn0', train=train_ph)

                h2 = self._nin(h2, 256, name='c_h2_nin0')
                h2 = tf.nn.leaky_relu(h2)
                h2 = self._batch_norm_contrib(h2, name='c_h2_bn1', train=train_ph)

                h2 = self._nin(h2, 128, name='c_h2_nin1')
                h2 = tf.nn.leaky_relu(h2)
                h2 = self._batch_norm_contrib(h2, name='c_h2_bn2', train=train_ph)

                h2 = tf.reduce_mean(h2, axis=[1, 2])  # Global pooling
                fm = h2
                h2 = self._linear_fc(h2, self.config.NUM_CLASSES, 'c_h2_lin')
                h2 = self._batch_norm_contrib(h2, name='c_h3_bn0', train=train_ph)
                return h2, fm
            elif self.config.DATA_NAME == "prostate":
                return
            else:
                raise ValueError("The specified dataset is not yet implemented!")

    def good_sampler(self, z, y, reuse = True):
        with tf.variable_scope("good_generator", reuse = reuse):
            if self.config.DATA_NAME == "mnist":
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
            elif self.config.DATA_NAME == "svhn":
                yb = tf.reshape(y, [tf.shape(y)[0], 1, 1, self.config.NUM_CLASSES])
                z = tf.concat([z, y], 1)

                z = self._linear_fc(z, 4 * 4 * 512, 'gg_h0_lin')
                h0 = tf.reshape(z, [-1, 4, 4, 512])
                h0 = tf.nn.relu(h0, 'gg_rl0')  # [4,4]
                h0 = self._batch_norm_contrib(h0, 'gg_bn0', train=True)
                h0 = self._conv_cond_concat(h0, yb)

                h0 = self._deconv2d(h0, 256, k_w=5, k_h=5, d_w=2, d_h=2, name='gg_dconv0')  # [8, 8]
                h0 = tf.nn.relu(h0, 'gg_rl1')
                h0 = self._batch_norm_contrib(h0, 'gg_bn1', train=True)
                h0 = self._conv_cond_concat(h0, yb)

                h1 = self._deconv2d(h0, 128, k_w=5, k_h=5, d_w=2, d_h=2, name='gg_dconv1')
                h1 = tf.nn.relu(h1, 'gg_rl2')  # [16,16]
                h1 = self._batch_norm_contrib(h1, 'gg_bn2', train=True)
                h1 = self._conv_cond_concat(h1, yb)

                h2 = self._WN_deconv2d(h1, 3, k_w=5, k_h=5, d_w=2, d_h=2, init_scale=0.1, init=False,
                                       name='gg_wndconv0')
                h2 = tf.nn.tanh(h2)  ## [32, 32]
                return h2

            elif self.config.DATA_NAME == "cifar10": #[32, 32, 3]
                # project 'z' and reshape
                yb = tf.reshape(y, [tf.shape(y)[0], 1, 1, self.config.NUM_CLASSES])
                z = tf.concat([z, y], 1)

                z = self._linear_fc(z, 4 * 4 * 512, 'gg_h0_lin')
                h0 = tf.reshape(z, [-1, 4, 4, 512])
                h0 = tf.nn.relu(h0, 'gg_rl0')  # [4,4]
                h0 = self._batch_norm_contrib(h0, 'gg_bn0', train=True)
                h0 = self._conv_cond_concat(h0, yb)

                h0 = self._deconv2d(h0, 256, k_w=5, k_h=5, d_w=2, d_h=2, name='gg_dconv0')
                h0 = tf.nn.relu(h0, 'gg_rl1')  # [4,4]
                h0 = self._batch_norm_contrib(h0, 'gg_bn1', train=True)
                h0 = self._conv_cond_concat(h0, yb)

                h1 = self._deconv2d(h0, 128, k_w=5, k_h=5, d_w=2, d_h=2, name='gg_dconv1')
                h1 = tf.nn.relu(h1, 'gg_rl2')  # [4,4]
                h1 = self._batch_norm_contrib(h1, 'gg_bn2', train=True)
                h1 = self._conv_cond_concat(h1, yb)

                h2 = self._WN_deconv2d(h1, 3, k_w=5, k_h=5, d_w=2, d_h=2, init=False, name='gg_wndconv0')
                h2 = tf.nn.tanh(h2)  ## [32, 32]
                return h2

            elif self.config.DATA_NAME == "prostate":
                return
            else:
                raise ValueError("The specified dataset is not yet implemented!")

    def forward_pass(self, z_g, y_g, x_l_c, y_l_c, x_l_d, y_l_d, x_u_d, x_u_c, train):
        """
        :param z: latent variable [200, 100]
        :param x_l_c: [200, 32, 32, 3]
        :param y_l_c: [200, 10]
        :param x_l_d: [20, 32, 32, 3]
        :param y_l_d: [20, 10]
        :param x_u_d: [180, 32, 32, 3]
        :param x_u_c: [200, 32, 32, 3]
        :return:
        """
        # output of G
        G = self.good_generator(z_g, y_g, reuse = False)

        # output of C for real images
        C_real_logits, _ = self.classifier(x_l_c, train, reuse = False)

        # output of C for unlabel images (as false examples to D)
        C_unl_logits, _ = self.classifier(x_u_c, train, reuse = True)
        C_unl_hard = tf.argmax(C_unl_logits, axis = 1)

        # output of C for unlabel images (as positive examples to C)
        C_unl_d_logits, _ = self.classifier(x_u_d, train, reuse=True)
        C_unl_d_hard = tf.argmax(C_unl_d_logits, axis = 1)

        # output of G for generated images
        C_fake_logits, _ = self.classifier(G, train, reuse = True)

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

    def forward_pass_CGAN(self, z, image, y):
        G = self.good_generator(z, y, reuse = False)
        D_real, D_real_logits = self.discriminator(image, y, reuse = False)
        D_fake, D_fake_logits = self.discriminator(G, y, reuse = True)
        D = [D_real, D_real_logits, D_fake, D_fake_logits]
        return G, D

if __name__ == "__main__":
    from config import Config

    class TempConfig(Config):
        DATA_NAME = "svhn"
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
