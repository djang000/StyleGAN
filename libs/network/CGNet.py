"""
The implementation of Cartoon Transfrom network ).

File author: TJ Park
Date: 24. Dec. 2018
"""


import libs.configs.config
import tensorflow as tf
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS

class CGNet(object):
    def __init__(self):
        """
        Cartoon GAN Construction
        """
        self.weight_decay = FLAGS.weight_decay
        self.style_dim = 8

    def inference(self, xa, xb):
        sa_code = tf.random_normal(shape=[1, 1, 1, 8], dtype=tf.float32)
        sb_code = tf.random_normal(shape=[1, 1, 1, 8], dtype=tf.float32)

        c_a, s_a_fake = self.Encoder_A(xa)
        c_b, s_b_fake = self.Encoder_B(xb)

        """ reconstruction image """
        self.recon_xa = self.Decoder_A(content_B=c_a, style_A=s_a_fake)
        self.recon_xb = self.Decoder_B(content_A=c_b, style_B=s_b_fake)

        """ generate image with fake style code """
        self.fake_ba = self.Decoder_A(content_B=c_b, style_A=s_a_fake, reuse=True)
        self.fake_ab = self.Decoder_B(content_A=c_a, style_B=s_b_fake, reuse=True)

        """ generate image with random style code """
        self.rand_ba = self.Decoder_A(content_B=c_b, style_A=sa_code, reuse=True)
        self.rand_ab = self.Decoder_B(content_A=c_a, style_B=sb_code, reuse=True)

        """ generate Cycle image with random style code """
        gen_ca, gen_sb = self.Encoder_B(self.rand_ab, reuse=True)
        gen_cb, gen_sa = self.Encoder_A(self.rand_ba, reuse=True)
        self.rcyc_aba = self.Decoder_A(content_B=gen_ca, style_A=sa_code, reuse=True)
        self.rcyc_bab = self.Decoder_B(content_A=gen_cb, style_B=sb_code, reuse=True)

        """ generate Cycle image with reference image code """
        gen_fca, gen_fsb = self.Encoder_B(self.fake_ab, reuse=True)
        gen_fcb, gen_fsa = self.Encoder_A(self.fake_ba, reuse=True)
        self.fcyc_aba = self.Decoder_A(content_B=gen_fca, style_A=s_a_fake, reuse=True)
        self.fcyc_bab = self.Decoder_B(content_A=gen_fcb, style_B=s_b_fake, reuse=True)

    def rand_style_infer(self, image, style_code):
        c_a, s_a_fake = self.Encoder_A(image)

        """ generate image with random style code """
        self.fake_image = self.Decoder_B(content_A=c_a, style_B=style_code)

    def train(self, dataA, dataB, random_style_A, random_style_B):
        """ weight """
        self.pw = 0.0009
        self.recon_w = 1.0
        self.recon_s_w = 0.0
        self.recon_c_w = 0.0
        self.cyc_w = 10.0
        self.gan_w = 1.0

        """ Generator params """
        self.n_res = 4
        self.mlp_dim = 256
        self.n_downsample = 2
        self.n_upsample = 2

        """ Discriminator params """
        self.n_dis = 4
        self.n_scale = 3
        print("##### Train Information #####")
        print("# dataset : ", FLAGS.dataset_name)
        print("# batch_size : ", FLAGS.batch_size)
        print("# max_iters : ", FLAGS.max_iters)

        print("# reconstruction weight : ", self.recon_w)
        print("# cycle weight : ", self.cyc_w)
        print("# Ganerator weight : ", self.gan_w)
        print("# Perceptual weight : ", self.pw)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)
        print("# Style dimension : ", self.style_dim)
        print("# MLP dimension : ", self.mlp_dim)
        print("# Down sample : ", self.n_downsample)
        print("# Up sample : ", self.n_upsample)

        print()

        print("##### Discriminator #####")
        print("# Discriminator layer : ", self.n_dis)
        print("# Multi-scale Dis : ", self.n_scale)

        # build Network
        self.dataA = dataA
        self.dataB = dataB
        self.random_styleA = random_style_A
        self.random_styleB = random_style_B
        self._build_graph()
        self.losses()

    def _build_graph(self):
        """ Encoder graph """
        # A -> B
        self.content_A, self.style_A_GT = self.Encoder_A(self.dataA)
        self.content_B, self.style_B_GT = self.Encoder_B(self.dataB)

        """ Decoder graph (reconstruction) """
        self.data_AA = self.Decoder_A(content_B=self.content_A, style_A=self.style_A_GT)
        self.data_BB = self.Decoder_B(content_A=self.content_B, style_B=self.style_B_GT)

        """ Decoder graph (cross domain Generation) """
        self.data_AB = self.Decoder_B(content_A=self.content_A, style_B=self.random_styleB, reuse=True)
        self.data_BA = self.Decoder_A(content_B=self.content_B, style_A=self.random_styleA, reuse=True)

        """ Encoding for generated data within cross domain """
        self.gen_content_A, self.gen_style_B = self.Encoder_B(self.data_AB, reuse=True)
        self.gen_content_B, self.gen_style_A = self.Encoder_A(self.data_BA, reuse=True)

        """ Decoding for Cycle reconstruction """
        self.data_ABA = self.Decoder_A(content_B=self.gen_content_A, style_A=self.style_A_GT, reuse=True)
        self.data_BAB = self.Decoder_B(content_A=self.gen_content_B, style_B=self.style_B_GT, reuse=True)

        """ Discriminator """
        self.real_A_logit = self.Discriminator(self.dataA, scope='discriminator_A')
        self.fake_A_logit = self.Discriminator(self.data_BA, reuse=True, scope='discriminator_A')

        self.real_B_logit = self.Discriminator(self.dataB, scope='discriminator_B')
        self.fake_B_logit = self.Discriminator(self.data_AB, reuse=True, scope='discriminator_B')

    def Encoder_A(self, input_A, reuse=False):
        style_A = self.Style_Encoder(input_A, reuse=reuse, scope='style_encoder_A')
        content_A = self.Content_Encoder(input_A, reuse=reuse, scope='content_encoder_A')

        return content_A, style_A

    def Encoder_B(self, input_B, reuse=False):
        style_B = self.Style_Encoder(input_B, reuse=reuse, scope='style_encoder_B')
        content_B = self.Content_Encoder(input_B, reuse=reuse, scope='content_encoder_B')

        return content_B, style_B

    def Decoder_A(self, content_B, style_A, reuse=False):
        data_BA = self.generate(content_code=content_B, style_code=style_A, reuse=reuse, scope='decoder_A')

        return data_BA

    def Decoder_B(self, content_A, style_B, reuse=False):
        data_AB = self.generate(content_code=content_A, style_code=style_B, reuse=reuse, scope='decoder_B')

        return data_AB

    def Style_Encoder(self, x, reuse, scope):
        with slim.arg_scope(training_scope(weight_decay=self.weight_decay)):
            with tf.variable_scope(scope, reuse=reuse):
                x = slim.conv2d(x, 64, [7, 7], stride=1, scope='conv_0')
                x = slim.conv2d(x, 128, [4, 4], stride=2, scope='conv_1')
                x = slim.conv2d(x, 256, [4, 4], stride=2, scope='conv_2')
                x = slim.conv2d(x, 256, [4, 4], stride=2, scope='conv_3')
                x = slim.conv2d(x, 256, [4, 4], stride=2, scope='conv_4')

                x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

                style_code = slim.conv2d(x, self.style_dim, [1, 1], stride=1, activation_fn=None, scope='Style_Code')

                return style_code


    def Content_Encoder(self, x, reuse, scope):
        with slim.arg_scope(training_scope(weight_decay=self.weight_decay)):
            with tf.variable_scope(scope, reuse=reuse):
                x = slim.conv2d(x, 64, [7, 7], stride=1, activation_fn=None, scope='conv_0')
                x = slim.instance_norm(x, activation_fn=tf.nn.relu)
                x = slim.conv2d(x, 128, [4, 4], stride=2, activation_fn=None, scope='conv_1')
                x = slim.instance_norm(x, activation_fn=tf.nn.relu)
                x = slim.conv2d(x, 256, [4, 4], stride=2, activation_fn=None, scope='conv_2')
                x = slim.instance_norm(x, activation_fn=tf.nn.relu)

                """ residual block """
                x = self.residual_block(x, scope='rb_1')
                x = self.residual_block(x, scope='rb_2')
                x = self.residual_block(x, scope='rb_3')
                x = self.residual_block(x, scope='rb_4')

                return x
    def Discriminator(self, feature, reuse=False, scope='discriminator'):
        D_logit=[]
        with tf.variable_scope(scope, reuse=reuse):
            x_ = feature
            for scale in range(self.n_scale):
                with tf.variable_scope('scale_%d'%scale, reuse=reuse):
                    x = slim.conv2d(x_, 64, [4, 4], stride=2, activation_fn=tf.nn.leaky_relu, scope='conv_0')
                    x = slim.conv2d(x, 128, [4, 4], stride=2, activation_fn=tf.nn.leaky_relu, scope='conv_1')
                    x = slim.conv2d(x, 256, [4, 4], stride=2, activation_fn=tf.nn.leaky_relu, scope='conv_2')
                    x = slim.conv2d(x, 512, [4, 4], stride=2, activation_fn=tf.nn.leaky_relu, scope='conv_3')
                    x = slim.conv2d(x, 1, [1, 1], activation_fn=None, padding='VALID', scope='D_logit')
                    D_logit.append(x)
                    x_ = tf.nn.avg_pool(x_, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

            return D_logit

    def generate(self, content_code, style_code, reuse, scope='decoder'):
        with slim.arg_scope(training_scope(weight_intitializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),
                                           weight_decay=self.weight_decay)):
            with tf.variable_scope(scope, reuse=reuse):
                mu, sigma = self.MLP(s_code=style_code, reuse=reuse)

                x = self.adaptive_residual_block(content_code, mu, sigma, scope='Adap_ResBlock_1')
                x = self.adaptive_residual_block(x, mu, sigma, scope='Adap_ResBlock_2')
                x = self.adaptive_residual_block(x, mu, sigma, scope='Adap_ResBlock_3')
                x = self.adaptive_residual_block(x, mu, sigma, scope='Adap_ResBlock_4')

                x = self.up_sampling(x, 128, scope='up_sampling_1')
                x = self.up_sampling(x, 64, scope='up_sampling_2')

                x = slim.conv2d(x, 3, [7, 7], activation_fn=None, scope='generate_logit')
                x = tf.tanh(x)

                return x

    def residual_block(self, x, scope='resblock'):
        with tf.variable_scope(scope):
            y = slim.conv2d(x, 256, [3, 3], activation_fn=None, scope='res1')
            y = slim.instance_norm(y, activation_fn=tf.nn.relu)
            y = slim.conv2d(y, 256, [3, 3], activation_fn=None, scope='res2')
            y = slim.instance_norm(y)

            return x + y

    def adaptive_residual_block(self, x_init, gamma, beta, eps=1e-5, scope='Adap_ResBlock'):
        with tf.variable_scope(scope):
            with tf.variable_scope('res1'):
                x = slim.conv2d(x_init, 256, [3, 3], activation_fn=None, scope='conv')

                """ adaptive instance norm """
                c_mean, c_var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
                c_std = tf.sqrt(c_var + eps)
                adaptive_inst_norm = gamma * ( (x - c_mean) / c_std ) + beta

                x = tf.nn.relu(adaptive_inst_norm)

            with tf.variable_scope('res2'):
                x = slim.conv2d(x, 256, [3, 3], activation_fn=None, scope='conv')

                """ adaptive instance norm """
                c_mean, c_var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
                c_std = tf.sqrt(c_var + eps)
                adaptive_inst_norm = gamma * ((x - c_mean) / c_std) + beta

            return x_init + adaptive_inst_norm

    def up_sampling(self, x, ch, scope='up_sampling'):
        with tf.variable_scope(scope):
            _, h, w, _ = x.get_shape().as_list()
            new_size = [h*2, w*2]
            up_x = tf.image.resize_nearest_neighbor(x, size=new_size)
            x = slim.conv2d(up_x, ch, [5, 5], activation_fn=None, scope='conv')
            x = slim.layer_norm(x, activation_fn=tf.nn.relu)

            return x

    def MLP(self, s_code, reuse=False, scope='MLP'):
        with tf.variable_scope(scope, reuse=reuse):
            x = slim.flatten(s_code)
            x = slim.fully_connected(x, 256)
            x = slim.fully_connected(x, 256)

            mu = slim.fully_connected(x, 256, activation_fn=None, scope='mu')
            sigma = slim.fully_connected(x, 256, activation_fn=None, scope='sigma')

            mu = tf.reshape(mu, shape=[-1, 1, 1, 256])
            sigma = tf.reshape(sigma, shape=[-1, 1, 1, 256])

            return mu, sigma


    def losses(self):
        """ 1. Reconstruction Loss (L1 Loss) """
        """ Loss for Image reconstruction """
        recon_A = tf.reduce_mean(tf.abs(self.data_AA - self.dataA))
        recon_B = tf.reduce_mean(tf.abs(self.data_BB - self.dataB))

        recon_style_A = tf.reduce_mean(tf.abs(self.gen_style_A - self.random_styleA))
        recon_style_B = tf.reduce_mean(tf.abs(self.gen_style_B - self.random_styleB))

        recon_content_A = tf.reduce_mean(tf.abs(self.gen_content_A - self.content_A))
        recon_content_B = tf.reduce_mean(tf.abs(self.gen_content_B - self.content_B))

        """ 2. Cycle Loss (L1 Loss) """
        cycA_loss = tf.reduce_mean(tf.abs(self.data_ABA - self.dataA))
        cycB_loss = tf.reduce_mean(tf.abs(self.data_BAB - self.dataB))

        """ 3. Adversial Loss (LS GAN Loss) """
        adv_GA = 0
        adv_GB = 0
        adv_DA = 0
        adv_DB = 0

        for i in range(self.n_scale):
            """ Generator loss """
            adv_GA += tf.reduce_mean(tf.squared_difference(self.fake_A_logit[i], 1.0))
            adv_GB += tf.reduce_mean(tf.squared_difference(self.fake_B_logit[i], 1.0))

            """ Discriminator loss """
            ra_loss = tf.reduce_mean(tf.squared_difference(self.real_A_logit[i], 1.0 ))
            fa_loss = tf.reduce_mean(tf.square(self.fake_A_logit[i]))
            adv_DA += tf.add(ra_loss, fa_loss)

            rb_loss = tf.reduce_mean(tf.squared_difference(self.real_B_logit[i], 1.0 ))
            fb_loss = tf.reduce_mean(tf.square(self.fake_B_logit[i]))
            adv_DB += tf.add(rb_loss, fb_loss)


        """ 4. Perceptual Loss using VGG19 network """
        _, e = self.vgg_19(self.vgg_preprocessing(self.data_AB, name='output_AB'))
        gen_featureA = e['vgg_19/conv5/conv5_3']

        _, e = self.vgg_19(self.vgg_preprocessing(self.dataA, name='output_A'), reuse=True)
        featureA = e['vgg_19/conv5/conv5_3']

        _, e = self.vgg_19(self.vgg_preprocessing(self.data_BA, name='output_BA'), reuse=True)
        gen_featureB = e['vgg_19/conv5/conv5_3']

        _, e = self.vgg_19(self.vgg_preprocessing(self.dataB, name='output_B'), reuse=True)
        featureB = e['vgg_19/conv5/conv5_3']

        perceptual_A_Loss = tf.reduce_mean(tf.squared_difference(gen_featureA, featureA))
        perceptual_B_Loss = tf.reduce_mean(tf.squared_difference(gen_featureB, featureB))
        """ Each loss of the data A and the data B """
        self.Generator_A_loss = self.gan_w * adv_GA +\
                                self.recon_w * recon_A + \
                                self.recon_s_w * recon_style_A + \
                                self.recon_c_w * recon_content_A + \
                                self.cyc_w * cycA_loss + \
                                self.pw * perceptual_A_Loss


        self.Generator_B_loss = self.gan_w * adv_GB + \
                                self.recon_w * recon_B + \
                                self.recon_s_w * recon_style_B + \
                                self.recon_c_w * recon_content_B + \
                                self.cyc_w * cycB_loss + \
                                self.pw * perceptual_B_Loss

        self.Discriminator_A_loss = self.gan_w * adv_DA
        self.Discriminator_B_loss = self.gan_w * adv_DB

        """ Total Loss """
        self.Generator_loss = self.Generator_A_loss + self.Generator_B_loss
        self.Discriminator_loss = self.Discriminator_A_loss + self.Discriminator_B_loss

    def vgg_preprocessing(self, img, name):
        vgg_mean = [103.939, 116.779, 123.68]   # BGR

        vggInput_AB = (img + 1.0) * 127.5
        r, g, b = tf.split(vggInput_AB, 3, 3)
        output = tf.concat(values=[b-vgg_mean[0], g-vgg_mean[1], r-vgg_mean[2]], axis=3, name=name)

        return output


    def vgg_19(self,
               inputs,
               is_training=False,
               reuse=False,
               scope='vgg_19'):
        """Oxford Net VGG 19-Layers version E Example.
        Note: All the fully_connected layers have been transformed to conv2d layers.
              To use in classification mode, resize input to 224x224.
        Args:
          inputs: a tensor of size [batch_size, height, width, channels].
          is_training: whether or not the model is being trained.
          scope: Optional scope for the variables.
        Returns:
          net: the output of the logits layer (if num_classes is a non-zero integer),
            or the non-dropped-out input to the logits layer (if num_classes is 0 or
            None).
          end_points: a dict of tensors with intermediate activations.
        """
        with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], trainable = is_training, scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable = is_training, scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], trainable = is_training, scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], trainable = is_training, scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], trainable = is_training, scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        return net, end_points

def training_scope(weight_intitializer=slim.initializers.xavier_initializer(),
                   weight_decay=0.00004):

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_initializer=weight_intitializer,
        activation_fn=tf.nn.relu) :
        with slim.arg_scope([slim.conv2d], padding='SAME', weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
            return sc





