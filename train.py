"""
The implementation of Cartoon Transfrom using MultiModal ( Cross Domain Transform ).

File author: TJ Park
Date: 24. Dec. 2018
"""

import os, time
import libs.configs.config
from time import gmtime, strftime
import numpy as np
import scipy.misc as sm
import tensorflow as tf
import datasets.datapipe as datapipe
import libs.network.CGNet as model
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS

def saveimg(img, path):
    invert_img = (img + 1.) /2
    sm.imsave(path, invert_img)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    global_step = tf.train.create_global_step()
    dataA, dataB = datapipe.get_dataset()

    rand_style_A = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 1, 1, 8], name='style_A')
    rand_style_B = tf.placeholder(tf.float32,shape=[FLAGS.batch_size, 1, 1, 8], name='style_B')
    lr = tf.placeholder(tf.float32, name='learning_rate')

    """ build network """
    net = model.P2SNet()
    net.train(dataA, dataB, rand_style_A, rand_style_B)

    """ setting traning vars and Optimizer """
    global_vars = tf.global_variables()
    train_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(train_vars, print_info=True)

    G_vars = [var for var in train_vars if 'decoder' in var.name or 'encoder' in var.name]
    D_vars = [var for var in train_vars if 'discriminator' in var.name]
    vgg_vars = [var for var in global_vars if'vgg_19' in var.name]

    G_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999).minimize(net.Generator_loss,
                                                                                      var_list=G_vars)
    D_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999).minimize(net.Discriminator_loss,
                                                                                      global_step=global_step,
                                                                                      var_list=D_vars)

    """ summary """
    all_G_loss = tf.summary.scalar("Generator_loss", net.Generator_loss)
    all_D_loss = tf.summary.scalar("Discriminator_loss", net.Discriminator_loss)
    G_A_loss = tf.summary.scalar("G_A_loss", net.Generator_A_loss)
    G_B_loss = tf.summary.scalar("G_B_loss", net.Generator_B_loss)
    D_A_loss = tf.summary.scalar("D_A_loss", net.Discriminator_A_loss)
    D_B_loss = tf.summary.scalar("D_B_loss", net.Discriminator_B_loss)

    summary_op = tf.summary.merge([G_A_loss, G_B_loss, all_G_loss,
                                       D_A_loss, D_B_loss, all_D_loss,
                                       tf.summary.image(name='image/real_A', tensor=net.dataA, max_outputs=1),
                                       tf.summary.image(name='image/real_B', tensor=net.dataB, max_outputs=1),
                                       tf.summary.image(name='image/fake_A', tensor=(net.data_AB+1.)/2, max_outputs=1),
                                       tf.summary.image(name='image/fake_B', tensor=(net.data_BA+1.)/2, max_outputs=1)])

    logdir = os.path.join(FLAGS.summaries_dir, strftime('%Y%m%d%H%M%S', gmtime()))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    summary_writer = tf.summary.FileWriter(logdir, graph=tf.Session().graph)

    """ set saver for saving final model and backbone model for restore """
    saver = tf.train.Saver(max_to_keep=3, var_list=train_vars)

    """ Set Gpu Env """
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    gpu_opt = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opt)) as sess:
        sess.run(init_op)
        vgg_saver = tf.train.Saver(var_list=vgg_vars)
        vgg_saver.restore(sess, FLAGS.VGG19_model_path)
        ckpt = tf.train.get_checkpoint_state(FLAGS.last_checkpoint_model)
        """ resotre checkpoint of Backbone network """
        if ckpt is not None:
            lastest_ckpt = tf.train.latest_checkpoint(FLAGS.last_checkpoint_model)
            print('lastest', lastest_ckpt)
            re_saver = tf.train.Saver(var_list=tf.trainable_variables())
            re_saver.restore(sess, lastest_ckpt)

        """ Generate threads """
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                s_time = time.time()
                current_step = sess.run(global_step)

                """ generate random style code """
                style_a = np.random.normal(loc=0.0, scale=1.0, size=[1, 1, 1, 8])
                style_b = np.random.normal(loc=0.0, scale=1.0, size=[1, 1, 1, 8])
                learning_rate = 0.0001 * pow(0.1, current_step / 300000)
                feed_dict = {rand_style_A : style_a, rand_style_B : style_b, lr : learning_rate}

                """ Update Discriminator """
                _, d_loss = sess.run([D_opt, net.Discriminator_loss], feed_dict=feed_dict)

                """ Update Generator """
                _, g_loss = sess.run([G_opt, net.Generator_loss], feed_dict=feed_dict)

                duration_time = time.time() - s_time
                print ("""iter %d: time:%.3f(sec), d-loss %.4f, g-loss %.4f """ % (current_step, duration_time, d_loss, g_loss))

                if current_step % 1000 == 0:
                    # write summary
                    summary = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary, current_step)
                    summary_writer.flush()
                    # ra, rb, fa, fb = sess.run([net.dataA, net.dataB,
                    #                            net.data_BA, net.data_AB], feed_dict=feed_dict)
                    # saveimg(ra[0], 'ori_A.jpg')
                    # saveimg(rb[0], 'ori_B.jpg')
                    # saveimg(fa[0], 'Fake_BA.jpg')
                    # saveimg(fb[0], 'Fake_AB.jpg')

                if current_step % 3000 == 0:
                    # Save a checkpoint
                    save_path = 'output/training/Cartoon_GAN(p2s).ckpt'
                    saver.save(sess, save_path, global_step=current_step)

                if current_step + 1 == FLAGS.max_iters:
                    print('max iter : %d, current_step : %d' % (FLAGS.max_iters, current_step))
                    break

        except tf.errors.OutOfRangeError:
            print('Error occured')
        finally:
            saver.save(sess, './output/models/Cartoon_GAN_final(p2s).ckpt', write_meta_graph=False)
            coord.request_stop()

        coord.join(threads)
        sess.close()




