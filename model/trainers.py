import tensorflow as tf
import numpy as np
import config
import util
import sgan
import os
from ops import *

class Model(object):

    def __init__(self, vars_to_save):
        self.saver = tf.train.Saver(vars_to_save)

    def session(self, sess):
        if sess is not None:
            self.sess = sess
        else:
            cp = tf.ConfigProto()
            cp.gpu_options.allow_growth = True
            self.sess = tf.Session(config=cp)

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())

    def save(self, path):
        self.saver.save(self.sess, path)

    def restore(self, path):
        self.saver.restore(self.sess, path)

    def close(self):
        self.sess.close()

    def create_log_file(self, filename):
        self.log_file = filename
        f = open(self.log_file, 'w')
        f.close()

class Stage1(Model):

    def __init__(self, args, sess=None):
        self.session(sess)
        self.z = tf.placeholder(tf.float32, [args.batch_size, args.nz])
        self.x = tf.placeholder(tf.float32, [args.batch_size, args.nvx, args.nvx, args.nvx, 1])
        self.c = tf.placeholder(tf.float32, [args.batch_size, args.npx, args.npx, 1])
        self.l = tf.placeholder(tf.float32, [args.batch_size, args.n_cls])
        self.train = tf.placeholder(tf.bool)

        netE = sgan.Encoder()
        netG = sgan.Generator()
        netD = sgan.Discriminator(args.n_cls, args.nz)

        # Generator
        enc_edge = netE.edge(self.c, self.train, args.n_cls)
        self.x_g = netG.voxel(self.z, enc_edge, self.l, self.train)

        # Discriminator
        d_g, y_g, z_g = netD(self.x_g, self.c, self.train)
        d_r, y_r, _ = netD(self.x, self.c, self.train, reuse=True)

        # Variables
        t_vars = tf.trainable_variables()
        E_vars = [var for var in t_vars if var.name.startswith('enc_edge')]
        G_vars = [var for var in t_vars if var.name.startswith('g_voxel')]
        D_vars = [var for var in t_vars if var.name.startswith('d_voxel')]

        # Generator loss
        G_loss_adv = tf.reduce_mean(sigmoid_ce_with_logits(d_g, tf.ones_like(d_g)))
        G_loss_cls = tf.reduce_mean(sigmoid_ce_with_logits(y_g, self.l))
        G_loss_ent = tf.reduce_mean(tf.square(z_g - self.z))
        self.G_loss = G_loss_adv + G_loss_cls + G_loss_ent

        # Discriminator loss
        D_loss_adv = tf.reduce_mean(sigmoid_ce_with_logits(d_r, tf.ones_like(d_r)))
        D_loss_adv += tf.reduce_mean(sigmoid_ce_with_logits(d_g, tf.zeros_like(d_g)))
        D_loss_cls = tf.reduce_mean(sigmoid_ce_with_logits(y_r, self.l))
        self.D_loss = D_loss_adv + D_loss_cls + G_loss_ent

        # Optimizers
        self.G_opt = tf.train.AdamOptimizer(args.learning_rate, args.beta1).minimize(self.G_loss, var_list=E_vars + G_vars)
        self.D_opt = tf.train.AdamOptimizer(args.learning_rate, args.beta1).minimize(self.D_loss, var_list=D_vars)

        if sess is None:
            self.initialize()

        ma_vars = tf.moving_average_variables()
        super(Stage1, self).__init__(E_vars + G_vars + D_vars + ma_vars)

    def optimize(self, z, x, c, l):
        fd = {self.z:z, self.x:x, self.c:c, self.l:l, self.train:True}
        self.sess.run(self.D_opt, feed_dict=fd)
        self.sess.run(self.G_opt, feed_dict=fd)

    def get_errors(self, z, x, c, l):
        fd = {self.z:z, self.x:x, self.c:c, self.l:l, self.train:False}
        D_loss = self.sess.run(self.D_loss, feed_dict=fd)
        G_loss = self.sess.run(self.G_loss, feed_dict=fd)
        return D_loss, G_loss

    def generate(self, z, c, l):
        fd = {self.z:z, self.c:c, self.l:l, self.train:False}
        x_g = self.sess.run(self.x_g, feed_dict=fd)
        return x_g[:, :, :, :, 0]

    def save_log(self, z, x, c, l, epoch, batch, out_path):
        loss = self.get_errors(z, x, c, l)
        xg = self.generate(z, c, l)

        # Save generated samples
        for i, v in enumerate(xg):
            # TODO: save generated data
            pass

        # Write error rates to log_file
        with open(self.log_file, 'a') as f:
            print >> f, '{0:>3}, {1:>5}, {2:.8f}, {3:.8f}'.format(epoch, batch, loss[0], loss[1])

    def run(self, args, dataset):
        params_path = os.path.join('params', args.version)
        out_path = os.path.join('out', args.version)
        if not os.path.exists(params_path): os.mkdir(params_path)
        if not os.path.exists(out_path): os.mkdir(out_path)
        self.create_log_file(os.path.join(out_path, 'log.txt'))
        total_batch = dataset.num_examples / args.batch_size

        for epoch in range(1, args.n_iters+1):
            for batch in range(total_batch):
                x, c, l = dataset.next_batch(args.batch_size)
                z = np.random.uniform(-1, 1, size=(args.batch_size, args.nz))
                self.optimize(z, x, c, l)

                if batch % args.log_interval == 0:
                    self.save_log(z, x, c, l, epoch, batch, out_path)

            if epoch % args.save_interval == 0:
                filename = os.path.join(params_path, 'epoch-{0}.ckpt'.format(epoch))
                self.save(filename)

class Stage2(Model):

    def __init__(self, args, sess=None):
        self.session(sess)
        self.stage1 = Stage1(args, sess=self.sess)
        self.x = tf.placeholder(tf.float32, [args.batch_size, args.nvx, args.nvx, args.nvx, 3])
        self.c = tf.placeholder(tf.float32, [args.batch_size, args.npx, args.npx, 3])
        self.train = tf.placeholder(tf.bool)

        netE = sgan.Encoder()
        netG = sgan.Generator()
        netD = sgan.Discriminator(args.n_cls, args.nz)

        # Generator
        eh4, eh3, eh2 = netE.color(self.c, self.train, args.n_cls)
        self.x_g = netG.style(self.stage1.x_g, eh4, eh3, eh2, self.train)

        # Discriminator
        d_g, y_g, _ = netD(self.x_g, self.c, self.train, name='d_style')
        d_r, y_r, _ = netD(self.x, self.c, self.train, name='d_style', reuse=True)

        # Variables
        t_vars = tf.trainable_variables()
        E_vars = [var for var in t_vars if var.name.startswith('enc_color')]
        G_vars = [var for var in t_vars if var.name.startswith('g_style')]
        D_vars = [var for var in t_vars if var.name.startswith('d_style')]

        # Generator loss
        G_loss_adv = tf.reduce_mean(sigmoid_ce_with_logits(d_g, tf.ones_like(d_g)))
        G_loss_rec = tf.reduce_mean(tf.abs(self.x_g - self.x))
        self.G_loss = G_loss_adv + G_loss_rec

        # Discriminator loss
        D_loss_adv = tf.reduce_mean(sigmoid_ce_with_logits(d_r, tf.ones_like(d_r)))
        D_loss_adv += tf.reduce_mean(sigmoid_ce_with_logits(d_g, tf.zeros_like(d_g)))
        self.D_loss = D_loss_adv

        # Optimizers
        self.G_opt = tf.train.AdamOptimizer(args.learning_rate, args.beta1).minimize(self.G_loss, var_list=E_vars + G_vars)
        self.D_opt = tf.train.AdamOptimizer(args.learning_rate, args.beta1).minimize(self.D_loss, var_list=D_vars)

        if sess is None:
            self.initialize()
            self.stage1.restore(args.stage1_params_path) # Read stage1 parameters

        ma_vars = tf.moving_average_variables()
        super(Stage2, self).__init__(E_vars + G_vars + D_vars + ma_vars)

    def optimize(self, z, x, c1, c2, l):
        fd = {
            self.stage1.z: z,
            self.stage1.c: c1,
            self.stage1.l: l,
            self.stage1.train: False,
            self.x: x,
            self.c: c2,
            self.train: True
        }
        self.sess.run(self.D_opt, feed_dict=fd)
        self.sess.run(self.G_opt, feed_dict=fd)

    def get_errors(self, z, x, c1, c2, l):
        fd = {
            self.stage1.z: z,
            self.stage1.c: c1,
            self.stage1.l: l,
            self.stage1.train: False,
            self.x: x,
            self.c: c2,
            self.train: True
        }
        D_loss = self.sess.run(self.D_loss, feed_dict=fd)
        G_loss = self.sess.run(self.G_loss, feed_dict=fd)
        return D_loss, G_loss

    def generate(self, z, c1, c2, l):
        fd = {
            self.stage1.z: z,
            self.stage1.c: c1,
            self.stage1.l: l,
            self.stage1.train: False,
            self.x: x,
            self.c: c2,
            self.train: False
        }
        return self.sess.run(self.x_g, feed_dict=fd)

    def save_log(self, z, x, c1, c2, l, epoch, batch, out_path):
        loss = self.get_errors(z, x, c1, c2, l)
        xg = self.generate(z, c1, c2, l)

        # Save generated samples
        for i, v in enumerate(xg):
            # TODO: save generated data
            pass

        # Write error rates to log_file
        with open(self.log_file, 'a') as f:
            print >> f, '{0:>3}, {1:>5}, {2:.8f}, {3:.8f}'.format(epoch, batch, loss[0], loss[1])

    def run(self, args, dataset):
        params_path = os.path.join('params', args.version)
        out_path = os.path.join('out', args.version)
        if not os.path.exists(params_path): os.mkdir(params_path)
        if not os.path.exists(out_path): os.mkdir(out_path)
        self.create_log_file(os.path.join(out_path, 'log.txt'))
        total_batch = dataset.num_examples / args.batch_size

        for epoch in range(1, args.n_iters+1):
            for batch in range(total_batch):
                x, c1, c2, l = dataset.next_batch(args.batch_size)
                z = np.random.uniform(-1, 1, size=(args.batch_size, args.nz))
                self.optimize(z, x, c1, c2, l)

                if batch % args.log_interval == 0:
                    self.save_log(z, x, c1, c2, l, epoch, batch, out_path)

            if epoch % args.save_interval == 0:
                filename = os.path.join(params_path, 'epoch-{0}.ckpt'.format(epoch))
                self.save(filename)
