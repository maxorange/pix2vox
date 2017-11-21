import tensorflow as tf
import config
import util
from ops import *

class Model(object):

    def __init__(self, model_path):
        self.nvx, self.npx, self.n_cls = config.shapenet_32_64()
        self.current_shapes = None
        self.nz = 100
        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config_proto)
        self.build_model(model_path)

    def build_model(self, model_path, batch_size=1):
        self.color = tf.placeholder(tf.float32, [batch_size, self.npx, self.npx, 3])
        self.edge = tf.placeholder(tf.float32, [batch_size, self.npx, self.npx, 1])
        self.z = tf.placeholder(tf.float32, [batch_size, self.nz])
        self.label = tf.placeholder(tf.float32, [batch_size, self.n_cls])
        self.train = tf.placeholder(tf.bool)
        enc = Encoder()
        gen = Generator()

        # encoders
        edge = enc.edge(self.edge, self.train, self.n_cls)
        h4, h3, h2 = enc.color(self.color, self.train, self.n_cls)

        # generators
        self.voxel_gen = gen.voxel(self.z, edge, self.label, self.train)
        self.style_gen = gen.style(self.voxel_gen, h4, h3, h2, self.train)

        t_vars = tf.trainable_variables()
        vars_G = [var for var in t_vars if var.name.startswith('g_')]
        vars_E = [var for var in t_vars if var.name.startswith('enc_')]

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(vars_G + vars_E)
        self.saver.restore(self.sess, model_path)

    def update(self, color, edge, z, label):
        self.update_current_shapes(color, edge, z, label)

    def update_current_shapes(self, color, edge, z, label):
        feed_dict = {self.color:color, self.edge:edge, self.z:z, self.label:label, self.train:False}
        voxel = self.sess.run(self.voxel_gen, feed_dict=feed_dict)[0]
        style = self.sess.run(self.style_gen, feed_dict=feed_dict)[0]
        voxel = voxel > 0.1
        style = np.clip(util.tanh2rgb(style), 0, 255)
        self.current_shapes = np.concatenate([voxel, style], -1).astype(np.uint8)

class Generator(object):

    def voxel(self, z, edge, label, train, nc=1, nf=16, dropout=0.75, name="g_voxel", reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            batch_size, nz = z.get_shape().as_list()
            _, n_cls = label.get_shape().as_list()
            edge = tf.reshape(edge, [batch_size, 4, 4, 4, 32])

            # random noise
            l = linear(z, [nz, 4*4*4*nf*8], 'h11', bias=True)
            hz = tf.nn.dropout(tf.nn.elu(l), keep_prob(dropout, train))

            # class label
            l = linear(label, [n_cls, 4*4*4*nf*8], 'h12', bias=True)
            hl = tf.nn.dropout(tf.nn.elu(l), keep_prob(dropout, train))

            # encode edge
            c = conv3d(edge, [4, 4, 4, 32, nf*8], 'h13', bias=True, stride=1)
            hi = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            h = tf.reshape(hz + hl, [batch_size, 4, 4, 4, nf*8]) + hi

            c = deconv3d(h, [4, 4, 4, nf*4, nf*8], [batch_size, 8, 8, 8, nf*4], 'h2', bias=True)
            h = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = deconv3d(h, [4, 4, 4, nf*2, nf*4], [batch_size, 16, 16, 16, nf*2], 'h3', bias=True)
            h = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = deconv3d(h, [4, 4, 4, nf, nf*2], [batch_size, 32, 32, 32, nf], 'h4', bias=True)
            h = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = deconv3d(h, [3, 3, 3, nc, nf], [batch_size, 32, 32, 32, nc], 'h5', bias=True, stride=1)
            return tf.nn.sigmoid(c)

    def style(self, voxel, h4, h3, h2, train, nc=3, nf=16, dropout=0.75, name="g_style", reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            batch_size, _, _, _, _ = voxel.get_shape().as_list()
            _, _, _, nif4 = h4.get_shape().as_list()
            _, _, _, nif3 = h3.get_shape().as_list()
            _, _, _, nif2 = h2.get_shape().as_list()
            h4 = tf.tile(tf.expand_dims(h4, 1), [1, 4, 1, 1, 1])
            h3 = tf.tile(tf.expand_dims(h3, 1), [1, 8, 1, 1, 1])
            h2 = tf.tile(tf.expand_dims(h2, 1), [1, 16, 1, 1, 1])

            c = conv3d(voxel, [4, 4, 4, 1, nf], 'e1', bias=True, stride=1)
            e1 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv3d(e1, [4, 4, 4, nf, nf*2], 'e2', bias=True)
            e2 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv3d(e2, [4, 4, 4, nf*2, nf*4], 'e3', bias=True)
            e3 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv3d(e3, [4, 4, 4, nf*4, nf*8], 'e4', bias=True)
            e4 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = deconv3d(tf.concat([e4, h4], 4), [4, 4, 4, nf*8, nf*8+nif4], [batch_size, 8, 8, 8, nf*8], 'd6', bias=True)
            d6 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv3d(d6, [4, 4, 4, nf*8, nf*4], 'd5', bias=True, stride=1)
            d5 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = deconv3d(tf.concat([d5, h3], 4), [4, 4, 4, nf*4, nf*4+nif3], [batch_size, 16, 16, 16, nf*4], 'd4', bias=True)
            d4 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv3d(d4, [4, 4, 4, nf*4, nf*2], 'd3', bias=True, stride=1)
            d3 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = deconv3d(tf.concat([d3, h2], 4), [4, 4, 4, nf*2, nf*2+nif2], [batch_size, 32, 32, 32, nf*2], 'd2', bias=True)
            d2 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv3d(d2, [4, 4, 4, nf*2, nc], 'd1', bias=True, stride=1)
            return tf.nn.tanh(c)

class Encoder(object):

    def color(self, color, train, n_cls, nc=3, nf=64, dropout=0.75, name="enc_color", reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            c = conv2d(color, [4, 4, nc, nf], 'h1', bias=True)
            h1 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv2d(h1, [4, 4, nf, nf*2], 'h2', bias=True)
            h2 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv2d(h2, [4, 4, nf*2, nf*4], 'h3', bias=True)
            h3 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv2d(h3, [4, 4, nf*4, nf*8], 'h4', bias=True)
            h4 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            f = tf.reshape(h4, [-1, 4*4*nf*8])
            y = linear(f, [4*4*nf*8, n_cls], 'h5', bias=True)
            return h4, h3, h2

    def edge(self, edge, train, n_cls, nc=1, nf=32, dropout=0.75, name="enc_edge", reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            c = conv2d(edge, [4, 4, nc, nf], 'h1', bias=True)
            h1 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv2d(h1, [4, 4, nf, nf*2], 'h2', bias=True)
            h2 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv2d(h2, [4, 4, nf*2, nf*4], 'h3', bias=True)
            h3 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            c = conv2d(h3, [4, 4, nf*4, nf*4], 'h4', bias=True)
            h4 = tf.nn.dropout(tf.nn.elu(c), keep_prob(dropout, train))

            f = tf.reshape(h4, [-1, 4*4*nf*4])
            y = linear(f, [4*4*nf*4, n_cls], 'h5', bias=True)
            return h4

class Discriminator(object):

    def __init__(self, n_cls, nz):
        self.n_cls = n_cls
        self.nz = nz
        self.enc = Encoder()

    def __call__(self, x, c, train, nf=16, name="d_voxel", reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            shape = x.get_shape().as_list()
            batch_size = shape[0]
            
            if name == 'd_style':
                nc = 3
            else:
                nc = 1

            # add noise
            x += tf.random_normal(shape)
            c += tf.random_normal(c.get_shape())

            # encode image
            hc = self.enc.edge(c, train, self.n_cls, nc=nc, reuse=reuse)
            hc = tf.reshape(hc, [batch_size, -1])

            # encode voxels
            u = conv3d(x, [4, 4, 4, nc, nf], 'h1', bias=True, stride=1)
            hx = lrelu(u)

            u = conv3d(hx, [4, 4, 4, nf, nf*2], 'h2')
            hx = lrelu(batch_norm(u, train, 'bn2'))

            u = conv3d(hx, [4, 4, 4, nf*2, nf*4], 'h3')
            hx = lrelu(batch_norm(u, train, 'bn3'))

            u = conv3d(hx, [4, 4, 4, nf*4, nf*8], 'h4')
            hx = lrelu(batch_norm(u, train, 'bn4'))

            u = conv3d(hx, [4, 4, 4, nf*8, nf*16], 'h5')
            hx = lrelu(batch_norm(u, train, 'bn5'))
            hx = tf.reshape(hx, [batch_size, -1])

            # discriminator
            h = tf.concat([hc, hx], 1)
            d = linear(h, [h.get_shape().as_list()[-1], 1], 'd', bias=True)

            # classifier
            y = linear(hx, [hx.get_shape().as_list()[-1], self.n_cls], 'y', bias=True)

            # posterior
            u = linear(hx, [hx.get_shape().as_list()[-1], self.nz], 'z', bias=True)
            z = tf.nn.tanh(u)

            return d, y, z
