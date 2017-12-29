import numpy as np
import tensorflow as tf
from tensorflow import layers
# from tensorflow.contrib import layers
# from tensorflow.contrib.keras import layers, losses
# from tensorflow.contrib.keras import preprocessing
from tensorflow.contrib.keras import backend as K
import argparse
import os
import pickle
import skimage.io

np.random.seed(1)
tf.set_random_seed(1)

def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)

class GAN:
    def __init__(self, sess, args):
        self.sess = sess
        K.set_session(sess)
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.noise_dim = args.noise_dim
        self.encode_dim = args.encode_dim
        self.latent_dim = self.noise_dim + self.encode_dim

        self.gen_dim = args.gen_dim
        self.disc_dim = args.disc_dim
        self.batch_size = args.batch_size
        self.embed_dim = 2400
        self.image_size = 64
        self.learning_phase = tf.placeholder(tf.bool)

        # Build Network and Inputs
        self.noise_input = tf.placeholder(tf.float32, shape=[None, self.noise_dim], name='noise_input')
        self.real_image_input = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='real_image_input')
        self.wrong_image_input = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='wrong_image_input')
        self.real_caption_input = tf.placeholder(tf.float32, shape=[None, self.embed_dim], name='real_caption_input')
        self.fake_image = self.generator(self.real_caption_input, self.noise_input, training=self.learning_phase)
        disc_real_logits = self.discriminator(self.real_image_input, self.real_caption_input, training=self.learning_phase)
        disc_wrong_logits = self.discriminator(self.wrong_image_input, self.real_caption_input, reuse=True, training=self.learning_phase)
        disc_fake_logits = self.discriminator(self.fake_image, self.real_caption_input, reuse=True, training=self.learning_phase)

        # Loss
        ones = tf.ones([self.batch_size, 1, 1, 1])
        zeros = tf.zeros([self.batch_size, 1, 1, 1])
        # self.gen_loss = tf.reduce_mean(losses.binary_crossentropy(y_pred=disc_fake_logits, y_true=tf.ones_like(disc_fake_image)))
        # disc_loss1 = tf.reduce_mean(losses.binary_crossentropy(y_pred=disc_real_logits, y_true=tf.ones_like(disc_fake_image)))
        # disc_loss2 = tf.reduce_mean(losses.binary_crossentropy(y_pred=disc_wrong_logits, y_true=tf.zeros_like(disc_fake_image)))
        # disc_loss3 = tf.reduce_mean(losses.binary_crossentropy(y_pred=disc_fake_logits, y_true=tf.zeros_like(disc_fake_image)))
        self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=ones))
        disc_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_logits, labels=ones))
        disc_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_wrong_logits, labels=zeros))
        disc_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=zeros))
        self.disc_loss = disc_loss1 + (disc_loss2 + disc_loss3) * 0.5

        # Training Op
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.momentum)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.momentum)
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        disc_grads_vars = optimizer_disc.compute_gradients(self.disc_loss, var_list=disc_vars)
        disc_grads, _ = list(zip(*disc_grads_vars))
        disc_norms = tf.global_norm(disc_grads)
        self.train_disc = optimizer_disc.apply_gradients(disc_grads_vars, name='Disc_Train_Op')

        gen_grads_vars = optimizer_gen.compute_gradients(self.gen_loss, var_list=gen_vars)
        gen_grads, _ = list(zip(*gen_grads_vars))
        gen_norms = tf.global_norm(gen_grads)
        self.train_gen = optimizer_gen.apply_gradients(gen_grads_vars, name='Gen_Training_Op')

        # self.train_gen = optimizer_gen.minimize(self.gen_loss, var_list=gen_vars,
        #                                         global_step=tf.train.get_global_step())
        # self.train_disc = optimizer_disc.minimize(self.disc_loss, var_list=disc_vars,
        #                                           global_step=tf.train.get_global_step())

        # Summary Writers and Savers
        self.saver = tf.train.Saver()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.experiment_dir = os.path.abspath("./experiments/GAN-CLS2")
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        summary_dir = os.path.join(self.experiment_dir, "summaries")
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        self.summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        self.disc_summaries = tf.summary.merge([
            tf.summary.scalar("disc real loss", disc_loss1),
            tf.summary.scalar("disc wrong loss", disc_loss2),
            tf.summary.scalar("disc fake loss", disc_loss3),
            tf.summary.scalar("discriminator loss", self.disc_loss),
            tf.summary.scalar('discriminator gradient norm', disc_norms)
        ])
        self.gen_summaries = tf.summary.merge([
            tf.summary.scalar("generator loss", self.gen_loss),
            tf.summary.scalar('generator gradient norm', gen_norms)
        ])

    def generator(self, embed_input, noise_input, training=True, reuse=False):

        def deconv2d(x, filters, kernels=(4, 4), strides=(2, 2), training=True, output_shape=None):
            # x = layers.Conv2DTranspose(filters, kernel_size=kernels, strides=strides, padding='same')(x)
            # # x = layers.UpSampling2D(output_shape)
            # # x = layers.Conv2D()(filter, kernel_size=kernels, strides=strides)
            # x = layers.BatchNormalization()(x)
            # x = layers.Activation('relu')(x)
            x = layers.conv2d_transpose(x, filters, kernels, strides, padding='same')
            x = layers.batch_normalization(x, training=training)
            x = leaky_relu(x)
            return x

        s = self.image_size
        s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
        with tf.variable_scope('Generator', reuse=reuse):
            # encoding_vec = layers.Dense(self.encode_dim, name='embedding/dense')(embed_input)
            # encoding_vec = layers.LeakyReLU()(encoding_vec)
            # net = layers.concatenate([encoding_vec, noise_input])
            # net = layers.Dense(self.gen_dim * 8 * s16 * s16, name='concat/dense')(net)
            # # Deconvolution Layer in Original Paper ^
            # net = layers.BatchNormalization()(net)
            # net = layers.Reshape([s16, s16, self.gen_dim * 8])(net)
            # net = deconv2d(net, self.gen_dim * 4, output_shape=[s8, s8]) # Conv1 (8 x 8 x 512)
            # net = deconv2d(net, self.gen_dim * 2, output_shape=[s4, s4]) # Conv2 (16 x 16 x 256)
            # net = deconv2d(net, self.gen_dim, output_shape=[s2, s2]) # Conv3 (32 x 32 x 128)
            # net = layers.Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2), activation='tanh', padding='same')(net)
            # Conv4 (64 x 64 x 3)
            encoding_vec = layers.dense(embed_input, self.encode_dim, name='embedding/dense')
            encoding_vec = leaky_relu(encoding_vec)
            net = tf.concat([encoding_vec, noise_input], axis=-1)
            net = layers.dense(net, self.gen_dim * 8 * s16 * s16, name='concat/dense')
            # Deconvolution Layer in Original Paper ^
            net = tf.reshape(net, [tf.shape(embed_input)[0], s16, s16, self.gen_dim * 8])
            net = deconv2d(net, self.gen_dim * 4, output_shape=[s8, s8], training=training) # Conv1 (8 x 8 x 512)
            net = deconv2d(net, self.gen_dim * 2, output_shape=[s4, s4], training=training) # Conv2 (16 x 16 x 256)
            net = deconv2d(net, self.gen_dim, output_shape=[s2, s2], training=training) # Conv3 (32 x 32 x 128)
            net = layers.conv2d_transpose(net, 3, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.tanh, padding='same')
            return net

    def discriminator(self, img_input, embed_input, training=True, reuse=False):

        def conv2d(x, filter, kernel=(4, 4), strides=(2, 2), training=True):
            # x = layers.Conv2D(filter, kernel, strides, padding='same')(x)
            # x = layers.BatchNormalization()(x)
            # x = layers.LeakyReLU(0.2)(x)
            x = layers.conv2d(x, filter, kernel, strides, padding='same')
            x = layers.batch_normalization(x, training=training)
            x = leaky_relu(x)
            return x

        with tf.variable_scope('Discriminator', reuse=reuse):
            # encode_vec = layers.Dense(self.encode_dim, name='embedding/dense')(embed_input)
            # encode_vec = layers.LeakyReLU(0.2)(encode_vec)
            # net = conv2d(img_input, self.disc_dim)
            # net = conv2d(net, self.disc_dim * 2)
            # net = conv2d(net, self.disc_dim * 4)
            # net = conv2d(net, self.disc_dim * 8)
            # net = layers.Lambda(
            #     lambda x: K.concatenate([x[0], K.reshape(K.repeat(x[1], n=16), shape=(-1, 4, 4, self.encode_dim))])
            # )([net, encode_vec])
            # net = layers.Conv2D(1, kernel_size=[4, 4], strides=[1, 1], padding='valid', activation='sigmoid')(net)
            # net = layers.Flatten()(net)
            encode_vec = layers.dense(embed_input, self.encode_dim, name='embedding/dense')
            encode_vec = leaky_relu(encode_vec)
            net = conv2d(img_input, self.disc_dim, training=training)
            net = conv2d(net, self.disc_dim * 2, training=training)
            net = conv2d(net, self.disc_dim * 4, training=training)
            net = conv2d(net, self.disc_dim * 8, training=training)
            net = K.concatenate([net, K.reshape(K.repeat(encode_vec, n=16), shape=(-1, 4, 4, self.encode_dim))])
            net = layers.conv2d(net, 1, kernel_size=[4, 4], strides=[1, 1])
            return net

    def generate(self, caption):
        feed_dict={
            self.noise_input: np.random.normal(0., 1., [1, self.noise_dim]),
            self.real_caption_input:np.expand_dims(caption, 0),
            self.learning_phase:False
        }
        image = self.sess.run(self.fake_image, feed_dict)
        return np.squeeze(image, 0)

    def update_generator(self, batch_cap, step):
        feed_dict={
            self.noise_input: np.random.normal(0., 1., [self.batch_size, self.noise_dim]),
            self.real_caption_input: batch_cap,
            self.learning_phase:True
        }
        summaries, gen_loss, _ = self.sess.run([
            self.gen_summaries,
            self.gen_loss,
            self.train_gen], feed_dict)
        self.summary_writer.add_summary(summaries, step)
        return gen_loss

    def update_discriminator(self, batch_real_img, batch_real_cap, batch_wrong_img, step):
        feed_dict={
            self.noise_input: np.random.normal(0., 1., [self.batch_size, self.noise_dim]),
            self.real_image_input: batch_real_img,
            self.wrong_image_input: batch_wrong_img,
            self.real_caption_input: batch_real_cap,
            self.learning_phase:True
        }
        summaries, disc_loss, _ = self.sess.run([
            self.disc_summaries,
            self.disc_loss,
            self.train_disc], feed_dict)
        self.summary_writer.add_summary(summaries, step)
        return disc_loss

    def save(self, step):
        print("Saving checkpoint...")
        model_name = "GAN-CLS.ckpt"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, model_name),
                        global_step=step)
    def load(self):
        print("Loading checkpoint...")
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            print("No model found...")

def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL HW3")
    parser.add_argument('--learning_rate', default=0.0002, help='learning rate')
    parser.add_argument('--momentum', default=0.5, help='momentum')
    parser.add_argument('--batch_size', default=64, help='batch size')
    parser.add_argument('--epoch', default=300, help='number of epochs')
    parser.add_argument('--noise_dim', default=100, help='gaussian or uniform noise dim')
    parser.add_argument('--encode_dim', default=128, help='text embedding reduction dim')
    parser.add_argument('--gen_dim', default=128, help='num of conv in the first layer of generator')
    parser.add_argument('--disc_dim', default=64, help='num of conv in the first layer of discriminator')
    parser.add_argument('--load', action='store_true', help='loads latest checkpoint')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    with tf.Session() as sess:
        model = GAN(sess, args)
        sess.run(tf.global_variables_initializer())
        batch_size = args.batch_size
        noise_dim = args.noise_dim
        image_data = np.load('train_images.npy') # really bad data pipeline but...
        caption_data = np.load('train_embeddings.npy')
        test_embeddings = pickle.load(open('test_embeddings.pkl', 'rb'))
        for i in range(args.epoch):
            disc_loss = None
            gen_loss = None
            batch_indices = len(image_data) // args.batch_size
            shuffle_idx = np.arange(batch_indices)
            np.random.shuffle(shuffle_idx)
            random_idx = np.random.randint(image_data.shape[0], size=args.batch_size)
            num_batch = 0
            for index in shuffle_idx:
                batch_real_img = image_data[index * batch_size : (index + 1) * batch_size]
                batch_real_cap = caption_data[index * batch_size : (index + 1) * batch_size]
                batch_wrong_img = image_data[random_idx]
                step = i * batch_indices + num_batch
                disc_loss = model.update_discriminator(batch_real_img, batch_real_cap, batch_wrong_img, step)
                gen_loss = model.update_generator(batch_real_cap, step)
                image = model.generate(test_embeddings['blue hair blue eyes'])
                model.summary_writer.flush()
                print('Epoch %i: Batch: %i(%i): Generator Loss: %f, Discriminator Loss: %f' % (
                    i, num_batch, len(shuffle_idx), gen_loss, disc_loss))
                num_batch += 1
            print('Epoch %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gen_loss, disc_loss))
            if i % 10 == 0:
                model.save(i)
            if i % 3 == 0:
                image = model.generate(test_embeddings['blue hair blue eyes'])
                sample_dir = os.path.join(model.experiment_dir, 'samples')
                if not os.path.exists(sample_dir):
                    os.makedirs(sample_dir)
                skimage.io.imsave(os.path.join(sample_dir, str(i) + '.jpg'), image)
        print('Training Finished...')