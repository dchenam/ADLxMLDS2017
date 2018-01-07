from train import GAN, parse
import tensorflow as tf
import pickle
import numpy as np
import skimage.io
import os
import re

np.random.seed(10)
tf.set_random_seed(10)

if __name__ == '__main__':
    args = parse()
    sess = tf.InteractiveSession()
    model = GAN(sess, args)
    sess.run(tf.global_variables_initializer())
    test_embeddings = pickle.load(open('test_embeddings.pkl', 'rb'))
    model.load()
    captions = []
    test_id = []
    with open(args.testing, 'r') as f:
        for line in f:
            test_id.extend(re.findall(r'\d', line))
            line = re.sub('\d,', ' ', line)
            captions.append(line.strip())
    for i in range(len(captions)):
        for j in range(5):
            try:
                image = model.generate(test_embeddings[captions[i]])
            except:
                print('test id %i not found in pretrained embeddings' % (test_id[i]))
            skimage.io.imsave(os.path.join('./samples', 'sample_' + test_id[i] + '_' + str(j + 1) + '.jpg'), image)