import skimage.io
import skimage.transform
import numpy as np
import pandas as pd
import re
import fileinput
import os
from skip_thoughts import configuration
from skip_thoughts import encoder_manager
import pickle

VOCAB_FILE = "skip_thoughts_uni/vocab.txt"
EMBEDDING_MATRIX_FILE = "skip_thoughts_uni/embeddings.npy"
CHECKPOINT_PATH = "skip_thoughts_uni/model.ckpt-501424"

bad_idx = []
def clean_tags():
    print('cleaning tags...')
    tags = pd.read_csv('data/tags_clean.csv', header=None, sep='\t', names=list(range(50)))
    clean = tags.applymap(lambda x:re.sub('[\d,:""]','', str(x)))
    mask = clean.applymap(lambda x:("eyes" in str(x)) or ("hair" in str(x)))
    clean = clean.where(mask)
    clean.to_csv('tags_clean.txt', header=None, sep='\t')
    x = fileinput.input('tags_clean.txt', inplace=1)
    for line in x:
        line = re.sub('\t', ' ', line)
        line = re.sub("long hair", ' ', line)
        line = re.sub("short hair", ' ', line)
        if not ('hair' in line) and not ('eyes' in line): bad_idx.append(int(re.findall(r'\d+', line)[0]))
        line = re.sub('\d', ' ', line)
        line = re.sub('\s{2,}', ' ', line.strip())
        print(line)
    x.close()

def process_images():
    print('processing images...')
    print(len(bad_idx))
    img_data = []
    for i in range(33431):
        img = skimage.io.imread(os.path.join('data/faces', str(i) + '.jpg'))
        img = skimage.transform.resize(img, (64, 64))
        img_data.append(img)
    filtered_img_data = [img for i, img in enumerate(img_data) if i not in bad_idx]
    img_data = np.array(filtered_img_data)
    img_data = img_data * 2 - 1
    np.save('train_images.npy', img_data)

def encode_tags():
    print('encoding captions..')
    print(len(bad_idx))
    encoder = encoder_manager.EncoderManager()
    encoder.load_model(configuration.model_config(),
                       vocabulary_file=VOCAB_FILE,
                       embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                       checkpoint_path=CHECKPOINT_PATH)
    data = []
    with open('tags_clean.txt', 'r') as f:
        data.extend(line.strip() for line in f)
    filtered_data = [caption for i, caption in enumerate(data) if i not in bad_idx]
    encodings = encoder.encode(filtered_data)
    encodings = np.array(encodings)
    np.save('train_embeddings.npy', encodings)

    color_hairs = {'orange hair', 'white hair', 'aqua hair', 'gray hair',
        'green hair', 'red hair', 'purple hair', 'pink hair',
        'blue hair', 'black hair', 'brown hair', 'blonde hair'}

    color_eyes = {'gray eyes', 'black eyes', 'orange eyes',
        'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
        'green eyes', 'brown eyes', 'red eyes', 'blue eyes'}

    testing_data = []
    for hair in color_hairs:
        for eye in color_eyes:
            testing_data.append(hair + ' ' + eye)
            testing_data.append(eye + ' ' + hair)
    for hair in color_hairs:
        testing_data.append(hair)
    for eye in color_eyes:
        testing_data.append(eye)

    testing_encoding = encoder.encode(testing_data)
    testing_dict = dict(zip(testing_data, testing_encoding))
    test_embeddings = pickle.dump(testing_dict, open('test_embeddings.pkl', 'wb'))

if __name__ == '__main__':
    clean_tags()
    encode_tags()
    process_images()