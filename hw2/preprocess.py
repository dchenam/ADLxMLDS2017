def get_feat(data_dir='MLDS_hw2_data/', type='train'):
    import glob
    import os
    import numpy as np
    feat_path = os.path.join(data_dir, type + 'ing_data/feat/')

    feat = {}
    for name in glob.glob(os.path.join(feat_path, '*.npy')):
        feat[os.path.splitext(os.path.basename(name))[0]] = np.load(name)

    sorted_feat = sorted(feat.items())
    idx = [None] * len(sorted_feat)
    feat_vec = [None] * len(sorted_feat)
    for i, feat in enumerate(sorted_feat):
        idx[i] = feat[0]
        feat_vec[i] = feat[1]

    # Normalize Input Vector
    # mean = np.mean(feat_vec)
    # std = np.std(feat_vec)
    # normalized_feat = (feat_vec - mean) / std

    normalized_feat = feat_vec
    return normalized_feat, idx

def get_label(data_dir='MLDS_hw2_data/', type='train'):
    import os
    import json
    label_path = os.path.join(data_dir, type + 'ing_label.json')
    with open(label_path) as data_file:
        label = json.load(data_file)

    sorted_label = sorted(label, key=lambda k: k['id'])
    for label in sorted_label:
        label['caption'] = list(map(lambda x: x + '<eos>', label['caption']))
    return sorted_label

def load_data(data_dir='MLDS_hw2_data/', type='train', include='train_data', vocab_size=3000):
    import numpy as np
    import pickle
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences

    if type == 'train':
        normalized_feat, idx = get_feat(data_dir, 'train')
        sorted_label = get_label(data_dir, 'train')
        all_captions = ['<bos>' + item for label in sorted_label for item in label['caption']]

        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(all_captions)
        tokenized_captions = [tokenizer.texts_to_sequences(label['caption']) for label in sorted_label]
        tokenizer.word_index['pad'] = 0
        idx_to_word = {v: k for k, v in tokenizer.word_index.items()}

        with open('word_to_idx.pickle', 'wb') as handle:
            pickle.dump(tokenizer.word_index, handle)
        with open('idx_to_word.pickle', 'wb') as handle:
            pickle.dump(idx_to_word, handle)

        #padded_captions = [pad_sequences(caption, padding='post', maxlen=44) for caption in tokenized_captions]
        np.save('y_' + type + '.npy', tokenized_captions)
        np.save('idx_' + type + '.npy', idx)
        np.save('x_' + type + '.npy', normalized_feat)

        if include == 'test_data':
            normalized_feat, idx = get_feat(data_dir, 'test')
            sorted_label = get_label(data_dir, 'test')
            tokenized_captions = [tokenizer.texts_to_sequences(label['caption']) for label in sorted_label]
            #padded_captions = [pad_sequences(caption, padding='post', maxlen=44) for caption in tokenized_captions]
            np.save('y_test.npy', tokenized_captions)
            np.save('idx_test.npy', idx)
            np.save('x_test.npy', normalized_feat)
    else:
        np.save('idx_' + type + '.npy', idx)
        np.save('x_' + type + '.npy', normalized_feat)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./MLDS_hw2_data/')
    parser.add_argument('-mode', default='train')
    parser.add_argument('-include', default='train_data')

    args = parser.parse_args()
    print(args)

    load_data(data_dir=args.data_dir, type=args.mode, include=args.include)
