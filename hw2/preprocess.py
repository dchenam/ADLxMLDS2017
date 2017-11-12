
def load_data(data_dir='MLDS_hw2_data/', type='train', vocab_size=3000):
    import numpy as np
    import json
    import glob
    import os
    import pickle
    from keras.preprocessing.text import Tokenizer, one_hot
    from keras.preprocessing.sequence import pad_sequences

    feat_path = os.path.join(data_dir, type + 'ing_data/feat/')
    label_path = os.path.join(data_dir, type+ 'ing_label.json')

    feat = {}
    for name in glob.glob(os.path.join(feat_path,'*.npy')):
        feat[os.path.splitext(os.path.basename(name))[0]] = np.load(name)

    sorted_feat = sorted(feat.items())
    idx = [None] * len(sorted_feat)
    feat_vec = [None] * len(sorted_feat)
    for i, feat in enumerate(sorted_feat):
        idx[i] = feat[0]
        feat_vec[i] = feat[1]

    # Normalize Input Vector
    mean = np.mean(feat_vec)
    std = np.std(feat_vec)
    normalized_feat = (feat_vec - mean) / std

    if type == 'train':
        with open(label_path) as data_file:
            label = json.load(data_file)

        sorted_label = sorted(label, key=lambda k: k['id'])
        for label in sorted_label:
            label['caption'] = list(map(lambda x: x + '<eos>', label['caption']))

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

        padded_captions = [pad_sequences(caption, padding='post', maxlen=44) for caption in tokenized_captions]

        np.save('y_' + type + '.npy', padded_captions)
    np.save('idx_' + type + '.npy', idx)
    np.save('x_' + type + '.npy', normalized_feat)


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./MLDS_hw2_data/')
    parser.add_argument('--train_or_test', default='train')

    args = parser.parse_args()
    print(args)

    load_data(data_dir=args.data_dir, type=args.train_or_test)
