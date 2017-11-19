def get_feat(data_dir='MLDS_hw2_data/', type='train'):
    import glob
    import os
    import numpy as np
    feat_path = os.path.join(data_dir, type + 'ing_data/feat/')

    feat_dict = {}
    for name in glob.glob(os.path.join(feat_path, '*.npy')):
        feat_dict[os.path.splitext(os.path.basename(name))[0]] = np.load(name)
    return feat_dict

def get_label(data_dir='MLDS_hw2_data/', type='train'):
    import os
    import json

    label_path = os.path.join(data_dir, type + 'ing_label.json')
    with open(label_path) as data_file:
        label_dict = json.load(data_file)
    for label in label_dict:
        label['caption'] = list(map(lambda x: x + '<eos>', label['caption']))
    return label_dict

def get_data(data_dir='MLDS_hw2_data/', type='train', id_list='none'):
    import numpy as np
    import os
    feat_dict = get_feat(data_dir, type)
    label_dict = get_label(data_dir, type)
    idx = [None] * len(feat_dict)
    sorted_feat = [None] * len(feat_dict)
    sorted_label = [None] * len(label_dict)
    for i in range(len(feat_dict)):
        sorted_label[i] = label_dict[i]['caption']
        key = label_dict[i]['id']
        sorted_feat[i] = feat_dict[key]
        idx[i] = key
        if id_list != 'none':
            id_path = os.path.join(data_dir, id_list)
            result = []
            with open(id_path, 'r') as f:
                result = f.readlines()
                for i, line in enumerate(result):
                    result[i] = line.strip('\n')
            selected_idx = [None] * len(result)
            selected_feat = [None] * len(result)
            selected_label = [None] * len(result)
            for i, j in enumerate(result):
                selected_feat[i] = sorted_feat[idx.index(j)]
                selected_label[i] = sorted_label[idx.index(j)]
                selected_idx[i] = j
            # Normalize Data
            mean = np.mean(selected_feat)
            std = np.std(selected_feat)
            selected_feat = (sorted_feat - mean) / std
            return selected_idx, selected_feat, selected_label
        # Normalize Data
        mean = np.mean(sorted_feat)
        std = np.std(sorted_feat)
        sorted_feat = (sorted_feat - mean) / std
    return idx, sorted_feat, sorted_label

def load_data(data_dir='MLDS_hw2_data/', type='train', include='train_data', id_list='none', vocab_size=3000):
    import numpy as np
    import pickle
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences

    if type == 'train':
        idx, normalized_feat, sorted_label= get_data(data_dir, 'train')

        all_captions = ['<bos>' + item for label in sorted_label for item in label]
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(all_captions)
        tokenized_captions = [tokenizer.texts_to_sequences(label) for label in sorted_label]
        tokenizer.word_index['pad'] = 0
        idx_to_word = {v: k for k, v in tokenizer.word_index.items()}

        with open('word_to_idx.pickle', 'wb') as handle:
            pickle.dump(tokenizer.word_index, handle)
        with open('idx_to_word.pickle', 'wb') as handle:
            pickle.dump(idx_to_word, handle)

        shifted_tokenized_captions = [[[2] + sequence for sequence in captions] for captions in tokenized_captions]
        padded_shift = [pad_sequences(caption, padding='post', maxlen=44) for caption in shifted_tokenized_captions]
        padded_captions = [pad_sequences(caption, padding='post', maxlen=44) for caption in tokenized_captions]

        np.save('y_shift_' + type + '.npy', padded_shift)
        np.save('y_' + type + '.npy', padded_captions)
        np.save('idx_' + type + '.npy', idx)
        np.save('x_' + type + '.npy', normalized_feat)

        if include == 'test_data':
            idx, normalized_feat, sorted_label = get_data(data_dir, 'test')

            tokenized_captions = [tokenizer.texts_to_sequences(label) for label in sorted_label]
            shifted_tokenized_captions = [[[2] + sequence for sequence in captions] for captions in
                                          tokenized_captions]
            padded_shift = [pad_sequences(caption, padding='post', maxlen=44) for caption in shifted_tokenized_captions]
            padded_captions = [pad_sequences(caption, padding='post', maxlen=44) for caption in tokenized_captions]

            np.save('y_shift_test.npy', padded_shift)
            np.save('y_test.npy', padded_captions)
            np.save('idx_test.npy', idx)
            np.save('x_test.npy', normalized_feat)
    else:
        idx, normalized_feat, sorted_label = get_data(data_dir, type, id_list)
        np.save('idx_' + type + '.npy', idx)
        np.save('x_' + type + '.npy', normalized_feat)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./MLDS_hw2_data/')
    parser.add_argument('-mode', default='train')
    parser.add_argument('-include', default='train_data')
    parser.add_argument('-use_list', default='none')
    args = parser.parse_args()
    print(args)

    load_data(data_dir=args.data_dir, type=args.mode, include=args.include, id_list=args.use_list)
