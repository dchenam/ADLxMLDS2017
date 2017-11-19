def get_feat(data_dir='MLDS_hw2_data/', mode ='training_data'):
    import glob
    import os
    import numpy as np
    feat_path = os.path.join(data_dir, mode +'/feat/')
    feat_dict = {}
    for name in glob.glob(os.path.join(feat_path, '*.npy')):
        feat_dict[os.path.splitext(os.path.basename(name))[0]] = np.load(name)
    return feat_dict

def get_label(data_dir='MLDS_hw2_data/', mode='training_label'):
    import os
    import json
    label_path = os.path.join(data_dir, mode + '.json')
    with open(label_path) as data_file:
        label_dict = json.load(data_file)
    return label_dict

def normalize_feat(feat_data):
    import numpy as np
    mean = np.mean(feat_data)
    std = np.std(feat_data)
    return (feat_data - mean) / std

def dict_to_data(feat_dict, label_dict):
    video_id = [None] * len(feat_dict)
    feat_data = [None] * len(feat_dict)
    label_data = [None] * len(feat_dict)
    for label in label_dict:
        label['caption'] = list(map(lambda x: x + '<eos>', label['caption']))
    for i in range(len(feat_dict)):
        label_data[i] = label_dict[i]['caption']
        key = label_dict[i]['id']
        feat_data[i] = feat_dict[key]
        video_id[i] = key
    return video_id, feat_data, label_data

def process_data(data_dir='MLDS_hw2_data/', mode='train'):
    import numpy as np
    import os
    if mode == 'train':
        feat_dict = get_feat(data_dir, 'training_data')
        label_dict = get_label(data_dir, 'training_label')
        video_id, feat_data, label_data = dict_to_data(feat_dict, label_dict)
        normalize_feat(feat_data)
    elif mode == 'test':
        feat_dict = get_feat(data_dir, 'testing_data')
        label_dict = get_label(data_dir, 'testing_label')
        video_id, feat_data, label_data = dict_to_data(feat_dict, label_dict)
        normalize_feat(feat_data)
    elif mode == 'peer_review':
        feat_dict = get_feat(data_dir, 'peer_review')
        id_list_path = os.path.join(data_dir, 'peer_review_id.txt')
        id_list = []
        video_id = [None] * len(feat_dict)
        feat_data = [None] * len(feat_dict)
        with open(id_list_path, 'r') as f:
            id_list = f.readlines()
        for i, j in enumerate(id_list):
            key = j.strip('\n')
            video_id[i] = key
            feat_data[i] = feat_dict[key]
        normalize_feat(feat_data)
        return video_id, feat_data
    return video_id, feat_data, label_data


def load_data(data_dir='MLDS_hw2_data/', mode='train', include='test', vocab_size=3000):
    import numpy as np
    import pickle
    from keras.preprocessing.text import Tokenizer

    if mode == 'train':
        video_id, feat_data, label_data = process_data(mode='train')
        flattened_labels = ['<bos>' + item for label in label_data for item in label]
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(flattened_labels)
        tokenizer.word_index['pad'] = 0
        label_data = [tokenizer.texts_to_sequences(label) for label in label_data]

        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer.word_index, handle)
        np.save('id_' + mode + '.npy', video_id)
        np.save('x_' + mode + '.npy', feat_data)
        np.save('y_' + mode + '.npy', label_data)
        if include == 'test':
            video_id, feat_data, label_data = process_data(mode='test')
            label_data = [tokenizer.texts_to_sequences(label) for label in label_data]
            np.save('id_' + include + '.npy', video_id)
            np.save('x_' + include + '.npy', feat_data)
            np.save('y_' + include + '.npy', label_data)
    else:
        video_id, feat_data = process_data(mode='peer_review')
        np.save('id_' + mode + '.npy', video_id)
        np.save('x_' + mode + '.npy', feat_data)


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', default='./MLDS_hw2_data/')
    parser.add_argument('-mode', default='train')
    parser.add_argument('-include', default='test')
    args = parser.parse_args()
    print(args)
    load_data(data_dir=args.data_dir, mode=args.mode, include=args.include)