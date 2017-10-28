def sort_label(path):
    import os
    import pandas as pd
    import numpy as np
    import utils
    
    """Load TIMIT data"""
    labels_path = os.path.join(path,'label/train.lab')
    print("loading labels...")
    
    train_label = pd.read_table(labels_path, sep='_|,',header=None, engine='python')
    train_label = train_label.values

    """Manually Calculated T.T"""
    new_train_label = train_label[366664:406634 + 1].tolist() + train_label[784613:820516 + 1].tolist()
    new_train_label = new_train_label + train_label[855733:1124822 + 1].tolist() + train_label[0:366663 + 1].tolist()
    new_train_label = new_train_label + train_label[406635:784612 + 1].tolist() + train_label[820517:855732 + 1].tolist()

    new_train_label = np.array(new_train_label)
    
    """Convert from 48 Phones to 39 Phones"""
    print("converting from 48 to 39 phones...")
    map_path = os.path.join(path, 'phones/48_39.map')
    map48_39 = {}
    with open(map_path) as f:
        for line in f:
           (key, val) = line.split()
           map48_39[key] = val
    utils.phone_map(new_train_label, map48_39)
    
    assert(new_train_label.shape == train_label.shape)
    """Export to CSV"""
    print("labels sorted...")
    
    return new_train_label

        
def load_timit(path, kind='train'):
    import os
    import pandas as pd
    import numpy as np
    import utils
    from sklearn.preprocessing import LabelBinarizer
   
    print("=====loading " + kind + 'ing data=====')
    labels = np.zeros((1124823, 1))
    sequence_labels = np.zeros((3696, 1))
    
    """Load TIMIT data from `path`"""
    if kind == 'train':
        labels = sort_label(path)
        labels = np.delete(labels, (2), axis=1)
        labels = labels.astype('object')
        sequence_labels, label_ids = utils.frame_to_seq(labels)

    mfcc_path = os.path.join(path,'mfcc/%s.ark'% kind)
    frame_data = pd.read_table(mfcc_path, delimiter='_| ', header=None, engine='python')
    frame_data = frame_data.values
    frame_data = np.delete(frame_data, (2), axis=1)
            
    print("preprocessing data...")

    sequence_data, data_ids = utils.frame_to_seq(frame_data)
    
    assert(len(sequence_data) == len(data_ids))

    """Normalize MFCC"""
    normalized_sequence = [utils.normalize_mfcc(s) for s in sequence_data]
    
    """More Processes (Frame Context? Padded Sequences?"""
    
    print("loading complete...")
    
    return np.array(normalized_sequence), np.array(sequence_labels), np.array(data_ids)

