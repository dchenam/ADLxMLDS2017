def phone_map(phones, mapping):
    for a in phones:
        if a[3] in mapping:
            a[3]= mapping[a[3]]
            
def frame_to_seq(frames):
    import numpy as np
    sequence_list = []
    ids = []
    sequence = frames[0][2:]
    start_frame = frames[0][1]
    for i in range(frames.shape[0]):
        if frames[i][1] == start_frame:
            sequence = np.vstack((sequence, frames[i][2:]))
        else:
            ids.append(frames[i - 1][0] + '_' + start_frame)
            start_frame = frames[i][1]
            if sequence.any():
                sequence_list.append(sequence)
                sequence = frames[i][2:]
    ids.append(frames[i - 1][0] + '_' + start_frame)
    sequence_list.append(sequence)
    return sequence_list, ids

def normalize_mfcc(mfcc):
    import numpy as np
    mfcc = mfcc.astype('float')
    mean = np.mean(mfcc, 0)
    std = np.std(mfcc, 0)
    return ((mfcc - mean) / std)

def pad_last(sentence, target_length):
    import numpy as np
    last = sentence[-1]
    pad = [last for i in range(0, target_length - len(sentence))]
    return np.append(sentence, pad, 0)

def pad_to_sequence_length(sentence, seq_length):
    padding_length = seq_length - (sentence.shape[0] % seq_length)
    target_length = sentence.shape[0] + padding_length
    return pad_last(sentence, target_length)