def seq_to_phones(path, csv_name, label, ids):
    import os
    import pandas as pd
    import numpy as np
    import itertools
    import utils
    
    """Convert from Phones to their English Letter"""
    map_path = os.path.join(path, 'phones/48phone_char.map')
    phone_to_char = {}
    with open(map_path) as f:
        for line in f:
           l = line.split()
           phone_to_char[l[0]] = l[2]

    history = [phone_mapping1(s, phone_to_char) for s in label]
    print("mapping to alphabet...")
    label = remove_duplicates(label)
    
    ids = ids.reshape(ids.shape[0], 1)
    label = label.reshape(label.shape[0], 1)
    
    finished_label = np.append(ids, label, axis=1)
    
    print("exporting to CSV...")
    finished_label = pd.DataFrame(finished_label).to_csv(
        os.path.join(path, csv_name), index=False, header=['id','phone_sequence'] )
    print("===========Finished============")

def trim_sil(s):
    if s.endswith("L"): s = s[:-1]
    if s.startswith("L"): s = s[1:]
    return s

def phone_mapping1(phones, mapping):
    for a in range(len(phones)):
        if phones[a] in mapping:
            phones[a] = mapping[phones[a]]
            
def remove_duplicates(result):
    import numpy as np
    import itertools
    """Remove Consecutive Duplicates"""
    output = []
    print("removing duplicates...")
    for s in result:
        s = ''.join(i for i, _ in itertools.groupby(s))
        """Trimming Front and End Sils"""
        s = trim_sil(s)
        output.append(s)
    return np.array(output)