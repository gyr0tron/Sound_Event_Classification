import numpy as np
import pandas as pd
import librosa

# from keras.models import Sequential
# from keras.layers import Dense
from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.model_selection import RepeatedKFold
# from sklearn.metrics import accuracy_score

from tqdm import tqdm
import os

DEV_ROOT = './Data/FSD50K.dev_audio/'
TEST_ROOT = './Data/FSD50K.eval_audio/'

raw_dev_df = pd.read_csv('./Data/FSD50K.ground_truth/dev.csv', dtype={'fname': 'string'})

test_df = pd.read_csv('./Data/FSD50K.ground_truth/eval.csv', dtype={'fname': 'string'})
vocab_df = pd.read_csv('./Data/FSD50K.ground_truth/vocabulary.csv', index_col=0, header=None, names=['label','mids'])
vocab_dict = dict(zip(vocab_df.mids, vocab_df.label))

raw_dev_df['mids'] = raw_dev_df['mids'].apply(lambda x: tuple(x.split(',')))
test_df['mids'] = test_df['mids'].apply(lambda x: tuple(x.split(',')))

one_hot_encoder = MultiLabelBinarizer()
one_hot_encoder.fit(raw_dev_df['mids'])
output_vec = one_hot_encoder.transform(raw_dev_df['mids'])
raw_dev_df['output_vec'] = output_vec.tolist()
output_vec = one_hot_encoder.transform(test_df['mids'])
test_df['output_vec'] = output_vec.tolist()

train_df = raw_dev_df[raw_dev_df['split'] == 'train']
val_df = raw_dev_df[raw_dev_df['split'] == 'val']

def replicate(data, min_clip_len):
  if len(data) < min_clip_len:
    tile_size = (min_clip_len // data.shape[0]) + 1
    data = np.tile(data, tile_size)[:min_clip_len]
  return data

def extract_features(fname_list,output_vec_list,DIR,bands=60,frames=41):
    def _windows(data, window_size):
        start = 0
        while start < len(data):
            yield int(start), int(start + window_size)
            start += (window_size // 2) # 50% overlap
            
    window_size = 512 * (frames - 1)
    # min_clip_len = int(22050 * 1)
    min_clip_len = window_size
    features, labels = [], []
    for idx,file_name in enumerate(tqdm(fname_list)):
        fn = DIR + file_name + '.wav'
        segment_log_specgrams, segment_labels = [], []
        sound_clip,sr = librosa.load(fn)
        label = output_vec_list[idx]
        if len(sound_clip) < min_clip_len:
            sound_clip = replicate(sound_clip, min_clip_len)
        for (start,end) in _windows(sound_clip,window_size):
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                melspec = librosa.feature.melspectrogram(y=signal,n_mels=bands)
                logspec = librosa.amplitude_to_db(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                segment_log_specgrams.append(logspec)
                segment_labels.append(label)
            
        segment_log_specgrams = np.asarray(segment_log_specgrams).reshape(
            len(segment_log_specgrams),bands,frames,1)
        segment_features = np.concatenate((segment_log_specgrams, np.zeros(
            np.shape(segment_log_specgrams))), axis=3)
        for i in range(len(segment_features)): 
            segment_features[i, :, :, 1] = librosa.feature.delta(
                segment_features[i, :, :, 0])
        
        if len(segment_features) > 0: # check for empty segments 
            features.append(segment_features)
            labels.append(segment_labels)
    return features, labels

STORE_DIR = './Data/Checkpoints/'

features, labels = extract_features(train_df['fname'].to_list(), train_df['output_vec'].to_list(), DEV_ROOT)
print('='*20 + 'Train Features Generated!' + '='*20)
np.savez_compressed(STORE_DIR+'train', features=np.asarray(features,dtype=object), labels=np.asarray(labels,dtype=object))
print('='*20 + 'Train npz saved!' + '='*20)

features, labels = extract_features(val_df['fname'].to_list(), val_df['output_vec'].to_list(), DEV_ROOT)
print('='*20 + 'Valid Features Generated!' + '='*20)
np.savez_compressed(STORE_DIR+'val', features=np.asarray(features,dtype=object), labels=np.asarray(labels,dtype=object))
print('='*20 + 'Valid npz saved!' + '='*20)

features, labels = extract_features(test_df['fname'].to_list(), test_df['output_vec'].to_list(), TEST_ROOT)
print('='*20 + 'Test Features Generated!' + '='*20)
np.savez_compressed(STORE_DIR+'test', features=np.asarray(features,dtype=object), labels=np.asarray(labels,dtype=object))
print('='*20 + 'Test npz saved!' + '='*20)