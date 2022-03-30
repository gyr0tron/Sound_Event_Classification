import numpy as np
import pandas as pd
import librosa

from multiprocessing import Pool
import soundfile as sf
import os



DEV_ROOT = './Data/FSD50K.dev_audio/'
TEST_ROOT = './Data/FSD50K.eval_audio/'

raw_dev_df = pd.read_csv('./Data/FSD50K.ground_truth/dev.csv', dtype={'fname': 'string'})

train_df = raw_dev_df[raw_dev_df['split'] == 'train'].drop(columns=['split']).copy()
val_df = raw_dev_df[raw_dev_df['split'] == 'val'].drop(columns=['split']).copy()
test_df = pd.read_csv('./Data/FSD50K.ground_truth/eval.csv', dtype={'fname': 'string'})
vocab_df = pd.read_csv('./Data/FSD50K.ground_truth/vocabulary.csv', index_col=0, header=None, names=['label','mids'])
vocab_dict = dict(zip(vocab_df.mids, vocab_df.label))

files_list = ['63','699']
lf = len(files_list)


def replicate(data, min_clip_len):
  if len(data) < min_clip_len:
    tile_size = (min_clip_len // data.shape[0]) + 1
    data = np.tile(data, tile_size)[:min_clip_len]
  return data

# Saving so that we only need to do this once
PARTS_DIR = './Data/Chunks/'
def process_idx(idx):
  # print("in pool: "+str(idx))
  # return idx
  file_name = files_list[idx]
  file_path = DEV_ROOT + file_name + '.wav'
  data, sample_rate = librosa.load(file_path)
  min_clip_len = int(sample_rate * 1)
  parts = []
  if len(data) < min_clip_len:
    data = replicate(data, min_clip_len)
    parts.append(data)
  else:
    overlap = int(sample_rate * 0.5) # 50% overlap
    for ix in range(0, len(data), overlap):
      clip_ix = data[ix:ix+min_clip_len]
      clip_ix = replicate(clip_ix, min_clip_len)
      parts.append(clip_ix)

  for i in range(len(parts)):
    path = os.path.join(PARTS_DIR, "{}_{:04d}.wav".format(file_name, i))
    sf.write(path, parts[i], sample_rate, "PCM_16")
  print("File: " + str(file_name) + '.wav ' + "Done!")
  return idx

if __name__ == '__main__':
  # Multithreaded processing to process multiple files at the same time
  pool = Pool(8)
  o = pool.map_async(process_idx, range(lf))
  res = o.get()
  pool.close()
  pool.join()

