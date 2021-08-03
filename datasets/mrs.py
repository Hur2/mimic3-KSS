from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from util import audio
import sqlite3
import sys


def build_from_path(in_dir, out_dir, username, num_workers=1, tqdm=lambda x: x):
  '''Preprocesses the recordings from mimic-recording-studio (based on sqlite db) into a given output directory.

    Args:
      in_dir: The root directory of mimic-recording-studio
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
  '''

  # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  index = 1

  # Query sqlite db of mimic-recording-studio
  dbfile = os.path.join(in_dir,"backend","db","mimicstudio.db")
  print("Reading data from mimic-recording-studio sqlite db file: " + dbfile)
  conn = sqlite3.connect(dbfile)
  c = conn.cursor()

  uid = ''
  sql_get_guid = "SELECT uuid FROM usermodel LIMIT 1;"
  if username:
    print("Query user guid for " + username + " in sqlite db")
    sql_get_guid = "SELECT uuid FROM usermodel WHERE UPPER(user_name) = '" + username.upper() + "' LIMIT 1;"

  for row in c.execute(sql_get_guid):
    uid = row[0]
  
  if uid == '':
    print("No userid could be found in sqlite db.")
    sys.exit()
  
  print("Found speaker user guid in sqlite: " + uid)

  wav_dir = os.path.join(in_dir,"backend","audio_files",uid)
  print("Search for wav files in " + wav_dir)
  
  for row in c.execute('SELECT DISTINCT audio_id, lower(prompt) FROM audiomodel ORDER BY length(prompt)'):
    wav_path = os.path.join(wav_dir, '%s.wav' % row[0])
    if os.path.isfile(wav_path):
      text = row[1]
      futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, text)))
      index += 1
    else:
      print("File " + wav_path + " no found. Skipping.")
  return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text):
  '''Preprocesses a single utterance audio/text pair.

  This writes the mel and linear scale spectrograms to disk and returns a tuple to write
  to the train.txt file.

  Args:
    out_dir: The directory to write the spectrograms into
    index: The numeric index to use in the spectrogram filenames.
    wav_path: Path to the audio file containing the speech input
    text: The text spoken in the input audio file

  Returns:
    A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
  '''

  # Load the audio to a numpy array:
  wav = audio.load_wav(wav_path)

  # Compute the linear-scale spectrogram from the wav:
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]

  # Compute a mel-scale spectrogram from the wav:
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

  # Write the spectrograms to disk:
  spectrogram_filename = 'mrs-spec-%05d.npy' % index
  mel_filename = 'mrs-mel-%05d.npy' % index
  np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

  # Return a tuple describing this training example:
  return (spectrogram_filename, mel_filename, n_frames, text)
