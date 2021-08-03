# This script generates a folder structure for ljspeech-1.1 processing from mimic-recording-studio database
# Written by Thorsten Mueller (MrThorstenM@gmx.net) on november 2019 without any warranty

import argparse
import sqlite3
import os
from shutil import copyfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default=os.path.expanduser('~/tacotron'))
    parser.add_argument('--mrs_dir', required=True)
    args = parser.parse_args()

    dir_base_ljspeech = os.path.join(args.out_dir,"LJSpeech-1.1")
    dir_base_ljspeech_wav = os.path.join(dir_base_ljspeech,"wavs")
    dir_base_mrs = args.mrs_dir
    os.makedirs(dir_base_ljspeech_wav, exist_ok=True)

    conn = sqlite3.connect(os.path.join(dir_base_mrs,"backend","db","mimicstudio.db"))
    c = conn.cursor()

    # Get user id from sqlite to find recordings in directory structure
    # TODO: Currently just works with one user
    for row in c.execute('SELECT uuid FROM usermodel LIMIT 1;'):
        uid = row[0]
  
    print("Found speaker user guid in sqlite: " + uid)

    # Create new metadata.csv for ljspeech
    metadata = open(os.path.join(dir_base_ljspeech,"metadata.csv"),mode="w", encoding="utf8") 

    for row in c.execute('SELECT DISTINCT audio_id, prompt, lower(prompt) FROM audiomodel ORDER BY length(prompt)'):
        audio_file_source = os.path.join(dir_base_mrs,"backend","audio_files", uid, row[0] + ".wav")
        if os.path.isfile(audio_file_source):
            metadata.write(row[0] + "|" + row[1] + "|" + row[2] + "\n")
            audio_file_dest = os.path.join(dir_base_ljspeech_wav,row[0] + ".wav")
            copyfile(audio_file_source,audio_file_dest)
        else:
            print("File " + audio_file_source + " no found. Skipping.")


    metadata.close()
    conn.close()

if __name__ == '__main__':
    main()