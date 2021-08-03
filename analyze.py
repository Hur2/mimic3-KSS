# visualisation tools for mimic2 
import matplotlib.pyplot as plt
from statistics import stdev, mode, mean, median
from statistics import StatisticsError
import argparse
import glob
import os
import csv
import copy
import seaborn as sns
import random
from text.cmudict import CMUDict
from text.korean import Korean

def get_audio_seconds(frames):
    return (frames*12.5)/1000


def append_data_statistics(meta_data):
    # get data statistics
    for char_cnt in meta_data:
        data = meta_data[char_cnt]["data"]
        audio_len_list = [d["audio_len"] for d in data]
        mean_audio_len = mean(audio_len_list)
        try:
            mode_audio_list = [round(d["audio_len"], 2) for d in data]
            mode_audio_len = mode(mode_audio_list)
        except StatisticsError:
            mode_audio_len = audio_len_list[0]
        median_audio_len = median(audio_len_list)

        try:
            std = stdev(
                d["audio_len"] for d in data
            )
        except:
            std = 0

        meta_data[char_cnt]["mean"] = mean_audio_len
        meta_data[char_cnt]["median"] = median_audio_len
        meta_data[char_cnt]["mode"] = mode_audio_len
        meta_data[char_cnt]["std"] = std
    return meta_data


def process_meta_data(path):
    meta_data = {}
    # load meta data
    with open(path, 'r', encoding='UTF8') as f:
        data = csv.reader(f, delimiter='|')
        for row in data:
            frames = int(row[2])
            utt = row[3]
            audio_len = get_audio_seconds(frames)
            char_count = len(utt)
            if not meta_data.get(char_count):
                meta_data[char_count] = {
                    "data": []
                }

            meta_data[char_count]["data"].append(
                {
                    "utt": utt,
                    "frames": frames,
                    "audio_len": audio_len,
                    "row": "{}|{}|{}|{}".format(row[0], row[1], row[2], row[3])
                }
            )
    meta_data = append_data_statistics(meta_data)

    return meta_data


def get_data_points(meta_data):
    x = [char_cnt for char_cnt in meta_data]
    y_avg = [meta_data[d]['mean'] for d in meta_data]
    y_mode = [meta_data[d]['mode'] for d in meta_data]
    y_median = [meta_data[d]['median'] for d in meta_data]
    y_std = [meta_data[d]['std'] for d in meta_data]
    y_num_samples = [len(meta_data[d]['data']) for d in meta_data]
    return {
        "x": x,
        "y_avg": y_avg,
        "y_mode": y_mode,
        "y_median": y_median,
        "y_std": y_std,
        "y_num_samples": y_num_samples
    }


def save_training(file_path, meta_data):
    rows = []
    for char_cnt in meta_data:
        data = meta_data[char_cnt]['data']
        for d in data:
            rows.append(d['row'] + "\n")

    random.shuffle(rows)
    with open(file_path, 'w+') as f:
        for row in rows:
            f.write(row)


def plot(meta_data, save_path=None):
    save = False
    if save_path:
        save = True

    graph_data = get_data_points(meta_data)
    x = graph_data['x']
    y_avg = graph_data['y_avg']
    y_std = graph_data['y_std']
    y_mode = graph_data['y_mode']
    y_median = graph_data['y_median']
    y_num_samples = graph_data['y_num_samples']
   
    plt.figure()
    plt.plot(x, y_avg, 'ro')
    plt.xlabel("character lengths", fontsize=30)
    plt.ylabel("avg seconds", fontsize=30)

    if save:
        name = "char_len_vs_avg_secs"
        plt.savefig(os.path.join(save_path, name))
    
    plt.figure()
    plt.plot(x, y_mode, 'ro')
    plt.xlabel("character lengths", fontsize=30)
    plt.ylabel("mode seconds", fontsize=30)
    if save:
        name = "char_len_vs_mode_secs"
        plt.savefig(os.path.join(save_path, name))

    plt.figure()
    plt.plot(x, y_median, 'ro')
    plt.xlabel("character lengths", fontsize=30)
    plt.ylabel("median seconds", fontsize=30)
    if save:
        name = "char_len_vs_med_secs"
        plt.savefig(os.path.join(save_path, name))

    plt.figure()
    plt.plot(x, y_std, 'ro')
    plt.xlabel("character lengths", fontsize=30)
    plt.ylabel("standard deviation", fontsize=30)
    if save:
        name = "char_len_vs_std"
        plt.savefig(os.path.join(save_path, name))

    plt.figure()
    plt.plot(x, y_num_samples, 'ro')
    plt.xlabel("character lengths", fontsize=30)
    plt.ylabel("number of samples", fontsize=30)
    if save:
        name = "char_len_vs_num_samples"
        plt.savefig(os.path.join(save_path, name))


def plot_phonemes(train_path, cmu_dict_path, save_path):
    cmudict = CMUDict(cmu_dict_path)

    phonemes = {}

    with open(train_path, 'r', encoding='UTF8') as f:
        data = csv.reader(f, delimiter='|')
        phonemes["None"] = 0
        for row in data:
            words = row[3].split()
            for word in words:
                word2 = Korean(word)
                for letter in word2:
                    pho=[]
                    try:
                        pho.append(letter.phoneme_onset)
                    except:
                        pass
                    try:
                        pho.append(letter.phoneme_nucleus)
                    except:
                        pass
                    try:
                        pho.append(letter.phoneme_coda)
                    except:
                        pass
                    if pho:
                        for nemes in pho:
                            if phonemes.get(nemes):
                                phonemes[nemes] += 1
                            else:
                                phonemes[nemes] = 1
    x, y = [], []
    for key in phonemes:
        x.append(key)
        y.append(phonemes[key])
    
    plt.figure()
    plt.rcParams["figure.figsize"] = (50, 20)
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plot = sns.barplot(x, y)
    if save_path:
        fig = plot.get_figure()
        fig.savefig(os.path.join(save_path, "phoneme_dist"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_file_path', required=True,
        help='this is the path to the train.txt file that the preprocess.py script creates'
    )
    parser.add_argument(
        '--save_to', help='path to save charts of data to'
    )
    parser.add_argument(
        '--cmu_dict_path', help='give cmudict-0.7b to see phoneme distribution' 
    )
    args = parser.parse_args()
    meta_data = process_meta_data(args.train_file_path)
    plt.rcParams["figure.figsize"] = (10, 5)
    plot(meta_data, save_path=args.save_to)
    if args.cmu_dict_path:
        plt.rcParams["figure.figsize"] = (30, 10)
        plot_phonemes(args.train_file_path, args.cmu_dict_path, args.save_to)
    plt.show()

if __name__ == '__main__':
    main()