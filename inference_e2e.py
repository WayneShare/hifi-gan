from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import Generator
from datetime import datetime



config = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    print(f'init the process...')
    generator = Generator(config).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_mels_dir)
    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    print(f'init ready =============')

    with torch.no_grad():
        for i, filname in enumerate(filelist):
            x = np.load(os.path.join(a.input_mels_dir, filname))
            print(f'open a mel file in npy format ')
            # print(x.dtype)

            # # whole to process
            # # whole to process
            print(f'process as whole ==========================================================')
            start=datetime.now()
            xH = torch.FloatTensor(x).to(device)
            y_g_hat = generator(xH)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            print(f'mel shape  = {x.shape} result shape {y_g_hat.shape} audio {audio.shape} time used {datetime.now()-start}')

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated_e2e.wav')
            write(output_file, config.sampling_rate, audio)
            print(output_file)


            # split to process
            # split to process
            print(f'process as splits ==========================================================')
            newarr = np.array_split(x, 50, axis=2) # split into 10 part by frames of axis=2

            all_array = None
            # print(newarr.shape)
            for part in newarr:
                start=datetime.now()
                x = torch.FloatTensor(part).to(device)
                y_g_hat = generator(x)
                audio = y_g_hat.squeeze()
                audio = audio * MAX_WAV_VALUE
                audio = audio.cpu().numpy().astype('int16')
                print(f'split part = {part.shape} result shape {y_g_hat.shape} audio {audio.shape} time used {datetime.now()-start}')
                # 
                if all_array is not None:
                    all_array = np.concatenate((all_array, audio))
                else :
                    all_array = audio

            print(f'merged whole audio ===== ')
            print(all_array.shape)

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated_e2e_M.wav')
            write(output_file, config.sampling_rate, all_array)
            print(output_file)


def main():
    print(f'Starting...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='test_mel_files')
    parser.add_argument('--output_dir', default='generated_files_from_mel')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global config
    json_config = json.loads(data)
    config = AttrDict(json_config)

    torch.manual_seed(config.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

