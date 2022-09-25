from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator
# 
from datetime import datetime
from numpy import asarray, save, savetxt
import matplotlib.pyplot as plt
# import glob
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt



config = None
device = None
print('enter file ...')

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("load_checkpoint Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, config.n_fft, config.num_mels, config.sampling_rate, config.hop_size, config.win_size, config.fmin, config.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inferOneFile(argsAll, generator, filname):
    start=datetime.now()

    # 1 ####################################
    print("start read wav - ================================")
    wav, sr = load_wav(os.path.join(argsAll.input_wavs_dir, filname))
    wav = wav / MAX_WAV_VALUE
    wav = torch.FloatTensor(wav).to(device)
    print(f"wav before mel - {wav.dtype} {wav.shape}")
    
    # 2 ####################################
    x = get_mel(wav.unsqueeze(0))
    print(f"mel result - {x.dtype} {x.shape}")
    # save mel into a npy file
    output_file = os.path.join(argsAll.output_dir, os.path.splitext(filname)[0] + '_mel.npy')
    save(output_file, x.cpu().numpy())
    print(f"save mel - {output_file} time used {datetime.now()-start}")
            
    print(f'process as whole ================================')
    # whole inference ######################################################################
    # whole inference ######################################################################

    # start=datetime.now()
    # y_g_hat = generator(x)
    # audio = y_g_hat.squeeze()
    # audio = audio * MAX_WAV_VALUE
    # audio = audio.cpu().numpy().astype('int16')
    # print(f'mel shape  = {x.shape} result shape {y_g_hat.shape} audio {audio.shape} time used {datetime.now()-start}')
    # start=datetime.now()
    # # 
    # # save result wav file
    # output_file = os.path.join(argsAll.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
    # write(output_file, config.sampling_rate, audio)
    # print(output_file)

    # split to process ####################################################################
    # split to process ####################################################################
    print(f'process as splits ===============================')
    all_array = None
    slices = 20
    # 
    overlap_frames = 16
    resultsize_perframe = 256
    overlap = overlap_frames * resultsize_perframe
    # for indexA in range(slices):
    all_array = None
    for index in range(slices):
        # ////////////////////////////////////
        # get start/end of this slice ////////
        step = int(x.shape[2]/slices)
        star = index*(step)-overlap_frames
        if star<0:
            star=0
        end = (index+1)*(step)
        if index==slices-1 or end > x.shape[2]:
            end = x.shape[2]
        print(f'{index} {star} {end}')
        part=x[:,:,star:end]

        # ////////////////////////////////////
        # process this  slice ////////////////
        start=datetime.now()
        x_tem = torch.FloatTensor(part).to(device)
        y_g_hat = generator(x_tem)

        # ////////////////////////////////////
        # transfer result to audio ///////////
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
        # remove overlap in audio 
        # print(f' len(audio) ={len(audio)} len(audio)-overlap/2 = {len(audio)-int(overlap/2)} ')
        if star==0:
            audio = audio[0:(len(audio)-int(overlap/2))]
        if star>0 and end == x.shape[2]:
            audio = audio[int(overlap/2):len(audio)]
        if star>0 and end < x.shape[2]:
            audio = audio[int(overlap/2):(len(audio)-int(overlap/2))]

        # ////////////////////////////////////
        # merge audio into result ////////////
        print(f'split part = {part.shape} result shape {y_g_hat.shape} audio {audio.shape} time used {datetime.now()-start}')
        if all_array is not None:
            all_array = np.concatenate((all_array, audio))
        else :
            all_array = audio

    # save audio into file 
    # output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated_' + str(indexA) + '.wav')
    output_file = os.path.join(argsAll.output_dir, os.path.splitext(filname)[0] + '_generated_O_'+ str(overlap_frames) + '.wav')
    write(output_file, config.sampling_rate, all_array)
    print(f'end splits =======================================')
    print(output_file)
    
    # end of split to process ####################################################################
    return


def inference(argsAll):

    generator = Generator(config).to(device)

    state_dict_g = load_checkpoint(argsAll.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(argsAll.input_wavs_dir)

    os.makedirs(argsAll.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            return inferOneFile(argsAll, generator, filname)


def main():

    print('Start process arguments...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    argAll = parser.parse_args()

    config_file = os.path.join(os.path.split(argAll.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global config
    json_config = json.loads(data)
    config = AttrDict(json_config)

    torch.manual_seed(config.seed)
    global device
    # if torch.cuda.is_available():
    #     print('cuda.is_available() true')
    #     torch.cuda.manual_seed(config.seed)
    #     device = torch.device('cuda')
    # else:
    #     print('cuda.is_available() false')
    #     device = torch.device('cpu')
    device = torch.device('cpu')

    print('Start inference...')

    inference(argAll)


if __name__ == '__main__':
    print('Start ...')
    main()







