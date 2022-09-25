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


import sys
import wave
from tqdm import tqdm
import pyaudio
import time
# CHUNK = 1024



config = None
device = None
generator1 = None
generator2 = None
generator3 = None
generator4 = None

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

# /////////////////////////////////////////////////////////////////////////////////////////////
# thread related code /////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////
import queue
import threading
import time
from time import sleep, perf_counter
import random
from collections import OrderedDict

overlap_frames = 2
slices = 70
resultsize_perframe = 256
overlap = overlap_frames * resultsize_perframe


all_array = None
orderD = OrderedDict()

def getSlice(melData, slices, index):
    step = int( melData.shape[2]/(slices) ) + 1
    star = index*(step)-overlap_frames
    if star<0:
        star=0
    end = (index+1)*(step)
    if end > melData.shape[2]:
        end = melData.shape[2]
    print(f'{index} {star} {end}')
    return star, end

# thread for data producer - 生产者线程
class DataProducer(threading.Thread):
    def __init__(self, t_name, queue, melData):
        threading.Thread.__init__(self, name=t_name)
        self.queue = queue
        self.melData = melData

    def run(self):
        s111= perf_counter()
        
        for index in range(slices):
            # ////////////////////////////////////
            # get start/end of this slice ////////
            star, end = getSlice(self.melData, slices, index)
            part = self.melData[:,:,star:end]
            slices_inSecond = (end - star-overlap_frames)*config.hop_size/config.sampling_rate
            if slices_inSecond < 0:
                slices_inSecond=0
            time.sleep(slices_inSecond)
            # 
            print (f"Data producer {time.ctime()}: {self.getName()} is producing {index} - {star} : {end} to the queue! hop length is {slices_inSecond} second.")
            self.queue.put((index, part))  # 将生产的数据放入队列
            # time.sleep(random.randrange(10)/5)
        
        e111 = perf_counter()
        print (f"===== Data producer {time.ctime()}, {self.getName()}, It took {e111- s111: 0.2f} second(s) to produce the data.")
        return


# /////////////////////////////////////////////////////////////
# thread for data processing
def data_worker(q, melData, generator):
    while True:
        part = q.get()
        orderD[str(part[0])] = None
        print(f"process {part[0]} start - {part[1].dtype}  {part[1].shape}")

        # ////////////////////////////////////
        # process this slice /////////////////
        start = datetime.now()
        x_tem = torch.FloatTensor(part[1]).to(device)
        print(f"time used {datetime.now()-start}")
        y_g_hat = generator(x_tem)

        # ////////////////////////////////////
        # transfer result to audio ///////////
        print(f"time used {datetime.now()-start}")
        audio = y_g_hat.squeeze()
        print(f"processed audio {audio.dtype} {audio.shape} time used {datetime.now()-start}")

        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().detach().numpy().astype('int16')  # ???
        # remove overlap in audio 
        star, end = getSlice(melData, slices, part[0])
        # print(f' len(audio) ={len(audio)} len(audio)-overlap/2 = {len(audio)-int(overlap/2)} ')
        if star==0:
            audio = audio[0:(len(audio)-int(overlap/2))]
        if star>0 and end == melData.shape[2]:
            audio = audio[int(overlap/2):len(audio)]
        if star>0 and end < melData.shape[2]:
            audio = audio[int(overlap/2):(len(audio)-int(overlap/2))]

        # randN = random.randrange(1,10)
        # sleep(randN)
        print(f"process {part[0]} done")
        orderD[str(part[0])] = audio
        print(f'split part = {part[1].shape} result shape {y_g_hat.shape} audio {audio.shape} time used {datetime.now()-start}')
        q.task_done()

# thread result reader
def data_resultreader(od, stream):
    s222 = perf_counter()
    time.sleep(0.13)
    print ('----- Result reader - wait for the orderqueue to have data...')

    # # play stream (3)
    # for d in tqdm(datas):
    #     stream.write(d)

    while len(orderD) > 0:
        time.sleep(0.001)
        if list(orderD.values())[0] is not None:
            pair = orderD.popitem(last=False)
            audio = pair[1]
            print(f"read result {pair[0]} --- {len(audio)}")
            stream.write(audio)
            # time.sleep(0.02)
            # rt.append(od.popitem(last=False))
            # ////////////////////////////////////
            # merge audio into result ////////////
            global all_array
            if all_array is not None:
                all_array = np.concatenate((all_array, audio))
                e222 = perf_counter()
                print(f'----- Result reader - It took {e222 - s222: 0.2f} second(s) to complete.')
            else :
                all_array = audio

    # all data received now
    # stop stream (4)
    stream.stop_stream()
    stream.close()
    print('----- Result reader done')
    return

rt=[]
# def testThreads():    

#     # start queue producer //////////////////////////
#     que = queue.Queue()
#     producer = DataProducer('DataProducer', que)
#     # consumer = Consumer('Con.', queue)
#     producer.start()
#     # consumer.start()

#     # start queue processor  /////////////////////////
#     for i in range(3):
#          t = threading.Thread(target=data_worker,args=(que,))
#          t.daemon = True
#          t.start()

#     # # start queue processor  /////////////////////////
#     resultReader = threading.Thread(target=data_resultreader,args=(od,))
#     resultReader.start()

#     # join to wait the queue procesing ///////////////
#     que.join()  # # block until all tasks are done, 阻塞，直到生产者生产的数据全都被消费掉
#     producer.join() # 等待生产者线程结束
#     resultReader.join()
#     # consumer.join() # 等待消费者线程结束


#     # printe result //////////////////////////////////
#     print ('all done ////////////////////////////')
#     for key, value in od.items():
#         print(key, value)
#     print(rt)



# /////////////////////////////////////////////////////////////////////////////////////////////
# end thread related code /////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////

def slicesProcess(melData, stream):
    start_time = perf_counter()

    global orderD
    # start queue producer //////////////////////////
    que = queue.Queue()
    producer = DataProducer('DataProducer', que, melData)
    # consumer = Consumer('Con.', queue)
    producer.start()
    # consumer.start()

    # start queue processor  /////////////////////////
    # for i in range(1):
    #      t = threading.Thread(target=data_worker,args=(que, melData, generator))
    #      t.daemon = True
    #      t.start()
    global generator1, generator2, generator3, generator4
    t1 = threading.Thread(target=data_worker,args=(que, melData, generator1))
    t1.daemon = True
    t1.start()
    t2 = threading.Thread(target=data_worker,args=(que, melData, generator2))
    t2.daemon = True
    t2.start()
    t3 = threading.Thread(target=data_worker,args=(que, melData, generator3))
    t3.daemon = True
    t3.start()
    t4 = threading.Thread(target=data_worker,args=(que, melData, generator4))
    t4.daemon = True
    t4.start()

    # # start queue processor  /////////////////////////
    resultReader = threading.Thread(target=data_resultreader,args=(orderD, stream))
    resultReader.start()
    
    # print(f"read result ==================================")
    # while len(od) > 0:
    #     print ('wait for the orderqueue to have data ///////////////')
    #     time.sleep(0.3)
    #     if list(od.values())[0] is not None:
    #         pair = od.popitem(last=False)
    #         audio = pair[1]
    #         print(f"read result {pair[0]} --- {len(audio)}")
    #         # rt.append(od.popitem(last=False))
    #         # ////////////////////////////////////
    #         # merge audio into result ////////////
    #         global all_array
    #         if all_array is not None:
    #             all_array = np.concatenate((all_array, audio))
    #         else :
    #             all_array = audio
    
    # join to wait the queue procesing ///////////////
    
    
    producer.join() # 等待生产者线程结束
    resultReader.join()
    que.join()  # # block until all tasks are done, 阻塞，直到生产者生产的数据全都被消费掉
    
    # t1.join() # 等待worker线程结束
    # t2.join() # 等待worker线程结束
    # t3.join() # 等待worker线程结束
    # t4.join() # 等待worker线程结束


    # print result //////////////////////////////////
    print ('===== all done')
    end_time = perf_counter()
    print(f'All took {end_time- start_time: 0.2f} second(s) to complete.')

    # for key, value in od.items():
    #      print(key, len(value))
    # print(len(rt))
    return all_array,overlap_frames


def createStream(wave_path, pAudio):
    
    waveF = wave.open(wave_path, 'rb')
    print(waveF.getsampwidth())
    print(pAudio.get_format_from_width(waveF.getsampwidth()))
    print(waveF.getnchannels())
    print(waveF.getframerate())
    # instantiate PyAudio (1)
    # open stream (2)
    stream = pAudio.open(format=pAudio.get_format_from_width(waveF.getsampwidth()),
                    channels=waveF.getnchannels(),
                    rate=waveF.getframerate(),
                    output=True)
    return stream


def inferOneFile(argsAll, filname):
    start=datetime.now()

    # 1 ####################################
    wave_path = os.path.join(argsAll.input_wavs_dir, filname)
    pA = pyaudio.PyAudio()
    stream = createStream(wave_path, pA)

    print("start read wav - ================================")
    wav, sr = load_wav(wave_path)
    wav = wav / MAX_WAV_VALUE
    wav = torch.FloatTensor(wav).to(device)
    print(f"wav before mel - {wav.dtype} {wav.shape}")
    
    # 2 ####################################
    melData = get_mel(wav.unsqueeze(0))
    print(f"mel result - {melData.dtype} {melData.shape}")
    # save mel into a npy file
    output_file = os.path.join(argsAll.output_dir, os.path.splitext(filname)[0] + '_mel.npy')
    save(output_file, melData.cpu().numpy())
    print(f"save mel - {output_file} time used {datetime.now()-start}")
            
    print(f'process as whole ================================')
    # whole inference ######################################################################
    # whole inference ######################################################################

    # start=datetime.now()
    # y_g_hat = generator(melData)
    # audio = y_g_hat.squeeze()
    # audio = audio * MAX_WAV_VALUE
    # audio = audio.cpu().numpy().astype('int16')
    # print(f'mel shape  = {melData.shape} result shape {y_g_hat.shape} audio {audio.shape} time used {datetime.now()-start}')
    # # 
    # # save result wav file
    # output_file = os.path.join(argsAll.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
    # write(output_file, config.sampling_rate, audio)
    # print(output_file)

    # split to process ####################################################################
    # split to process ####################################################################
    print(f'process as splits ===============================')
    global all_array
    all_array, overlap_frames = slicesProcess(melData, stream)

    if all_array is not None:
        # save audio into file 
        # output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated_' + str(indexA) + '.wav')
        output_file = os.path.join(argsAll.output_dir, os.path.splitext(filname)[0] + '_generated_O_'+ str(overlap_frames) + '.wav')
        write(output_file, config.sampling_rate, all_array)
        print(f'end splits =======================================')
        print(output_file)
    
    # end of split to process ###
    # close PyAudio (5)
    pA.terminate()
    return



def inference(argsAll):
    global generator1, generator2, generator3, generator4
    generator1 = createGenerator(argsAll)
    generator2 = createGenerator(argsAll)
    generator3 = createGenerator(argsAll)
    generator4 = createGenerator(argsAll)


    filelist = os.listdir(argsAll.input_wavs_dir)

    os.makedirs(argsAll.output_dir, exist_ok=True)

    
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            return inferOneFile(argsAll, filname)

def createGenerator(argsAll):
    generator = Generator(config).to(device)
    state_dict_g = load_checkpoint(argsAll.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator


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




