import matplotlib.pyplot as plt
import librosa
import librosa.display
import glob
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt


def main():

    # matplotlib inline
    # librosa.util.example('brahms')
    filename = librosa.util.example_audio_file()
    filename = '/home/xiaowei/Documents/TTS/hifi-gan/test_files/LJ001-0001.wav'
    signal, sr = librosa.load(filename)

    # # display wav
    # # display wav
    # plt.figure(figsize=(20, 5))
    # librosa.display.waveplot(signal, sr=sr)
    # plt.title('Waveplot', fontdict=dict(size=18))
    # plt.xlabel('Time', fontdict=dict(size=15))
    # plt.ylabel('Amplitude', fontdict=dict(size=15))
    # plt.show()

    # play sound
    # import IPython.display as ipd  
    # ipd.Audio(signal, rate=sr)

    # # display MEL Spectrum 
    # # display MEL Spectrum 
    # this is the number of samples in a window per fft
    n_fft = 2048
    # The amount of samples we are shifting after each fft
    hop_length = 512
    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hop_length, n_fft=n_fft)
    spectrogram = np.abs(mel_signal)
    print(spectrogram.dtype)
    print(spectrogram.shape)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    print(power_to_db.dtype)
    print(power_to_db.shape)


    plt.figure(figsize=(8, 7))
    librosa.display.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma', 
    hop_length=hop_length)
    plt.colorbar(label='dB')
    plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
    plt.xlabel('Time', fontdict=dict(size=15))
    plt.ylabel('Frequency', fontdict=dict(size=15))
    plt.show()


    # path=glob.glob('/home/xiaowei/Documents/TTS/hifi-gan/test_files/*.wav') 
    # fig, ax =  plt.subplots(nrows=1, ncols=3, sharex=True)
        
    # for i in range(1) :
    
    #     y, sr = librosa.load(path[i], sr=22050)
    #     librosa.display.waveplot(y, sr, ax=ax[i, 0])  # put wave in row i, column 0
    #     plt.axis('off')
        
        
    
    #     mfcc=librosa.feature.mfcc(y) 
    #     librosa.display.specshow(mfcc, x_axis='time', ax=ax[i, 1]) # mfcc in row i, column 1
    

    #     S = librosa.feature.melspectrogram(y, sr)
    #     librosa.display.specshow(librosa.power_to_db(S), x_axis='time', y_axis='log', ax=ax[i, 2])  # spectrogram in row i, column 2
    return

if __name__ == '__main__':
    main()








