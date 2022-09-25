"""PyAudio Example: Play a WAVE file."""

import sys
import wave
from tqdm import tqdm
import pyaudio
import time

CHUNK = 1024

# ##################################################
# def generate_sample(ob, preview):
#     print("* Generating sample...")
#     tone_out = array(ob, dtype=int16)
#
#     if preview:
#         print("* Previewing audio file...")
#
#         bytestream = tone_out.tobytes()
#         pya = pyaudio.PyAudio()
#         stream = pya.open(format=pya.get_format_from_width(width=2), channels=1, rate=OUTPUT_SAMPLE_RATE, output=True)
#         stream.write(bytestream)
#         stream.stop_stream()
#         stream.close()
#
#         pya.terminate()
#         print("* Preview completed!")
#     else:
#         write('sound.wav', SAMPLE_RATE, tone_out)
#         print("* Wrote audio file!")


def record_audio(wave_out_path,record_second):
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)


    wf = wave.open(wave_out_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)


    print("* recording")

    for i in tqdm(range(0, int(RATE / CHUNK * record_second))):
        data = stream.read(CHUNK)
        wf.writeframes(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()

    p.terminate()

    wf.close()

def play_audio(wave_path):
    waveF = wave.open(wave_path, 'rb')

    # instantiate PyAudio (1)
    p = pyaudio.PyAudio()

    print(waveF.getsampwidth())
    print(p.get_format_from_width(waveF.getsampwidth()))
    print(waveF.getnchannels())
    print(waveF.getframerate())
    # open stream (2)
    stream = p.open(format=p.get_format_from_width(waveF.getsampwidth()),
                    channels=waveF.getnchannels(),
                    rate=waveF.getframerate(),
                    output=True)

    # data = wf.readframes(CHUNK)
    # while len(data) != 0:
    #     print(f"{len(data)}")
    #     stream.write(data)
    #     data = wf.readframes(CHUNK)

    # read data
    datas = []
    data = waveF.readframes(CHUNK)
    while len(data) > 0:
        datas.append(data)
        data = waveF.readframes(CHUNK)

    # play stream (3)
    for d in tqdm(datas):
        stream.write(d)

    # stop stream (4)
    stream.stop_stream()
    stream.close()

    # close PyAudio (5)
    p.terminate()
    print('done')



def play_audio_callback(wave_path):

    wf = wave.open(wave_path, 'rb')

    # instantiate PyAudio (1)
    pAudio = pyaudio.PyAudio()

    def callback(in_data, frame_count, time_info, status):
        data = wf.readframes(frame_count)
        return (data, pyaudio.paContinue)


    # open stream (2)
    stream = pAudio.open(format=pAudio.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    stream_callback=callback)

    # read data
    stream.start_stream()

    while stream.is_active():
        time.sleep(0.1)

    # stop stream (4)
    stream.stop_stream()
    stream.close()

    # close PyAudio (5)
    pAudio.terminate()


# ##################################################
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
        sys.exit(-1)

    play_audio(sys.argv[1])
    # play_audio_callback(sys.argv[1])

    # record_audio("output.wav", record_second=4)
    # play_audio("output.wav")


