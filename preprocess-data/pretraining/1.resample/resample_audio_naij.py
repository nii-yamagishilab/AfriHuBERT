import argparse
import os
from pathlib import Path
import glob
import librosa
import soundfile as sf
import numpy as np

# Function to divide a list into n parts
def divide_list(input_list, n):
    avg = len(input_list) / float(n)
    out = []
    last = 0.0

    while last < len(input_list):
        out.append(input_list[int(last):int(last + avg)])
        last += avg

    return out

def resample_audio(folder_path,index, channel, rate, output_path):
    # Use glob to find all files with .wav extension in the folder
    #wav_files = glob.glob(os.path.join(folder_path, '*.wav'))
    # search_pattern = os.path.join(folder_path, '*.wav')
    # wav_files = glob.glob(search_pattern)

    # Read the text file into a list
    with open(folder_path, 'r') as filex:
        lines = filex.readlines()
    # Remove any trailing newline characters
    lines = [line.strip() for line in lines]
    allfiles = divide_list(lines, 4)

    wav_files = allfiles[index] #  Path(folder_path).glob('**/*.wav')
    fd =  open(f"naija_+{index}.txt", 'w')
    for wav_file in wav_files:
        filename=os.path.basename(wav_file)
        output_file=f"{output_path}/{filename}"
        fd.write(output_file)
        fd.write("\n")
        resample(str(wav_file), channel, rate, output_file)

def resample(input_file, channel, rate, output_file):
    # Load audio file
    data, sr = librosa.load(input_file, sr=48_000)

    # take only one channel if more than one channel
    if data.ndim > 1:
        data = data[0, :]

    # re-sampling as float numbers
    data_downsample = librosa.resample(data, orig_sr = sr, target_sr = rate)

    # scale amplitude 
    data_downsample = data_downsample / np.abs(data_downsample).max()
    # save to PCM wav
    sf.write(output_file, data_downsample, rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resample audio file.')
    parser.add_argument('--input', required=True, help='Input audio file path')
    parser.add_argument('--index', type=int, choices=[0, 1, 2, 3], required=True, help='Number of channels (1 for mono, 2 for stereo)')
    parser.add_argument('--channel', type=int, choices=[1, 2], required=True, help='Number of channels (1 for mono, 2 for stereo)')
    parser.add_argument('--rate', type=int, required=True, help='Sampling rate in Hz')
    parser.add_argument('--output', required=True, help='Output audio file path')

    args = parser.parse_args()
    
    resample_audio(args.input, args.index, args.channel, args.rate, args.output)
