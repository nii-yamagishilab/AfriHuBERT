# merge speech 
import os
import json
import glob
import random
import torch
import torchaudio
random.seed(42)

def load_from_json(input_path):
    with open(input_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def readtext(filename):
    path = []
    with open(filename) as f:
        for line in f:
            path.append(line.strip())
    return path

def writetext(filename, data):
    with open(filename, 'w') as f:
        for line in data:
            f.write(line.strip())
            f.write('\n')

def check_short(paths, short):
    short_wav = []
    for item in paths:
        if item in short:
            short_wav.append(item)
    return short_wav

def has_repetition(lst):
    return len(lst) != len(set(lst))

dictpath = "/data/users/jalabi/Internship_NII/Data/scripts/download/asr/files_path"

short1 = load_from_json(f"{dictpath}/short_train_wav.json")
short2 = load_from_json(f"{dictpath}/short_valid_wav.json")

short = {**short1, **short2}
new_sr = 16000
# check languages 
pattern = f"{dictpath}/short_wavs/train/*.txt"
for file_path in glob.glob(pattern):
    # Extract the file name with extension
    file_name_with_ext = os.path.basename(file_path)
    
    # Split the file name into name and extension
    lan, _ = os.path.splitext(file_name_with_ext)

    audiofiles = readtext(file_path)

    if len(audiofiles) <= 1:
        continue

    print(lan)
    framesize = 0
    concat = []
    concatenated_segments = []
    for item in audiofiles:
        framesize += short[item]
        # merge file 
        data, sr = torchaudio.load(item)
        # # take only one channel if more than one channel
        if data.shape[0] > 1:
            data = data[:1, :]
        # Concatenate the waveforms
        concat.append(data)
        
        if framesize>= 160000 and framesize<=320000 :
            # print(framesize)
            concatenated_waveform = torch.cat(concat, dim=1)
            concatenated_segments.append(concatenated_waveform)
            framesize = 0
            concat = []

    if framesize>0 and framesize<16000:
        print("There is a probe", framesize)
        remaining_waveform = torch.cat(concat, dim=1)
        if concatenated_segments:
                # Merge with the last segment if it's available
                last_segment = concatenated_segments.pop()
                remaining_waveform = torch.cat((last_segment, remaining_waveform), dim=1)
                concatenated_segments.append(remaining_waveform)
    
    
    os.makedirs(os.path.dirname(f"/data/users/jalabi/Internship_NII/Data/processed/merged_audios/{lan}/"), exist_ok=True)
    os.makedirs(os.path.dirname(f"{dictpath}/eventual/merged_text/"), exist_ok=True)
    wavnames = []
    for i, item in enumerate(concatenated_segments):
        # write audio
        outname = f'merged_audio_{i}.wav'
        torchaudio.save(f"/data/users/jalabi/Internship_NII/Data/processed/merged_audios/{lan}/{outname}", item, new_sr)
        wavnames.append(f"/data/users/jalabi/Internship_NII/Data/processed/merged_audios/{lan}/{outname}")
    writetext(f"{dictpath}/eventual/merged_text/{lan}.txt", wavnames)
        

