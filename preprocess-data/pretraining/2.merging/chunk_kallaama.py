# 1.
import os
import sys
import xml.etree.ElementTree as ET
import torch
import torchaudio

def parse_trs(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    segments = []
    for section in root.iter('Section'):
        previous_end = 0.0
        for turn in section.iter('Turn'):
            start_time = float(turn.attrib.get('startTime', previous_end))
            end_time = float(turn.attrib.get('endTime', start_time + 10))  # Fallback if no endTime

            syncs = turn.findall('Sync')
            if syncs:
                for i in range(len(syncs)):
                    current_start = float(syncs[i].attrib['time'])
                    next_start = float(syncs[i + 1].attrib['time']) if (i + 1) < len(syncs) else end_time
                    segments.append((current_start, next_start))
            else:
                segments.append((start_time, end_time))

            previous_end = end_time

    return segments

def chunk_audio(audio_path, file_no_exten, outpath, segments, output_format="wav"):
    # Load the audio file
    filenames = []
    waveform, osr = torchaudio.load(audio_path)
    sr = 16000
    # Iterate over each segment and save the chunk
    for i, (start_time, end_time) in enumerate(segments):
        start_sample = int(start_time * osr)
        end_sample = int(end_time * osr)
        
        # Extract the chunk
        chunk = waveform[:, start_sample:end_sample]
        
        # Save the chunk
        chunk_path = f"{outpath}{file_no_exten}_{i}_{start_time:.2f}-{end_time:.2f}.{output_format}"
        filenames.append(chunk_path)
        # # save to PCM wav
        torchaudio.save(chunk_path, chunk, sr, bits_per_sample=16)
        #print(f"Exported {chunk_path}")
    return filenames


# Define paths
def main():
    # read filenames
    lan = sys.argv[1]

    langnames = {'fuc':'pulaar', 'srr':'sereer', 'wol':'wolof'}
    allnames = []
    pathnames = []
    with open(f"/data/users/jalabi/Internship_NII/Data/processed/kallaama2/clean_dataset_ready4release/filenames_{lan}.txt") as f:
        for line in f:
            pathnames.append(line.strip().replace("/kallaama/", "/kallaama2/"))
    
    outpath = f"/data/users/jalabi/Internship_NII/Data/processed/kallaama/clean_dataset_ready4release/{langnames[lan]}/speech_dataset/"
    # create path
    os.makedirs(outpath, exist_ok=True)

    trs_path = f"/data/users/jalabi/Internship_NII/Data/raw/kallaama-speech-dataset/data/transcriptions/raw/transcriptions-{lan}/trs"
    for audio_file_path in pathnames:
        # Extract the filename with extension
        file_name_with_extension = os.path.basename(audio_file_path)
        # Remove the extension
        file_no_exten = os.path.splitext(file_name_with_extension)[0]
        trs_file_path = f'{trs_path}/{file_no_exten}.trs'
        # audio_file_path = '/data/users/jalabi/Internship_NII/Data/processed/kallaama/clean_dataset_ready4release/pulaar/speech_dataset/1/fuc_4110.wav'
        # Parse TRS file and get segments
        segments = parse_trs(trs_file_path)
        # Chunk the audio file based on the segments
        outnames = chunk_audio(audio_file_path, file_no_exten, outpath, segments)
        allnames.extend(outnames)
    # write file
    with open(f"/data/users/jalabi/Internship_NII/Data/scripts/download/asr/files_path/{lan}.txt", "w") as f:
        for line in allnames:
            f.write(line.strip())
            f.write("\n")
    with open(f"/data/users/jalabi/Internship_NII/Data/processed/kallaama/clean_dataset_ready4release/filenames_{lan}.txt", "w") as f:
        for line in allnames:
            f.write(line.strip())
            f.write("\n")
    print("Audio chunks have been successfully created.")

if __name__ == '__main__':
    main()