# 3
import os
import json
import torch
import random
import torchaudio

# Filter short and long audios, merge short audio up to 20s and remove long 

def read_from_json(input_path):
    with open(input_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def save_to_json(data, output_path):
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def get_good_audio(data):
    good_audio = dict()
    short_audio = dict()
    for item in data:
        if item == "Total":
            continue
        aud_time = float(data[item])
        if aud_time >= 1 and aud_time <= 30:
            # the data we want
            good_audio[item] = aud_time 
        
        if aud_time < 1:
            short_audio[item] = aud_time
    return good_audio, short_audio

def merge_audio(short):
    concat = []

    concatenated_segments = []
    concatnames = []
    allconcatnames = []
    duration = []
    dur = 0.0
    for item in short:
        dur += short[item]
        # merge file 
        data, sr = torchaudio.load(item)
        # # take only one channel if more than one channel
        if data.shape[0] > 1:
            data = data[:1, :]
        # Concatenate the waveforms
        concat.append(data)
        concatnames.append(item)
        
        
        if dur>= 20 and dur<=30 :
            # print(framesize)
            concatenated_waveform = torch.cat(concat, dim=1)
            concatenated_segments.append(concatenated_waveform)
            allconcatnames.append(concatnames)
            duration.append(dur)
            dur = 0.0
            concat = []
            concatnames = []
    
    if dur > 0 and dur < 1:
        # combine with previous
        remaining_waveform = torch.cat(concat, dim=1)
        if concatenated_segments:
                # Merge with the last segment if it's available
                last_segment = concatenated_segments.pop()
                last_dur = duration.pop()
                last_names = allconcatnames.pop()
                print("lastnames = ", last_names)
                print("concat name  = ", concatnames)
                print(last_names.extend(concatnames))
                print(last_names+concatnames)
                remaining_waveform = torch.cat((last_segment, remaining_waveform), dim=1)
                concatenated_segments.append(remaining_waveform)
                duration.append(last_dur+dur)
                allconcatnames.append(last_names+concatnames)

                assert last_dur+dur <= 30, "There is a merged file greater than 30s"

    elif dur>=1:
        concatenated_waveform = torch.cat(concat, dim=1)
        concatenated_segments.append(concatenated_waveform)
        allconcatnames.append(concatnames)
        duration.append(dur)
        
    assert len(concatenated_segments) == len(allconcatnames) == len(duration); "lengths not equal"
    return concatenated_segments, allconcatnames, duration
               
def write_merged_audio(concatenated_segments, allconcatnames, duration, lan="de"):
    index = 0
    sr = 16000
    mergedaudio = {}
    mergedaudiodet = dict()
    savedir = f"/data/users/jalabi/Internship_NII/Data/processed/merged_audios/{lan}/"
    os.makedirs(os.path.dirname(savedir), exist_ok=True)
    for x, y, z in zip(concatenated_segments, allconcatnames, duration):
        index += 1
        filename = f"{savedir}merged_audio_{index}.wav"
        mergedaudio[filename] = z
        mergedaudiodet[filename] = y
        # # save to PCM wav
        torchaudio.save(filename, x, sr, bits_per_sample=16)
    
    # write mergedaudiodet
    save_to_json(mergedaudiodet, f"{savedir}merged.json")

    return mergedaudio

def get_train_valid(data, randompaths):
    dur = 0.0
    tdur = 0.0
    valid = []
    train = []
    for item in randompaths:
        if dur <= 3600:
            dur += data[item]
            valid.append(item)
            continue
        tdur += data[item]
        train.append(item)
    return train, valid, dur, tdur
        
def write_trainjson(wav_files, wavdict, writepath):
    dictionary = dict()
    for wav_file in wav_files:
        utt_id = wav_file.split("/")[-1].replace(".wav","")
        dictionary[utt_id] = dict()
        dictionary[utt_id]["audio"] = wav_file
        dictionary[utt_id]["duration"] = float(wavdict[wav_file])
        dictionary[utt_id]["durstr"] = str(wavdict[wav_file])
        dictionary[utt_id]["gender"] = "U"
        dictionary[utt_id]["accent"] = "U"
        dictionary[utt_id]["age"] = "U"
        dictionary[utt_id]["speaker_id"] = "U"
    save_to_json(dictionary, f"{writepath}")

def generate_manifest(data, datadict, root_folder, lang):
    manifest_lines = list()
    for element in data:
        frames = int(datadict[element] * 16000)
        string = element.replace(root_folder,"") + "\t" + str(frames) + "\n"
        manifest_lines.append(string)
    return manifest_lines

def write_manifest(manifest, file_name, root_folder):
    with open(file_name, "w") as output_file:
        output_file.write(root_folder + "\n")
        for line in manifest:
            output_file.write(line)

def main():
    random.seed(42)
    jsonpath = "/data/users/jalabi/Internship_NII/Data/scripts/download/asr/files_path/json_dur"
    data_stats = []
    alldur = 0.0
    for lan in ['afr', 'aka1', 'aka2', 'amh', 'ara', 'bem', 'eng', 'ewe', 'fra', 'fuc', 'hau', 'ibo', 'kin', 'lin', 'loz', 'lug', 'lun', 'mlg', 'nbl', 'nso', 'nya', 'por', 'sna', 'som', 'sot', 'srr', 'ssw', 'swh', 'toi', 'tsn', 'tso', 'ven', 'wol', 'xho', 'yor', 'zul', 'nico']:
        # read json
        print(lan)
        inputjson = f"{jsonpath}/{lan}.json"
        data = read_from_json(inputjson)

        good_audio, short_audio = get_good_audio(data)

        # for short audio, merge them 1. get the audio
        con_seg, cont_names, duration = merge_audio(short_audio)
        mgaudios = write_merged_audio(con_seg, cont_names, duration, lan = lan)

        # combine the merged and the good
        combined = {**good_audio, **mgaudios}
        total_dur = sum(combined.values())

        if lan != 'nico':
            allaudiopaths = list(combined.keys()) # shuffle them and sample 1hour
            # Shuffle the list
            random.shuffle(allaudiopaths)

            # sampled 1 hour as valid
            train, valid, valid_dur, tdur = get_train_valid(combined, allaudiopaths)
            traindur = sum([combined[ts] for ts in train])
            # validdur = sum([combined[ts] for ts in valid])
            error = f"Train duration is not same!!! "
            assert traindur == tdur, f"{error}"
            assert len(allaudiopaths) == len(train) + len(valid); "Length of valid and train not same as original data"
        else:
            # read nico dev and save
            train = list(data.keys())
            inputjson = f"{jsonpath}/{lan}_dev.json"
            data = read_from_json(inputjson)
            valid = list(data.keys())
            combined = {**combined, **data}
            

        statspath = "/data/users/jalabi/Internship_NII/Data/scripts/download/asr/files_path/dataset_stat"


        os.makedirs(os.path.dirname(f"{statspath}/train/"), exist_ok=True)
        os.makedirs(os.path.dirname(f"{statspath}/valid/"), exist_ok=True)

        write_trainjson(train, combined, f"{statspath}/train/{lan}.json")
        # write origianl manifest for both train and valid
        write_trainjson(valid, combined, f"{statspath}/valid/{lan}.json")

        # manifestpath = "/data/users/jalabi/Internship_NII/Data/scripts/download/asr/files_path/manifest"
        #os.makedirs(os.path.dirname(f"{manifestpath}/valid/"), exist_ok=True)
        #basefolder = "/data/users/jalabi/Internship_NII/Data/processed/"
        #devmanifest = generate_manifest(data, basefolder, lan)
        #write_manifest(devmanifest, f"{manifestpath}/valid/{lan}.tsv", basefolder)


if __name__ ==  '__main__':
    main()
