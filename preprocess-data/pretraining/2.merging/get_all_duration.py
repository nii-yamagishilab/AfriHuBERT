#2 get duration as requires for manifest
import os
import sys
import json
import librosa

def save_to_json(data, output_path):
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def main():
    lan = sys.argv[1]
    # real filename with 
    filepaths = []
    with open(f"/data/users/jalabi/Internship_NII/Data/scripts/download/asr/files_path/{lan}.txt") as f:
        for lines in f:
            filepaths.append(lines.strip())
    dur_dict = dict()
    totalt = 0
    for item in filepaths:
        if item.strip() in dur_dict:
            continue
        times = librosa.get_duration(filename=item.strip())
        dur_dict[item.strip()] = str(times)
        totalt += times
    dur_dict["Total"] = str(totalt)
    
    os.makedirs(os.path.dirname(f"/data/users/jalabi/Internship_NII/Data/scripts/download/asr/files_path/json_dur/"), exist_ok=True)
    output_path = f"/data/users/jalabi/Internship_NII/Data/scripts/download/asr/files_path/json_dur/{lan}.json"
    save_to_json(dur_dict, output_path)

if __name__ ==  '__main__':
    main()