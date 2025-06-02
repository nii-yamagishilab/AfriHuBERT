################################################################################
#
# Converts the unzipped <LANG_ID>/<VIDEO---0000.000-0000.000.wav> folder
# structure of FLEURs into a WebDataset format
#
# Author(s): Tanel Alumäe, Nik Vaessen
################################################################################

import argparse
import json
import pathlib
import random
import re
from collections import defaultdict

import torch
import torchaudio
import webdataset as wds
# import librosa
from speechbrain.dataio.dataio import read_audio
################################################################################
# methods for writing the shards

ID_SEPARATOR = "&"
SAMPLERATE = 16000

def load_audio(audio_file_path: pathlib.Path) -> torch.Tensor:
    t, sr = torchaudio.load(audio_file_path)

    if sr != 16000:
        raise ValueError("expected sampling rate of 16 kHz")

    return t


def write_shards(
    fleurs_folder_path: pathlib.Path,
    shards_path: pathlib.Path,
    seed: int,
    samples_per_shard: int,
    min_dur: float,
):
    """
    Arguments
    ---------
    fleurs_folder_path: pathlib.Path
        folder where extracted voxceleb data is located
    shards_path: pathlib.Path
        folder to write shards of data to
    seed: int
        random seed used to initially shuffle data into shards
    samples_per_shard: int
        number of data samples to store in each shards.
    min_dur: float
    """
    # make sure output folder exist
    shards_path.mkdir(parents=True, exist_ok=True)

    # find all audio files
    # audio_files = sorted([f for f in fleurs_folder_path.rglob("*.wav")])
    languages = ['af_za', 'am_et',  'ar_eg',  'en_us',  'ff_sn',  'fr_fr',  'ha_ng',  'ig_ng',  'kam_ke',  'lg_ug',  'ln_cd',  'luo_ke',  'nso_za',  'ny_mw',  'om_et',  'pt_br',  'rw_rw',  'sn_zw',  'so_so',  'sw_ke',  'umb_ao',  'wo_sn',  'xh_za',  'yo_ng',  'zu_za']
    allfiles = []
    flores_path = "/data/users/jalabi/Internship_NII/Data/raw/fleurs"
    if "train" in fleurs_folder_path:
        split = "train"
    else:
        split = "dev"
    data_path = f"/data/users/jalabi/Internship_NII/Data/processed/fleurs_lid/{split}"
    for lan in languages:
        orig_tsv_file = f"{flores_path}/{lan}/{fleurs_folder_path}"
        loaded_csv = open(orig_tsv_file, "r").readlines()
        for line in loaded_csv:
            linsp = line.split("\t")
            audio_id = linsp[1].strip().replace(".mp3", ".wav")
            allfiles.append(pathlib.Path(f"{data_path}/{lan}/{audio_id}"))
    audio_files = sorted(allfiles)

    # create tuples (unique_sample_id, language_id, path_to_audio_file, duration)
    data_tuples = []

    # track statistics on data
    all_language_ids = set()
    sample_keys_per_language = defaultdict(list)

    for f in audio_files:
        # path should be
        # fleurs_folder_path/<LANG_ID>/<VIDEO---0000.000-0000.000.wav>
        m = re.match(
            r"(.*/((.+)/\d+)\.wav)",
            f.as_posix(),
        )
        if m:
            loc = m.group(1)
            key = m.group(2)
            lang = m.group(3)
            # start = float(m.group(4))
            # end = float(m.group(5))
            # read the file and get the duration in seconds
            signal = read_audio(loc)
            dur = signal.shape[0] / SAMPLERATE
            # dur = librosa.get_duration(filename=loc)
            # Period is not allowed in a WebDataset key name
            key = key.replace(".", "_")
            if dur > min_dur:
                # store statistics
                all_language_ids.add(lang)
                sample_keys_per_language[lang].append(key)
                t = (key, lang, loc, dur)
                data_tuples.append(t)
        else:
            raise Exception("Unexpected wav name: " + f)

    all_language_ids = sorted(all_language_ids)

    # write a meta.json file which contains statistics on the data
    # which will be written to shards
    meta_dict = {
        "language_ids": list(all_language_ids),
        "sample_keys_per_language": sample_keys_per_language,
        "num_data_samples": len(data_tuples),
    }

    with (shards_path / "meta.json").open("w") as f:
        json.dump(meta_dict, f)

    # shuffle the tuples so that each shard has a large variety in languages
    random.seed(seed)
    random.shuffle(data_tuples)

    # write shards
    all_keys = set()
    shards_path.mkdir(exist_ok=True, parents=True)
    pattern = str(shards_path / "shard") + "-%06d.tar"

    with wds.ShardWriter(pattern, maxcount=samples_per_shard) as sink:
        for key, language_id, f, duration in data_tuples:
            # load the audio tensor
            tensor = load_audio(f)

            # verify key is unique
            assert key not in all_keys
            all_keys.add(key)

            # extract language_id, youtube_id and utterance_id from key
            # language_id = all_language_ids[language_id_idx]

            # create sample to write
            sample = {
                "__key__": key,
                "audio.pth": tensor,
                "language_id": language_id,
            }

            # write sample to sink
            sink.write(sample)


################################################################################
# define CLI

parser = argparse.ArgumentParser(
    description="Convert FLEURs to WebDataset shards"
)

parser.add_argument(
    "fleurs_path",
    type=str,
    help="directory containing the (unzipped) FLEURs dataset",
)
# pathlib.Path
parser.add_argument(
    "shards_path", type=pathlib.Path, help="directory to write shards to"
)
parser.add_argument(
    "--seed",
    type=int,
    default=12345,
    help="random seed used for shuffling data before writing to shard",
)
parser.add_argument(
    "--samples_per_shard",
    type=int,
    default=5000,
    help="the maximum amount of samples placed in each shard. The last shard "
    "will most likely contain fewer samples.",
)
parser.add_argument(
    "--min-duration",
    type=float,
    default=3.0,
    help="Minimum duration of the audio",
)


################################################################################
# execute script

if __name__ == "__main__":
    args = parser.parse_args()

    write_shards(
        args.fleurs_path,
        args.shards_path,
        args.seed,
        args.samples_per_shard,
        args.min_duration,
    )
