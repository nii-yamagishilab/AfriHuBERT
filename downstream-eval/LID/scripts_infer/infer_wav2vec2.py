import torchaudio 
from pathlib import Path
from speechbrain.inference import EncoderClassifier
from speechbrain.inference.interfaces import foreign_class

def compute_accuracy(predictions, ground_truth):
    # Ensure both lists have the same length
    if len(predictions) != len(ground_truth):
        raise ValueError("Lists must have the same length.")
    
    # Initialize variables to count correct predictions
    correct_count = 0
    
    # Iterate through each pair of predictions and ground truth
    for pred, truth in zip(predictions, ground_truth):
        if pred == truth:
            correct_count += 1
    
    # Calculate accuracy
    accuracy = correct_count / len(predictions) * 100.0
    return accuracy

actual_label = []
predit_label = []
model="/data/users/jalabi/Internship_NII/speechbrain/recipes/VoxLingua107/lang_id_fleur/results/epaca/1988/save/CKPT+2024-06-26+05-09-40+00/"
tempf="/data/users/jalabi/Internship_NII/speechbrain/recipes/VoxLingua107/lang_id_fleur/tmp2/"
classifier = foreign_class(source=model, savedir=tempf, pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
language_id = EncoderClassifier.from_hparams(source=model, savedir="/data/users/jalabi/Internship_NII/speechbrain/recipes/VoxLingua107/lang_id_fleur/tmp2/")
for lan in ["af_za",  'am_et',  'ar_eg', "en_us",  "ff_sn",  "fr_fr",  "ha_ng",  "ig_ng",  "kam_ke",  "lg_ug",  "ln_cd",  "luo_ke",  "nso_za",  "ny_mw",  "om_et",  "pt_br",  "sn_zw",  "so_so", "sw_ke",  "umb_ao",  "wo_sn",  "xh_za",  "yo_ng",  "zu_za",]:
    print(lan)
    folder_path = f"/data/users/jalabi/Internship_NII/Data/unzipped/fleurs_lid/dev/{lan}/"
    wav_files = Path(folder_path).glob('*.wav')
    for wav_file in wav_files:
        out_prob, score, index, text_lab = classifier.classify_file(str(wav_file))
        actual_label.append(lan)
        predit_label.extend(text_lab)

print(compute_accuracy(actual_label, predit_label))
