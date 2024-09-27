import sys
import ruamel.yaml 
import argparse
import shutil
import pickle
from pathlib import Path
from sklearn.metrics import classification_report
from speechbrain.inference.interfaces import foreign_class

# Function to read YAML file
def read_yaml(yaml, file_path):
    with open(file_path, 'r') as file:
        return yaml.load(file,)


def main(args):
    actual_label = []
    predit_label = []
    yaml = ruamel.yaml.YAML()
    resultname = args.resultname
    modelname = args.model
    output = args.output

    # Define the source and destination paths
    source_path = f'/data/users/jalabi/Internship_NII/speechbrain/recipes/AfroLID_FLEURS/hparams_infer/{resultname}/{modelname}.yaml'
    destination_path = f'/data/users/jalabi/Internship_NII/speechbrain/recipes/AfroLID_FLEURS/scripts_infer/tmp/{modelname}/hyperparams.yaml'
    # Copy and rename the file
    shutil.copy(source_path, destination_path)
    shutil.copy('./scripts_infer/custom_interface_ecapa1.py', f'/data/users/jalabi/Internship_NII/speechbrain/recipes/AfroLID_FLEURS/scripts_infer/tmp/{modelname}/')

    oldyaml = f"/data/users/jalabi/Internship_NII/speechbrain/recipes/AfroLID_FLEURS/hparams_infer/experiment1/{modelname}.yaml"
    data = read_yaml(yaml, oldyaml)

    model = data['pretrained_path']
    tempf= f"/data/users/jalabi/Internship_NII/speechbrain/recipes/AfroLID_FLEURS/scripts_infer/tmp/{modelname}"
    classifier = foreign_class(source=model, savedir=tempf, pymodule_file="custom_interface_ecapa1.py", classname="CustomEncoderWav2vec2Classifier")
    # language_id = EncoderClassifier.from_hparams(source=model, savedir="/data/users/jalabi/Internship_NII/speechbrain/recipes/VoxLingua107/lang_id_fleur/tmp2/")
    for lan in ["af_za", 'am_et', 'ar_eg', "en_us",  "ff_sn",  "fr_fr",  "ha_ng",  "ig_ng",  "kam_ke",  "lg_ug",  "ln_cd",  "luo_ke",  "nso_za",  "ny_mw",  "om_et",  "pt_br", 'rw_rw',  "sn_zw",  "so_so", "sw_ke",  "umb_ao",  "wo_sn",  "xh_za",  "yo_ng",  "zu_za",]:
        # print(lan)
        folder_path = f"/data/users/jalabi/Internship_NII/Data/processed/fleurs_lid/test/{lan}/"
        wav_files = Path(folder_path).glob('*.wav')
        for wav_file in wav_files:
            out_prob, score, index, text_lab = classifier.classify_file(str(wav_file))
            #print(text_lab)
            actual_label.append(lan)
            predit_label.extend(text_lab)

    # Generate the classification report as a dictionary
    report_dict = classification_report(actual_label, predit_label, output_dict=True)
    # write dict
    print(report_dict)
    outfilename = f"{output}/{modelname}.pkl"
    with open(outfilename, 'wb') as f:
        pickle.dump(report_dict, f)
if __name__ == "__main__":
   # argparse
   parser = argparse.ArgumentParser(description='Generate YAML with custom tags')
   parser.add_argument('--model', type=str, default="xlsr_lg_300m", required=True, choices=['mhubert_147', 'ssa_hubert_60k', 'hubert_lg_ll60k', 'xlsr_lg_53', 'xlsr_lg_300m', 'xlsr_lg_300m_hd256', 'xlsr_lg_300m_frz', 'af_xlsr_lg_300m'], help='Value for field1')
   parser.add_argument('--output', type=str, required=True,  help='The url where the trained models should be saved')
   parser.add_argument('--resultname', type=str, default="afrolid_output", required=True, help='Value for custom tag')
   args = parser.parse_args()
   main(args)
