import ruamel.yaml 
import argparse



# Function to read YAML file
def read_yaml(yaml, file_path):
    with open(file_path, 'r') as file:
        return yaml.load(file,)

def main(args):
   modelname = args.model
   oldyaml = args.oldyaml
   outurl = args.outurl
   yaml_out = args.yaml_out
   classes = args.classes
   resultname = args.resultname
   yaml = ruamel.yaml.YAML()
   data = read_yaml(yaml, oldyaml)

   models = {
       'mhubert_147':{'model_hub': 'utter-project/mHuBERT-147', 'save': 'hubert', 'hidden_size': 768, 'lin_neurons':512 },
       'ssa_hubert_60k':{'model_hub': 'Orange/SSA-HuBERT-base-60k', 'save': 'hubert', 'hidden_size': 768, 'lin_neurons':512 }, 
       'hubert_lg_ll60k':{'model_hub': 'facebook/hubert-large-ll60k', 'save': 'hubert', 'hidden_size': 1024, 'lin_neurons':512  }, 
       'xlsr_lg_53':{'model_hub': 'facebook/wav2vec2-large-xlsr-53', 'save': 'wav2vec', 'hidden_size': 1024, 'lin_neurons':512 }, 
       'xlsr_lg_300m':{'model_hub': 'facebook/wav2vec2-xls-r-300m', 'save': 'wav2vec', 'hidden_size': 1024, 'lin_neurons':512 },
       'xlsr_lg_300m_hd256':{'model_hub': 'facebook/wav2vec2-xls-r-300m', 'save': 'wav2vec', 'hidden_size': 1024, 'lin_neurons':256 },
       'xlsr_lg_300m_frz':{'model_hub': 'facebook/wav2vec2-xls-r-300m', 'save': 'wav2vec', 'hidden_size': 1024, 'lin_neurons':512, 'freeze_wav2vec2_conv': True},
   
   }
   
   # the main properties that should change 
   data['output_url'] = outurl
   data['wav2vec2_hub'] = models[modelname]['model_hub']
   data['wav2vec2_folder'] = ruamel.yaml.comments.TaggedScalar(tag="!ref", value=f"<save_folder>/{models[modelname]['save']}")
   data['output_folder'] = ruamel.yaml.comments.TaggedScalar(tag="!ref", value=f"<output_url>/{resultname}/{modelname}/<seed>")
   data['attentive']['channels'] = models[modelname]['hidden_size']
   data['classifier']['input_shape'][-1] = 2*models[modelname]['hidden_size']
   data['classifier']['lin_neurons'] = models[modelname]['lin_neurons']
   data['encoder_dim'] = models[modelname]['hidden_size']
   data['out_n_neurons'] = classes
   if 'freeze_wav2vec2_conv' in models[modelname]:
      data['freeze_wav2vec2_conv'] = models[modelname]['freeze_wav2vec2_conv']
   
   file_path = f'{yaml_out}/{modelname}.yaml'
   with open(file_path, 'w') as f:
      yaml.dump(data, f)

if __name__ == "__main__":
   # argparse
   parser = argparse.ArgumentParser(description='Generate YAML with custom tags')
   parser.add_argument('--model', type=str, default="xlsr_lg_300m", required=True, choices=['mhubert_147', 'ssa_hubert_60k', 'hubert_lg_ll60k', 'xlsr_lg_53', 'xlsr_lg_300m', 'xlsr_lg_300m_hd256', 'xlsr_lg_300m_frz'], help='Value for field1')
   parser.add_argument('--oldyaml', type=str, default="hparams/ecapa_xlsr-300.yaml", help='The old yaml file to update or start with')
   parser.add_argument('--outurl', type=str, required=True,  help='The url where the trained models should be saved')
   parser.add_argument('--yaml_out', type=str, required=True, help='Value for custom tag')
   parser.add_argument('--resultname', type=str, default="afrolid_output", required=True, help='Value for custom tag')
   parser.add_argument('--classes', type=int, default=25, help='Number of classes')
   args = parser.parse_args()
   main(args)
