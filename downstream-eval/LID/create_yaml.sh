#!/bin/sh  

CODEDIR="/data/users/jalabi/Internship_NII/speechbrain/recipes/AfroLID_FLEURS"

echo "$CODEDIR/results/"
# Create results directory if it doesn't exist
mkdir -p "$CODEDIR/results/"


# Define setting variabl
setts="experiment1"

# Create directory for hparams based on settin
mkdir -p "$CODEDIR/hparams/$setts"

for model in mhubert_147 ssa_hubert_60k hubert_lg_ll60k xlsr_lg_53 xlsr_lg_300m; do
	# mhubert_147 ssa_hubert_60k hubert_lg_ll60k xlsr_lg_53 xlsr_lg_300m xlsr_lg_300m_hd256
	python $CODEDIR/create_yaml.py \
		--model $model \
		--outurl $CODEDIR/results/ \
		--yaml_out $CODEDIR/hparams/$setts \
		--resultname $setts
done



: <<'COMMENT'
# Define setting variabl
setts="ablation"

# Create directory for hparams based on settin
mkdir -p "$CODEDIR/hparams/$setts"

for model in mhubert_147; do 
	# xlsr_lg_300m xlsr_lg_300m_hd256 xlsr_lg_300m_frz; do
	# mhubert_147 ssa_hubert_60k hubert_lg_ll60k xlsr_lg_53 xlsr_lg_300m xlsr_lg_300m_hd256
	python $CODEDIR/create_yaml.py \
		--model $model \
		--outurl $CODEDIR/results/ \
		--yaml_out $CODEDIR/hparams/$setts \
		--resultname $setts
done
COMMENT