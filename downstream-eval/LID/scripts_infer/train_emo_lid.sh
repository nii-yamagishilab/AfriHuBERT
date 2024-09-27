source /data/users/jalabi/adaptation/layer_evals/rename_gpus.sh

# DATADIR="/data/users/jalabi/Internship_NII/Data/unzipped"
# OUTDIR="/data/users/jalabi/Internship_NII/Data/processed"

export NUMBA_CACHE_DIR="/data/users/jalabi/Internship_NII/speechbrain/recipes/AfroLID_FLEURS/cache"
export TRANSFORMERS_CACHE="/data/users/jalabi/Internship_NII/speechbrain/recipes/AfroLID_FLEURS/cache"
mkdir -p $TRANSFORMERS_CACHE $NUMBA_CACHE_DIR
CODEDIR="/data/users/jalabi/Internship_NII/speechbrain/recipes/AfroLID_FLEURS"
# python3 $CODEDIR/train_emo_lid.py $CODEDIR/hparams/method1/train_xlsr-53.yaml
setts="experiment1"

for model in xlsr_lg_300m af_xlsr_lg_300m; do
	# mhubert_147 ssa_hubert_60k hubert_lg_ll60k xlsr_lg_53 xlsr_lg_300m af_xlsr_lg_300m; do
	echo $model
	mkdir -p $CODEDIR/scripts_infer/output/$setts $CODEDIR/scripts_infer/tmp/$model
	# mhubert_147 ssa_hubert_60k hubert_lg_ll60k xlsr_lg_53 xlsr_lg_300m xlsr_lg_300m_hd256
	python3.9 $CODEDIR/scripts_infer/infer_lid_wav2vec2.py \
		--model $model \
		--resultname $setts \
		--output $CODEDIR/scripts_infer/output/$setts/
		
done


