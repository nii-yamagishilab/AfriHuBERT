source /data/users/jalabi/adaptation/layer_evals/rename_gpus.sh

export NUMBA_CACHE_DIR="/data/users/jalabi/Internship_NII/speechbrain/recipes/AfroLID_FLEURS/cache"
export TRANSFORMERS_CACHE="/data/users/jalabi/Internship_NII/speechbrain/recipes/AfroLID_FLEURS/cache"
mkdir -p $TRANSFORMERS_CACHE $NUMBA_CACHE_DIR
CODEDIR="/data/users/jalabi/Internship_NII/speechbrain/recipes/AfroLID_FLEURS"
# python3 $CODEDIR/train_emo_lid.py $CODEDIR/hparams/method1/train_xlsr-53.yaml
setup=$1
python3 $CODEDIR/train_emo_lid.py $CODEDIR/hparams/ablation/$1.yaml
