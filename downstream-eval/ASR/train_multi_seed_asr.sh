source AfriHuBERT/downstream-eval/rename_gpus.sh

export NUMBA_CACHE_DIR="AfriHuBERT/downstream-eval/ASR/cache"
export TRANSFORMERS_CACHE="AfriHuBERT/downstream-eval/ASR/cache"
mkdir -p $TRANSFORMERS_CACHE
CODEDIR="AfriHuBERT/downstream-eval/ASR"
config=$1
mode=$2
seed=$3

python3 $CODEDIR/train_with_wav2vec2.py $CODEDIR/hparams/$mode/$seed/$config.yaml
