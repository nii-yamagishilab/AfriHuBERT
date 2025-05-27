source AfriHuBERT/downstream-eval/rename_gpus.sh

export NUMBA_CACHE_DIR="AfriHuBERT/downstream-eval/LID/cache"
export TRANSFORMERS_CACHE="AfriHuBERT/downstream-eval/LID/cache"
mkdir -p $TRANSFORMERS_CACHE $NUMBA_CACHE_DIR
CODEDIR="AfriHuBERT/downstream-eval/LID"

setup=$1

for seed in 1 2 3; do   
    python3 $CODEDIR/trainlid.py $CODEDIR/hparams/mainlid/$seed/$1.yaml
done
