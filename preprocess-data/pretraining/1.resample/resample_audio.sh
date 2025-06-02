source /data/users/jalabi/adaptation/layer_evals/rename_gpus.sh
# mkdir -p $OUTDIR
LANG=$1
export NUMBA_CACHE_DIR="/data/users/jalabi/Internship_NII/Data/scripts/download/asr/cache"
inputpath="/data/users/jalabi/Internship_NII/Data/unzipped/zambezi/$LANG/"
outputpath="/data/users/jalabi/Internship_NII/Data/processed/zambezi/$LANG/"
mkdir -p $outputpath
python3 /data/users/jalabi/Internship_NII/Data/scripts/download/asr/resample_audio.py \
	--input $inputpath \
	--channel 1 \
	--rate 16000 \
	--output $outputpath
