source /data/users/jalabi/adaptation/layer_evals/rename_gpus.sh
# mkdir -p $OUTDIR
IDX=$1
export NUMBA_CACHE_DIR="/data/users/jalabi/Internship_NII/Data/scripts/download/asr/cache"
inputpath="/data/users/jalabi/Internship_NII/Data/scripts/download/asr/all_naija.txt"
outputpath="/data/users/jalabi/Internship_NII/Data/processed/naijavoices/"
mkdir -p $outputpath
python3 /data/users/jalabi/Internship_NII/Data/scripts/download/asr/resample_audio_naij.py \
	--input $inputpath \
	--index $IDX \
	--channel 1 \
	--rate 16000 \
	--output $outputpath
