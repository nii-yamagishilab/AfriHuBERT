source /data/users/jalabi/adaptation/layer_evals/rename_gpus.sh
CODEDIR="/data/users/jalabi/Internship_NII/speechbrain/recipes/VoxLingua107/lang_id_fleur"
DATADIR="/data/users/jalabi/Internship_NII/Data/processed/fleurs_lid"
OUTDIR="$CODEDIR/data"

mkdir -p $OUTDIR
python3 $CODEDIR/create_wds_shards.py $DATADIR/train $OUTDIR/train
python3 $CODEDIR/create_wds_shards.py $DATADIR/dev $OUTDIR/dev
