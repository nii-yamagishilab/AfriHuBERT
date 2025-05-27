source AfriHuBERT/downstream-eval/rename_gpus.sh
CODEDIR="AfriHuBERT/downstream-eval/LID"
DATADIR="Data/processed/fleurs_lid" # change according 
OUTDIR="$CODEDIR/data"

mkdir -p $OUTDIR
#python3 $CODEDIR/create_wds_shards.py $DATADIR/train $OUTDIR/train
#python3 $CODEDIR/create_wds_shards.py $DATADIR/dev $OUTDIR/dev
OUTDIR="$CODEDIR/data_unif_lid"
mkdir -p $OUTDIR
python3 $CODEDIR/create_wds_shards2.py train_lid.tsv  $OUTDIR/train
python3 $CODEDIR/create_wds_shards2.py dev.tsv $OUTDIR/dev
