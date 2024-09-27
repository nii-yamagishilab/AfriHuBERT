source /data/users/jalabi/adaptation/layer_evals/rename_gpus.sh

DATADIR="/data/users/jalabi/Internship_NII/Data/unzipped"
OUTDIR="/data/users/jalabi/Internship_NII/Data/processed"

export NUMBA_CACHE_DIR="/data/users/jalabi/Internship_NII/Data/scripts/download/asr/cache"
echo "voxlingua"
# voxlingua
# Uses 16kHz
for lan in ny_mw  om_et  pt_br  rw_rw  sn_zw  so_so  sw_ke  umb_ao  wo_sn  xh_za  yo_ng  zu_za; do
	# af_za  am_et  ar_eg  en_us  ff_sn  fr_fr  ha_ng  ig_ng  kam_ke  lg_ug  ln_cd  luo_ke  nso_za  ny_mw  om_et  pt_br  rw_rw  sn_zw  so_so  sw_ke  umb_ao  wo_sn  xh_za  yo_ng  zu_za; do
	echo $lan
	for split in train dev test; do
		# echo $lan
		fiDATA="$DATADIR/fleurs_lid/$split/$lan"
		# 16357386827603545.wav
		fOUTDIR="$OUTDIR/fleurs_lid/$split/$lan"
		mkdir -p $fOUTDIR
		for x in $fiDATA/*.wav;do
			b=${x##*/}
			filename="${b%.*}"
			# just resample
			# sox  $x -b 16 -c 1 -r 16000 $fOUTDIR/$filename.wav
			python3 /data/users/jalabi/Internship_NII/Data/scripts/download/asr/resample.py \
				--input $x \
				--channel 1 \
				--rate 16000 \
				--output $fOUTDIR/$filename.wav
			# echo "$OUTDIR/voxlingua/$lan/$filename.wav" >> $vxOUTDIR/filenames_$lan.txt
		done
	done
done
# rm -r $DATADIR/voxlingua/
