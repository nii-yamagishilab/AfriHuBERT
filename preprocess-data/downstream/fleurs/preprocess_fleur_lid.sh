DATADIR="/data/users/jalabi/Internship_NII/Data/raw"
OUTDIR="/data/users/jalabi/Internship_NII/Data/unzipped"

# This for LID

for lan in rw_rw; do
	#af_za am_et  ar_eg  en_us  ff_sn  fr_fr  ha_ng  ig_ng  kam_ke  lg_ug  ln_cd  luo_ke  nso_za  ny_mw  om_et  pt_br  rw_rw  sn_zw so_so  sw_ke  umb_ao  wo_sn  xh_za  yo_ng  zu_za; do
	echo $lan

	mkdir -p $OUTDIR/fleurs_lid/$split/$lan
	if [ "$lan" == "rw_rw" ]; then
		for split in train dev test; do
			# /data/users/jalabi/Internship_NII/Data/raw/fleurs/rw_rw/
			tar -xf $DATADIR/fleurs/$lan/audio/$split.tar.xz 
			# -C $OUTDIR/fleurs_lid/$split/$lan
			mv ./${split}_data/*.wav $OUTDIR/fleurs_lid/$split/$lan
			rm -r ./${split}_data/
		done
	else:
		for split in train dev test; do
			tar -xzf $DATADIR/fleurs/$lan/audio/$split.tar.gz -C $OUTDIR/fleurs_lid/$split/$lan
			mv $OUTDIR/fleurs_lid/$split/$lan/$split/*.wav $OUTDIR/fleurs_lid/$split/$lan
			rm -r $OUTDIR/fleurs_lid/$split/$lan/$split/
		done
	fi

done
