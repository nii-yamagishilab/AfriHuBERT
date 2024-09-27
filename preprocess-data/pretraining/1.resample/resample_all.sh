DATADIR="/data/users/jalabi/Internship_NII/Data/unzipped"
OUTDIR="/data/users/jalabi/Internship_NII/Data/processed"

SR=16000
BITS=16
CHANNEL=1

# -------- BibleTTS ---------------
# BIble TTS
# Uses 48kHz
echo "Bible TTS"
for lan in akuapem-twi asante-twi  ewe  hausa  lingala  yoruba; do
        echo $lan
        for split in train dev test; do
                btOUTDIR="$OUTDIR/BibleTTS"
                mkdir -p $btOUTDIR/tmp
                for x in $DATADIR/BibleTTS/$lan/$split/*/*.flac;do
                        b=${x##*/}
                        book=$(basename $(dirname $x))
                        mkdir -p $btOUTDIR/$lan/$split/$book
                        filename="${b%.*}"
                        #echo $b
                        # convert to wav 
                        # then downsample 
                        ffmpeg -y -loglevel error  -i $x $btOUTDIR/tmp/$filename.wav
                        sox $btOUTDIR/tmp/$filename.wav -b $BITS -c $CHANNEL -r $SR $btOUTDIR/$lan/$split/$book/$filename.wav
                        # sox $btOUTDIR/tmp_sample/$filename.wav  $btOUTDIR/$lan/$filename.wav silence 1 0.1 1% 1 0.1 1%
                        rm $btOUTDIR/tmp/$filename.wav
                        # $btOUTDIR/tmp_sample/$filename.wav
                        echo "$btOUTDIR/$lan/$split/$book/$filename.wav" >> $btOUTDIR/$lan/filenames_$split.txt
                done

        done
done


# -------- KALLAAMA ---------------
declare -A language_dict
language_dict=(['fuc']="pulaar" ['srr']="sereer" ['wol']="wolof")

echo "kallaama"
for lan in fuc srr wol; do
        ncOUTDIR="$OUTDIR/kallaama/clean_dataset_ready4release"
        ncDATADIR="$DATADIR/kallaama/clean_dataset_ready4release"
        mkdir -p $ncOUTDIR/$lan

        if [ $lan == "fuc" ]; then
                upper_limit=39
        elif [ $lan == "srr" ]; then
                upper_limit=40
        else
                upper_limit=68
        fi

        for i in $(seq 1 $upper_limit); do
                # Your code here
                for x in $ncDATADIR/"${language_dict[$lan]}"/speech_dataset/$i/*.wav;do
                        mkdir -p $ncOUTDIR/"${language_dict[$lan]}"/speech_dataset/$i/
                        b=${x##*/}
                        filename="${b%.*}"
                        # then downsample 
                        sox $x -b $BITS -c $CHANNEL -r $SR $ncOUTDIR/"${language_dict[$lan]}"/speech_dataset/$i/$filename.wav
                        echo "$ncOUTDIR/${language_dict[$lan]}/speech_dataset/$i/$filename.wav" >> $ncOUTDIR/filenames_$lan.txt
                done

        done
done


# -------- MCV ---------------
# Uses 48kHz
echo "MCV"
for lan in rw; do
        # sw lg rw; do
        echo $lan
        for split in train dev; do
                btOUTDIR="$OUTDIR/MCV"
                mkdir -p $btOUTDIR/tmp
                mkdir -p $btOUTDIR/$lan
                for x in $DATADIR/MCV/$lan/${lan}_${split}_*/*.mp3; do
                        b=${x##*/}
                        filename="${b%.*}"
                        echo $b
                        # category="$(cut -d'/' -f10 <<<  $x)"
                        category=$(echo "$x" | cut -d'/' -f10)
                        mkdir -p $btOUTDIR/$lan/$category
                        # convert to wav 
                        # then downsample 
                        ffmpeg -y -loglevel error  -i $x $btOUTDIR/tmp/$filename.wav
                        sox $btOUTDIR/tmp/$filename.wav -b $BITS -c $CHANNEL -r $SR $btOUTDIR/$lan/$category/$filename.wav
                        rm $btOUTDIR/tmp/$filename.wav
                        # $btOUTDIR/tmp_sample/$filename.wav
                        if [ "$split" == "dev" ] || [ "$split" == "test" ]
                        then
                                echo "$btOUTDIR/$lan/$category/$filename.wav" >> $btOUTDIR/filenames2_$split.txt
                        else
                            echo "$btOUTDIR/$lan/$category/$filename.wav" >> $btOUTDIR/filenames2.txt
                        fi
                done

        done
done



# -------- Nicolingua ---------------
# nicolingua
# Uses 16kHz
echo "nicolingua"
for split in train valid; do
        ncOUTDIR="$OUTDIR/nicolingua/nicolingua-0003-west-african-radio-corpus"
        ncDATADIR="$DATADIR/nicolingua/nicolingua-0003-west-african-radio-corpus"
        mkdir -p $ncOUTDIR/$split
        for x in $DATADIR/nicolingua/nicolingua-0003-west-african-radio-corpus/data/$split/audio_samples/*.wav;do
                b=${x##*/}
                filename="${b%.*}"
                # then downsample 
                sox $x -b $BITS -c $CHANNEL -r $SR $ncOUTDIR/$split/$filename.wav
                #sox $ncOUTDIR/tmp_sample/$filename.wav  $ncOUTDIR/$lan/$filename.wav silence 1 0.1 1% 1 0.1 1%
                #rm $ncOUTDIR/tmp_sample/$filename.wav
                echo "$ncOUTDIR/$split/$filename.wav" >> $ncOUTDIR/filenames_$split.txt


        done
done
rm -r $DATADIR/nicolingua/nicolingua-0003-west-african-radio-corpus/

# -------- NCHLT ---------------
echo "nchlt"
for lan in nbl nso sot ssw afr nbl nso sot ssw tsn tso ven xho zul; do
        for cat in baseline aux1 aux2; do
                echo $lan $cat
                #ncOUTDIR=$OUTDIR/NCHLT/$cat/nchlt_$lan/audio/
                if [ "$cat" == "baseline" ]; then
                        ncOUTDIR=$OUTDIR/NCHLT/$cat/nchlt_$lan/audio
                        mkdir -p $ncOUTDIR
                        for x in $DATADIR/NCHLT/$cat/nchlt_$lan/audio/*/*.wav; do
                                # baseline /data/users/jalabi/Internship_NII/Data/unzipped/NCHLT/$cat/nchlt_$lan/audio/*/*.wa
                                b=${x##*/}
                                filename="${b%.*}"
                                book=$(basename $(dirname $x))
                                if [ "$book" -ge 500 ] && [ "$book" -le 507 ]; then
                                        :
                                else
                                        mkdir -p $ncOUTDIR/$book
                                        sox $x -b $BITS -c $CHANNEL -r $SR $ncOUTDIR/$book/$filename.wav
                                        echo "$ncOUTDIR/$book/$filename.wav" >> $OUTDIR/NCHLT/$cat/filenames_$lan.txt
                                fi
                        done
                else
                        ncOUTDIR=$OUTDIR/NCHLT/$cat/$lan-$cat/$lan/audio
                        mkdir -p $ncOUTDIR
                        for x in $DATADIR/NCHLT/$cat/$lan-$cat/$lan/audio/*/*.wav; do
                                b=${x##*/}
                                filename="${b%.*}"
                                book=$(basename $(dirname $x))
                                mkdir -p $ncOUTDIR/$book
                                sox $x -b $BITS -c $CHANNEL -r $SR $ncOUTDIR/$book/$filename.wav
                                echo "$ncOUTDIR/$book/$filename.wav" >> $OUTDIR/NCHLT/$cat/filenames_$lan.txt
                        done
                fi
        done
done


# -------- VoxLingua107 ---------------
echo "voxlingua"
# voxlingua
# Uses 16kHz
for lan in af am  ar  en  fr  ha  ln  mg  pt  so  sw sn  yo; do
        echo $lan
        vxOUTDIR="$OUTDIR/voxlingua2"
        mkdir -p $vxOUTDIR/tmp_sample
        mkdir -p $vxOUTDIR/$lan
        for x in $DATADIR/voxlingua/$lan/*.wav;do
                b=${x##*/}
                filename="${b%.*}"
                #echo $b
                # just resample
                sox  $x -b $BITS -c $CHANNEL -r $SR $vxOUTDIR/$lan/$filename.wav
                # sox $vxOUTDIR/tmp_sample/$filename.wav $vxOUTDIR/$lan/$filename.wav silence 1 0.1 1% 1 0.1 1%
                # rm $vxOUTDIR/tmp_sample/$filename.wav
                echo "$OUTDIR/voxlingua/$lan/$filename.wav" >> $vxOUTDIR/filenames_$lan.txt
        done
done
rm -r $DATADIR/voxlingua/
