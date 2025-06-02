RAWDATA="/data/users/jalabi/Internship_NII/Data/raw"
uDATADIR="/data/users/jalabi/Internship_NII/Data/unzipped"

# -------- BibleTTS ---------------
# Download and Unzip Bible TTS
DATADIR="$RAWDATA/BibleTTS"
wget -O $DATADIR/akuapem-twi.tgz https://www.openslr.org/resources/129/akuapem-twi.tgz
wget -O $DATADIR/asante-twi.tgz https://www.openslr.org/resources/129/asante-twi.tgz
wget -O $DATADIR/ewe.tgz https://www.openslr.org/resources/129/ewe.tgz
wget -O $DATADIR/hausa.tgz https://www.openslr.org/resources/129/hausa.tgz
wget -O $DATADIR/lingala.tgz https://www.openslr.org/resources/129/lingala.tgz
wget -O $DATADIR/yoruba.tgz https://www.openslr.org/resources/129/yoruba.tgz

# 
unDATADIR="$uDATADIR/BibleTTS/"
mkdir -p $uDATADIR
tar -xzvf $DATADIR/akuapem-twi.tgz -C $uDATADIR
tar -xzvf $DATADIR/asante-twi.tgz -C $uDATADIR
tar -xzvf $DATADIR/ewe.tgz -C $uDATADIR
tar -xzvf $DATADIR/hausa.tgz -C $uDATADIR
tar -xzvf $DATADIR/lingala.tgz -C $uDATADIR 
tar -xzvf $DATADIR/yoruba.tgz -C $uDATADIR


# -------- Nicolingua ---------------

DATADIR="$RAWDATA/nicolingua"
mkdir -p $DATADIR
#wget -O $DATADIR/nicolingua-0004-west-african-va-asr-corpus.tgz https://www.openslr.org/resources/106/nicolingua-0004-west-african-va-asr-corpus.tgz
wget -O $DATADIR/nicolingua-0003-west-african-radio-corpus.tgz https://www.openslr.org/resources/105/nicolingua-0003-west-african-radio-corpus.tgz

unDATADIR="$uDATADIR/nicolingua"
mkdir -p $unDATADIR
#tar -xzvf $DATADIR/nicolingua-0004-west-african-va-asr-corpus.tgz -C $ODATADIR
tar -xzvf $DATADIR/nicolingua-0003-west-african-radio-corpus.tgz -C $unDATADIR


# -------- KALLAAMA ---------------

DATADIR="$RAWDATA/kallaama"
mkdir -p $DATADIR
wget -O $DATADIR/fuc.tar.gz  https://zenodo.org/records/10892569/files/speech_dataset_fuc.tar.gz
wget -O $DATADIR/srr.tar.gz  https://zenodo.org/records/10892569/files/speech_dataset_srr.tar.gz
wget -O $DATADIR/wol.tar.gz  https://zenodo.org/records/10892569/files/speech_dataset_wol.tar.gz


unDATADIR="uDATADIR/kallaama/"
mkdir -p $unDATADIR
for lan in fuc srr wol; do
	        tar -xzvf $DATADIR/$lan.tar.gz -C $unDATADIR
done

# -------- NAIJAVOICES---------------
# huggingface-cli login
DATADIR="$RAWDATA/naijavoices"
git clone https://huggingface.co/datasets/naijavoices/naijavoices-dataset-compressed/ $DATADIR
zip -F $/DATADIR/naijavoices-dataset-compressed/audio-files.zip --out $DATADIR/naijavoices-dataset-compressed/naija-audio.zip
unDATADIR="uDATADIR/naijavoices"
mkdir -p $unDATADIR
unzip $DATADIR/naijavoices-dataset-compressed/naija-audio.zip -d $unDATADIR/naijavoices/

# -------- ZAMBEZI ---------------

DATADIR="$RAWDATA/zambezi/"
mkdir -p $DATADIR
wget -O $DATADIR/nya.zip https://zenodo.org/record/7546317/files/nya.zip
wget -O $DATADIR/toi.zip https://zenodo.org/record/7543819/files/toi.zip?download=1
wget -O $DATADIR/loz.zip https://zenodo.org/record/7544601/files/loz.zip?download=1
wget -O $DATADIR/bem.zip https://zenodo.org/record/7540277/files/bem.zip?download=1
wget -O $DATADIR/lun.zip https://zenodo.org/record/7589496/files/lun.zip?download=1

unDATADIR="$uDATADIR/zambezi"
mkdir -p $unDATADIR
for lan in bem nya toi loz lun; do
        #unzip $DATADIR/$lan.zip  -d $tDATADIR/
        jar -xvf $DATADIR/$lan.zip
        mv $lan $unDATADIR/
done

# -------- NCHLT ---------------
OUTDIR=/data/users/jalabi/Internship_NII/Data/unzipped/NCHLT
for cat in baseline aux1 aux2; do
        for lan in afr nbl nso sot ssw tsn tso ven xho zul; do
               mkdir -p $OUTDIR/$cat
               if [ "$cat" == "baseline" ]; then
                       if [ "$lan" == "zul" ]; then
                               unzip /data/users/jalabi/Internship_NII/Data/raw/NCHLT/$cat/nchlt.speech.corpus.$lan.zip -d $OUTDIR/$cat/nchlt_$lan
                       else
                                unzip /data/users/jalabi/Internship_NII/Data/raw/NCHLT/$cat/nchlt.speech.corpus.$lan.zip -d $OUTDIR/$cat/
                       fi
               else
                      echo "unzip $lan from $cat"
                      #unzip /data/users/jalabi/Internship_NII/Data/raw/NCHLT/$cat/$lan-$cat.zip -d $OUTDIR/$cat/
                      cd $OUTDIR/$cat/  && jar -xvf /data/users/jalabi/Internship_NII/Data/raw/NCHLT/$cat/$lan-$cat.zip

               fi
       done
done

# -------- VoxLingua ---------------
DATADIR="$RAWDATA/voxlingua"
mkdir -p $DATADIR
for lan in am af ha ln mg so sw yo ar en fr pt sn yo; do
        wget -O $DATADIR/$lan.zip https://bark.phon.ioc.ee/voxlingua107/$lan.zip
done

unDATADIR="$uDATADIR/voxlingua"
mkdir -p $unDATADIR
for lan in am af ha ln mg so sw yo ar en fr pt sn yo; do
        unzip $DATADIR/$lan.zip  -d $unDATADIR/
done


# -------- VoxLingua ---------------
# get MCV
DATADIR="$RAWDATA/MCV"
unDATADIR="$uDATADIR/MCV"
mkdir -p $unDATADIR

for lan in lg sw rw; do
        echo $lan
        mkdir -p $unDATADIR/$lan
        for split in train dev; do
                for file in `ls $DATADIR/$lan/${lan}_${split}_*.tar`; do
                        echo $file >> mvfilenames.log
                        tar -xf $file -C $unDATADIR/$lan/ ;
                done
                # tar -xf $DATADIR/$lan/${lan}_${split}_*.tar -C $uDATADIR/$lan/
        done
done
