# AfriHuBERT: A self-supervised speech representation model for African languages

<a href='https://arxiv.org/abs/2409.20201'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

## Introduction
This is the code for the AfriHuBERT submitted to the [Interspeech 2025](https://arxiv.org/abs/2409.20201). AfriHuBERT is a compact multilingual self-supervised speech encoder based on mHuBERT-147. We performed continued pretraining through multilingual adaptive finetuning (MAFT) on over 10,000 hours of African languages' data aggregated from various sources. This model can be considered the fourth iteration of mHuBERT-147, specifically trained on African languages. According to the paper, this is the **AfriHuBERT-*n*** model. You can click [here](https://huggingface.co/ajesujoba/AfriHuBERTo) for the **AfriHuBERT-*o*** model.


## Pretraining data
- Dataset: AfriHuBERT was trained on data from 11 major sources, including BibleTTS, Kallaama, MMS Ulab v2, NaijaVoices, and NCHLT. All sources and their licenses are shown in the table below. Please refer to the paper for more information.

<br>
<p align="center">
    <img src="images/afrihubert_sources.png" width="95%"> <br>
    Overview
</p>
<be>
      
## Language Coverage
AfriHuBERT covers 1,230 languages in total including 1,226 indigenous African languages

## Acknowledgements
This work was conducted during the first author’s internship at NII, Japan. This study is partially supported by JST AIP Acceleration Research (JPMJCR24U3). Part of this study was carried out using the TSUBAME4.0 supercomputer at the Institute of Science Tokyo. Also, we thank Xin Wang, Badr M. Abdullah, Siyang Wang, Wanying Ge, David Adelani, and Aravind Krishnan for their helpful feedback.

## BibTeX entry and citation info.
```
@misc{alabi2024afrihubertselfsupervisedspeechrepresentation,
      title={AfriHuBERT: A self-supervised speech representation model for African languages}, 
      author={Jesujoba O. Alabi and Xuechen Liu and Dietrich Klakow and Junichi Yamagishi},
      year={2024},
      eprint={2409.20201},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.20201}, 
}

```
