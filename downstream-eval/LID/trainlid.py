#!/usr/bin/python3
"""Recipe for training language embeddings using the VoxLingua107 Dataset.

This recipe is heavily inspired by this: https://github.com/nikvaessen/speechbrain/tree/sharded-voxceleb/my-recipes/SpeakerRec

To run this recipe, use the following command:
> python train_lang_embeddings_wds.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:

    hparams/train_epaca_tdnn_wds.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
    * Tanel Alumäe 2021
    * @nikvaessen
"""
import json
import logging
import os
import random
import sys
from functools import partial
from typing import Dict

import torch
import webdataset as wds
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch

logger = logging.getLogger(__name__)


class LanguageBrain(sb.core.Brain):
    """Class for language ID training" """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        #print("Data shpae before augmentation = ", wavs.shape)
        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, lens = self.hparams.wav_augment(wavs, lens)

        # Feature extraction and normalization

        outputs = self.modules.wav2vec2(wavs,lens)
        outputs = outputs.transpose(1, 2)

        pooling = self.modules.attentive(outputs, lens)
        pooling = pooling.transpose(1, 2)
        outputs = self.modules.classifier(pooling)
        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label."""
        predictions, _ = predictions
        uttid = batch.id
        langid = batch.lang_id_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            langid = self.hparams.wav_augment.replicate_labels(langid)
        
        """to meet the input form of nll loss"""
        loss = self.hparams.compute_cost(predictions, langid.unsqueeze(1))

        if hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, langid.unsqueeze(1))

        # Compute Accuracy using MetricStats
        self.acc_metric.append(
            uttid, predict=predictions.squeeze(), target=langid.unsqueeze(0), lengths=None
        )

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Compute Accuracy using MetricStats
        # Define function taking (prediction, target, length) for eval
        def accuracy_value(predict, target, lengths=None):
            """Computes Accuracy"""
            nbr_correct, nbr_total = sb.utils.Accuracy.Accuracy(
                predict, target, lengths
            )
            acc = torch.tensor([nbr_correct / nbr_total])
            return acc
        self.acc_metric = sb.utils.metric_stats.MetricStats(
            metric=accuracy_value, n_jobs=1
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()


    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_acc = self.acc_metric.summarize("average")

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize("average"),
                "error_rate": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch) #stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            (
                old_lr_wav2vec2,
                new_lr_wav2vec2,
            ) = self.hparams.lr_annealing_wav2vec2(stats["loss"]) #error_rate"])
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec2_optimizer, new_lr_wav2vec2
            )

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr, "wave2vec_lr": old_lr_wav2vec2},
                train_stats={"loss": self.train_loss, "acc": self.train_acc},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats, min_keys=["error_rate"]
            )

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec2_optimizer = self.hparams.wav2vec2_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec2_opt", self.wav2vec2_optimizer
            )
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

        self.optimizers_dict = {
            "model_optimizer": self.optimizer,
            "wav2vec2_optimizer": self.wav2vec2_optimizer,
        }


def dataio_prep_shards(hparams):
    # load the meta info json file
    with wds.gopen(hparams["train_meta"], "rb") as f:
        train_meta = json.load(f)
    with wds.gopen(hparams["val_meta"], "rb") as f:
        val_meta = json.load(f)

    # define the mapping functions in the data pipeline
    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_iterables=[train_meta["language_ids"]],
        output_key="lang_id",
    )
    # breakpoint()

    def audio_pipeline(sample_dict: Dict, random_chunk=True):
        key = sample_dict["__key__"]
        language_id = sample_dict["language_id"].decode("ascii")
        audio_tensor = sample_dict["audio.pth"]

        # determine what part of audio sample to use
        audio_tensor = audio_tensor.squeeze()

        if random_chunk:
            if len(audio_tensor) - snt_len_sample - 1 <= 0:
                start = 0
            else:
                start = random.randint(
                    0, len(audio_tensor) - snt_len_sample - 1
                )

            stop = start + snt_len_sample
        else:
            start = 0
            stop = len(audio_tensor)

        sig = audio_tensor[start:stop]

        # determine the language ID of the sample
        lang_id_idx = label_encoder.encode_label(language_id)

        return {
            "sig": sig,
            "lang_id_encoded": lang_id_idx,
            "id": key,
        }

    train_data = (
        wds.WebDataset(
            hparams["train_shards"], cache_dir=hparams["shard_cache_dir"]
        )
        .repeat()
        .shuffle(1000)
        .decode("pil")
        .map(partial(audio_pipeline, random_chunk=True))
    )
    logger.info(
        f"Training data consist of {train_meta['num_data_samples']} samples"
    )

    valid_data = (
        wds.WebDataset(
            hparams["val_shards"], cache_dir=hparams["shard_cache_dir"]
        )
        .decode("pil")
        .map(partial(audio_pipeline, random_chunk=False))
    )
    logger.info(
        f"Validation data consist of {val_meta['num_data_samples']} samples"
    )

    return (
        train_data,
        valid_data,
        train_meta["num_data_samples"],
        val_meta["num_data_samples"],
    )


if __name__ == "__main__":
    logger.info("Starting training...")
    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Data preparation for augmentation
    sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])
    sb.utils.distributed.run_on_main(hparams["prepare_rir_data"])

    (
        train_data,
        valid_data,
        num_train_samples,
        num_valid_samples,
    ) = dataio_prep_shards(hparams)

    print("The number of training sample = ", num_train_samples, ". The number of Valid examples = ", num_valid_samples)
    # add collate_fn to dataloader options
    hparams["train_dataloader_options"]["collate_fn"] = PaddedBatch
    hparams["val_dataloader_options"]["collate_fn"] = PaddedBatch

    hparams["train_dataloader_options"]["looped_nominal_epoch"] = (
        num_train_samples // hparams["train_dataloader_options"]["batch_size"]
    )

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    hparams["wav2vec2"] = hparams["wav2vec2"].to(device=run_opts["device"])
    # freeze the feature extractor part when unfreezing
    if not hparams["freeze_wav2vec2"] and hparams["freeze_wav2vec2_conv"]:
        hparams["wav2vec2"].model.feature_extractor._freeze_parameters()

    # Brain class initialization
    language_brain = LanguageBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    language_brain.fit(
        language_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["val_dataloader_options"],
    )
