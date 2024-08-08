from typing import Any
import torch
import torchaudio
from torch.optim import Optimizer
from transformers import get_scheduler
from datamodule.transforms import TextTransform

# for testing
from espnet.asr.asr_utils import add_results_to_json, get_model_conf, torch_load
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from pytorch_lightning import LightningModule

def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(
        seq1.lower().split(), seq2.lower().split()
    )


class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.backbone_args = self.cfg.model.visual_backbone
        self.text_transform = TextTransform(
            sp_model_path="./spm/unigram/unigram5000.model",
            dict_path="./spm/unigram/unigram5000_units.txt",
        )
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list), self.backbone_args)

        # -- initialise
        if self.cfg.ckpt_path:
            ckpt = torch.load(self.cfg.ckpt_path, map_location=lambda storage, loc: storage)
            if self.cfg.transfer_frontend:
                if "tcn" in self.cfg.ckpt_path:
                    tmp_ckpt = {
                        k: v
                        for k, v in ckpt["state_dict"].items()
                        if k.startswith("model.trunk.") or k.startswith("model.frontend3D.")
                    }
                    tmp_ckpt = {
                        k.replace("model.", ""): v for k, v in tmp_ckpt.items()
                    }
                    self.model.encoder.frontend.load_state_dict(tmp_ckpt)
                    print(f"Frontend loaded from {self.cfg.ckpt_path}")
                elif "transformer" in self.cfg.ckpt_path:
                    tmp_ckpt = {
                        k[7:]: v
                        for k, v in ckpt["state_dict"].items()
                        if k.startswith("stem3d.") 
                    }
                    self.model.encoder.stem3d.load_state_dict(tmp_ckpt, strict=False)

                    tmp_ckpt = {
                        k[7:]: v
                        for k, v in ckpt["state_dict"].items()
                        if k.startswith("resnet.") 
                    }
                    self.model.encoder.resnet.load_state_dict(tmp_ckpt)
                else:
                    new_state_dict = {}
                    for k, v in ckpt["state_dict"].items():
                        if k.startswith("encoder.frontend") or k.startswith("encoder.embed"):
                            new_state_dict[k] = v 
                        else:
                            pass
                    self.model.load_state_dict(new_state_dict, strict=False)
                    print(f"Frontend Model loaded from {self.cfg.ckpt_path} for {self.cfg.data.language.lower()} language")

            else:
                new_state_dict = {}
                for k, v in ckpt["state_dict"].items():
                    if k.startswith("model.decoder") or k.startswith("wav2vec") or k.startswith("category_classifier") or k.startswith("cutmix"):
                        pass
                    elif k.startswith("audio_projection"):
                        k = k.replace("audio_projection", "audio_classifier")
                        new_state_dict[k] = v
                    else:
                        new_state_dict[k] = v
                self.model.load_state_dict(new_state_dict, strict=False)
                print(f"Encoder Model loaded from {self.cfg.ckpt_path} for {self.cfg.data.language.lower()} language")
                
            
    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        do_decay = [p for p in self.parameters() if p.requires_grad and p.ndim >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.ndim < 2]
        param_groups = [{"params": do_decay}, {"params": no_decay, "weight_decay": 0.0}]

        optimizer = torch.optim.AdamW(param_groups, **self.cfg.optimizer)
        scheduler = get_scheduler(optimizer=optimizer, **self.cfg.scheduler)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, sample):
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        enc_feat, _ = self.model.encoder(sample.unsqueeze(0).to(self.device), None)
        enc_feat = enc_feat.squeeze(0)
        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted = add_results_to_json(nbest_hyps, self.token_list)
        predicted = predicted.replace("▁", " ").strip().replace("<eos>", "")
        return predicted

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="val")

    def test_step(self, sample, sample_idx):
        enc_feat, _ = self.model.encoder(
            sample["input"].unsqueeze(0).to(self.device), None
        )
        enc_feat = enc_feat.squeeze(0)
        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted = add_results_to_json(nbest_hyps, self.token_list)
        predicted = predicted.replace("▁", " ").strip().replace("<eos>", "")

        token_id = sample["target"]
        actual = self.text_transform.post_process(token_id)

        self.total_edit_distance += compute_word_level_distance(actual, predicted)
        self.total_length += len(actual.split())
        return

    def _step(self, batch, batch_idx, step_type):
        loss, loss_ctc, loss_att, loss_audio, acc = self.model(
            batch["inputs"], batch["input_lengths"], batch["audios"], batch["targets"]
        )
        batch_size = len(batch["inputs"])

        if step_type == "train":
            self.log(
                "loss", 
                loss, 
                on_step=True, 
                on_epoch=True, 
                batch_size=batch_size,
                sync_dist=True
            )
            self.log(
                "loss_ctc",
                loss_ctc,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True
            )
            self.log(
                "loss_att",
                loss_att,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True
            )
            self.log(
                "decoder_acc", 
                acc, 
                on_step=True, 
                on_epoch=True, 
                batch_size=batch_size,
                sync_dist=True
            )
            self.log(
                "loss_audio", 
                loss_audio,  
                on_step=True, 
                on_epoch=True, 
                batch_size=batch_size,
                sync_dist=True
            )
        else:
            self.log(
                "loss_val", 
                loss, 
                batch_size=batch_size,
                sync_dist=True
            )
            self.log(
                "loss_ctc_val", 
                loss_ctc, 
                batch_size=batch_size,
                sync_dist=True
            )
            self.log(
                "loss_att_val", 
                loss_att, 
                batch_size=batch_size,
                sync_dist=True
            )
            self.log(
                "decoder_acc_val", 
                acc, 
                batch_size=batch_size,
                sync_dist=True
            )
            self.log(
                "loss_audio_val", 
                loss_audio, 
                batch_size=batch_size,
                sync_dist=True
            )

        if step_type == "train":
            self.log(
                "monitoring_step", torch.tensor(self.global_step, dtype=torch.float32)
            )

        return loss

    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.loaders.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()

    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0
        self.text_transform = TextTransform(
            sp_model_path="./spm/unigram/unigram5000.model",
            dict_path="./spm/unigram/unigram5000_units.txt",
        )
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)

    def on_test_epoch_end(self):
        self.log("wer", self.total_edit_distance / self.total_length, sync_dist=True)


def get_beam_search_decoder(
    model,
    token_list,
    rnnlm=None,
    rnnlm_conf=None,
    penalty=0,
    ctc_weight=0.1,
    lm_weight=0.0,
    beam_size=40,
):
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()

    if not rnnlm:
        lm = None
    else:
        lm_args = get_model_conf(rnnlm, rnnlm_conf)
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(len(token_list), lm_args)
        torch_load(rnnlm, lm)
        lm.eval()

    scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": lm_weight,
        "length_bonus": penalty,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
