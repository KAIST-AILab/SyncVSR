from __future__ import annotations

from typing import Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import OneCycleLR
import timm
from timm.optim import create_optimizer_v2

from transformers import (
    BertConfig,
    BertModel,
    get_scheduler,
    Wav2Vec2ForPreTraining
)
import pytorch_lightning as pl
from utils import check_availability
from omegaconf import DictConfig
from x_transformers import Encoder
if check_availability("fairseq"):
    from fairseq.checkpoint_utils import load_model_ensemble, load_model_ensemble_and_task

from augment import CutMix
from tcn.model import Lipreading


class CustomIdentity(nn.Module):
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return x


class TransformerLightningModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.is_train = False
        self.word_labels = self.config.model.bert.num_labels
        self.lambda_audio = config.optim.lambda_audio
        self.label_smoothing = config.train.label_smoothing
        self.use_wb = config.data.use_word_boundary
        self.emb_dropout_bert = nn.Dropout(config.model.bert.emb_dropout)

        extra_dim = 1 if self.use_wb else 0

        self.stem3d = nn.Sequential(
            nn.Conv3d(1, 64, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
        )
        self.resnet = timm.create_model(config.model.resnet)

        # Cross Modal Sync Related
        if "vq" in config.model.wav2vec.path: 
            self.codec = "vq"
            self.audio_alignment = 4
            self.vq_groups = 2
            self.audio_vocab_size = 320 # or metadata.model.vq_vars
        elif "wav2vec2" in config.model.wav2vec.path:
            self.codec = "wav2vec2"
            self.audio_alignment = 2
            self.vq_groups = 2
            self.audio_vocab_size = 640
        
        if "vq" in config.model.wav2vec.path and check_availability("fairseq"): # vq-wav2vec
            wav2vec, metadata = load_model_ensemble([config.model.wav2vec.path])            
            self.wav2vec = wav2vec[0].requires_grad_(False).eval()
        elif "wav2vec2" in config.model.wav2vec.path: # wav2vec2
            wav2vec = Wav2Vec2ForPreTraining.from_pretrained(config.model.wav2vec.path)
            del wav2vec.wav2vec2.encoder # remove transformer encoder blocks
            wav2vec = wav2vec.requires_grad_(False).eval()
            codevectors = torch.arange(wav2vec.quantizer.codevectors.size(1))
            codevectors = codevectors.view(1, -1, 1).expand_as(wav2vec.quantizer.codevectors)
            wav2vec.quantizer.codevectors.data = codevectors.float()
            self.wav2vec = wav2vec

        self.lambda_audio = config.optim.lambda_audio
        self.audio_projection = nn.Linear(config.model.bert.dim+extra_dim, self.audio_alignment * self.vq_groups * self.audio_vocab_size)  # 768 -> 4 * 2 * 320 or 2 * 2 * 640
        print(f"using {self.codec} neural audio codec")

        self.cutmix = CutMix(
            self.word_labels,
            self.wav2vec if check_availability("fairseq") else None,
        ).eval()

        if config.model.bert.type == "huggingface":
            print("using huggingface bert implementation")
            self.encoder = BertModel(BertConfig(**config.model.bert))
        elif config.model.bert.type == "x-transformers":
            print("using x-transformers bert implementation")
            self.encoder = Encoder(
                dim=config.model.bert.dim+extra_dim,
                depth=config.model.bert.depth,
                heads=config.model.bert.heads,
                attn_dropout=config.model.bert.attn_dropout,
                layer_dropout=config.model.bert.layer_dropout,
                ff_dropout=config.model.bert.ff_dropout,
                use_rmsnorm=config.model.bert.use_rmsnorm,
                ff_glu=config.model.bert.ff_glu,
                rotary_pos_emb=config.model.bert.rotary_pos_emb,
            )

        self.category_classifier = nn.Linear(config.model.bert.dim+extra_dim, config.model.bert.num_labels)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.model.bert.dim + extra_dim))
        if self.use_wb:    
            self.cls_token.data[0, 0, -1] = 0.0  # [CLS] token is not included in word_mask
    
    def forward_videos(self, videos: torch.Tensor) -> torch.Tensor:
        hidden = self.stem3d(videos).transpose(1, 2).flatten(0, 1)
        hidden = self.resnet.layer1(hidden)
        hidden = self.resnet.layer2(hidden)
        hidden = self.resnet.layer3(hidden)
        hidden = self.resnet.layer4(hidden)
        hidden = hidden.mean((2, 3)).unflatten(0, (videos.size(0), -1))
        return hidden

    def forward_audios(self, audios: torch.Tensor) -> torch.Tensor:
        """ quantize raw waveform into audio tokens """
        if self.codec == None or "vq" in self.codec.lower():
            audio_tokens = self.wav2vec.feature_extractor(audios)
            audio_tokens = self.wav2vec.vector_quantizer.forward_idx(audio_tokens)[1]
            return audio_tokens
        elif "wav2vec2" in self.codec.lower():
            feats = self.wav2vec.wav2vec2.feature_extractor(audios).transpose(1, 2)
            _, feats = self.wav2vec.wav2vec2.feature_projection(feats)
            audio_tokens = self.wav2vec.quantizer(feats)[0].unflatten(-1, (2, -1))[..., 0].long()
            return audio_tokens
    
    def forward(
        self,
        videos: torch.Tensor,
        audio_tokens: torch.Tensor,
        labels: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        # Video embedding
        inputs_embeds = self.forward_videos(videos)
        
        # Word Boundary information if available. Increases hidden dimension by 1.
        inputs_embeds = torch.cat((inputs_embeds, word_mask.unsqueeze(-1)), dim=-1) if self.use_wb else inputs_embeds
        
        B, seq_len, dim = inputs_embeds.shape
        audio_tokens = audio_tokens[:, : seq_len * self.audio_alignment]
        cls_tokens = self.cls_token.expand(videos.size(0), -1, -1)
        inputs_embeds = self.emb_dropout_bert(torch.cat((cls_tokens, inputs_embeds), dim=1))

        if self.config.model.bert.type == "huggingface":
            last_hidden_state = self.encoder(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
            ).last_hidden_state
        elif self.config.model.bert.type == "x-transformers":
            last_hidden_state = self.encoder(inputs_embeds)

        # Word Classification Loss
        logits_category = self.category_classifier(last_hidden_state[:, 0, :])
        logits_category = logits_category.float() # converting into float type before the loss calculation
        loss_category = F.cross_entropy(
            logits_category, labels, label_smoothing=self.label_smoothing
        )
        
        # Audio Reconstruction Loss
        logits_audio = self.audio_projection(last_hidden_state[:, 1:, :])
        logits_audio = logits_audio.float() # converting into float type before the loss calculation
        logits_audio = logits_audio.reshape(B, seq_len, self.audio_alignment * self.vq_groups, self.audio_vocab_size)
        loss_audio = F.cross_entropy(logits_audio.reshape(-1, self.audio_vocab_size), audio_tokens.flatten())
        
        # Composite Loss
        loss_total = loss_category + loss_audio * self.lambda_audio

        # Accuracy Metric
        if self.config.train.use_cutmix and self.training:
            label_args = labels.argmax(dim=-1)
            corrects = logits_category.topk(5, dim=1)[1] == label_args.unsqueeze(1)
        else:
            corrects = logits_category.topk(5, dim=1)[1] == labels.unsqueeze(1)
        accuracy_top1 = corrects[:, 0].float().mean()
        accuracy_top5 = corrects.float().amax(1).mean()

        return {
            "loss_total": loss_total,
            "loss_category": loss_category,
            "loss_audio": loss_audio,
            "accuracy_top1": accuracy_top1,
            "accuracy_top5": accuracy_top5,
        }


    def training_step(self, batch: tuple[torch.Tensor, ...], idx: int) -> torch.Tensor:
        self.is_train = True
        if self.config.train.use_cutmix:
            batch = self.cutmix(*batch)
        else:
            batch[1] = self.forward_audios(batch[1]) if check_availability("fairseq") else batch[1]
        metrics = self(*batch)
        self.log_dict({f"train/{k}": v for k, v in metrics.items()})
        return metrics["loss_total"]

    def validation_step(self, batch: tuple[torch.Tensor, ...], idx: int):
        self.is_train = False
        batch[1] = self.forward_audios(batch[1]) if check_availability("fairseq") else batch[1]
        metrics = self(*batch)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, sync_dist=True)

    def test_step(self, batch: tuple[torch.Tensor, ...], idx: int):
        self.is_train = False
        batch[1] = self.forward_audios(batch[1]) if check_availability("fairseq") else batch[1]
        metrics = self(*batch)
        self.log_dict({f"test/{k}": v for k, v in metrics.items()}, sync_dist=True)

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        do_decay = [p for p in self.parameters() if p.requires_grad and p.ndim >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.ndim < 2]
        param_groups = [{"params": do_decay}, {"params": no_decay, "weight_decay": 0.0}]

        optimizer = AdamW(param_groups, **self.config.optim.optimizer)
        scheduler = get_scheduler(optimizer=optimizer, **self.config.optim.scheduler)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class DCTCNLightningModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.mixup_alpha = config.optim.mixup_alpha
        self.audio_alignment = config.model.wav2vec.alignment
        self.lambda_audio = config.optim.loss_audio_weight
        self.label_smoothing = config.train.label_smoothing

        # 3D Conv + 2D Resnet + DC-TCN
        self.model = Lipreading(**config.model.dctcn)
        self.video_classifier = self.model.tcn.tcn_output
        self.model.tcn.consensus_func = CustomIdentity()
        self.model.tcn.tcn_output = nn.Identity()

        # Wav2Vec Codec
        wav2vec, metadata = load_model_ensemble([config.model.wav2vec.path])
        self.wav2vec = wav2vec[0].requires_grad_(False).eval()
        self.audio_vocab_size = metadata.model.vq_vars
        self.audio_projection = nn.Linear(
            1664,
            self.audio_alignment * self.vq_groups * self.audio_vocab_size,
        )

        # kicks in default weight initialization
        for module in self.model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(
        self,
        videos: torch.Tensor,
        audios: torch.Tensor,
        labels: torch.Tensor,
        word_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # mixup augmentation
        lam = 0.0
        if self.mixup_alpha > 0 and self.training:
            lam = 0.5 - np.abs(0.5 - np.random.beta(self.mixup_alpha, self.mixup_alpha))

        last_hidden_states = self.model(
            torch.lerp(videos, videos.roll(1, 0), lam),
            lengths=None,
            # We do not apply mixup on word boundaries
            boundaries=word_mask.unsqueeze(2),
        )
        logits_category = (last_hidden_states * attention_mask.unsqueeze(1)).sum(2)
        logits_category = logits_category / (attention_mask.sum(1, keepdim=True) + 1e-6)
        logits_category = self.video_classifier(logits_category)

        loss_category_a = F.cross_entropy(logits_category, labels)
        loss_category_b = F.cross_entropy(logits_category, labels.roll(1, 0))

        # FORWARD AUDIO
        audio_tokens = self.wav2vec.feature_extractor(audios)
        audio_tokens = self.wav2vec.vector_quantizer.forward_idx(audio_tokens)[1]
        audio_tokens = audio_tokens[:, : videos.size(2) * self.audio_alignment]

        # Audio Alignment Loss
        logits_audio = self.audio_projection(last_hidden_states.transpose(1, 2))
        logits_audio = logits_audio.unflatten(2, (-1, self.audio_vocab_size))
        loss_audio_a = F.cross_entropy(
            logits_audio.flatten(0, 2), audio_tokens.flatten()
        )
        loss_audio_b = F.cross_entropy(
            logits_audio.flatten(0, 2), audio_tokens.roll(1, 0).flatten()
        )

        # summate loss
        loss_category = torch.lerp(loss_category_a, loss_category_b, lam)
        loss_audio = torch.lerp(loss_audio_a, loss_audio_b, lam)
        loss_total = loss_category + loss_audio * self.lambda_audio

        # Yield metrics
        predictions = logits_category.topk(5, dim=-1)[1] == labels.unsqueeze(1)
        accuracy_top1 = predictions[:, 0].float().mean()
        accuracy_top5 = predictions.float().amax(1).mean()

        return {
            "loss_total": loss_total,
            "loss_category": loss_category,
            "loss_audio": loss_audio,
            "accuracy_top1": accuracy_top1,
            "accuracy_top5": accuracy_top5,
        }

    def training_step(self, batch: dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        metrics = self(**batch)
        self.log_dict({f"train/{k}": v for k, v in metrics.items()})
        return metrics["loss_total"]

    def validation_step(self, batch: dict[str, torch.Tensor], idx: int):
        metrics = self(**batch)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, sync_dist=True)

    def test_step(self, batch: dict[str, torch.Tensor], idx: int):
        metrics = self(**batch)
        self.log_dict({f"test/{k}": v for k, v in metrics.items()}, sync_dist=True)

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        optimizer = create_optimizer_v2(self, **self.config.optim.optimizer)
        scheduler = OneCycleLR(
            optimizer,
            **self.config.optim.scheduler,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

