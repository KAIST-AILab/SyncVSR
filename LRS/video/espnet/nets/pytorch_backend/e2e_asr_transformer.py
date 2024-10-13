# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

import logging
import math
from argparse import Namespace
from distutils.util import strtobool

import numpy
import torch

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect, ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import (
    get_subsample,
    make_non_pad_mask,
    th_accuracy,
)
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.scorers.ctc import CTCPrefixScorer

from utils import check_availability
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2ForPreTraining

if check_availability("fairseq"):
    from fairseq.checkpoint_utils import load_model_ensemble

class E2E(torch.nn.Module):
    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, odim, args, ignore_id=-1):
        """Construct an E2E object.
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        # Check the relative positional encoding type
        self.rel_pos_type = getattr(args, "rel_pos_type", None)
        if (
            self.rel_pos_type is None
            and args.transformer_encoder_attn_layer_type == "rel_mha"
        ):
            args.transformer_encoder_attn_layer_type = "legacy_rel_mha"
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )

        idim = 80

        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            encoder_attn_layer_type=args.transformer_encoder_attn_layer_type,
            macaron_style=args.macaron_style,
            use_cnn_module=args.use_cnn_module,
            cnn_module_kernel=args.cnn_module_kernel,
            zero_triu=getattr(args, "zero_triu", False),
            a_upsample_ratio=args.a_upsample_ratio,
            relu_type=getattr(args, "relu_type", "swish"),
        )

        self.transformer_input_layer = args.transformer_input_layer
        self.a_upsample_ratio = args.a_upsample_ratio

        self.proj_decoder = None
        if args.adim != args.ddim:
            self.proj_decoder = torch.nn.Linear(args.adim, args.ddim)

        if args.mtlalpha < 1:
            self.decoder = Decoder(
                odim=odim,
                attention_dim=args.ddim,
                attention_heads=args.dheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            )
        else:
            self.decoder = None
        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="transformer")

        # self.lsm_weight = a
        self.criterion = LabelSmoothingLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight,
            args.transformer_length_normalized_loss,
        )

        self.adim = args.adim
        self.mtlalpha = args.mtlalpha
        if args.mtlalpha > 0.0:
            self.ctc = CTC(
                odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )
        else:
            self.ctc = None

        # Cross Modal Sync Related
        if args.codec is None:
            self.codec = None
        elif "vq" in args.codec.lower() and check_availability("fairseq"):
            wav2vec, metadata = load_model_ensemble(["./vq-wav2vec_kmeans.pt"])
            self.wav2vec = wav2vec[0].requires_grad_(False).eval()
            self.audio_alignment = 4
            self.audio_vocab_size = metadata.model.vq_vars # 320
            self.audio_classifier = nn.Linear(768, self.audio_alignment * metadata.model.vq_groups * self.audio_vocab_size) # 768 -> 4 * 2 * 320
            self.audio_weight = args.audio_weight
            self.codec = "vq"
        elif "wav2vec2" in args.codec.lower():
            # facebook/wav2vec2-large-xlsr-53 is multilingual neural audio quantizer. 
            # We used facebook/wav2vec2-large-960h for English and kehanlu/mandarin-wav2vec2 for Mandarin.
            wav2vec = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")
            del wav2vec.wav2vec2.encoder # remove transformer encoder blocks
            wav2vec = wav2vec.requires_grad_(False).eval()
            codevectors = torch.arange(wav2vec.quantizer.codevectors.size(1))
            codevectors = codevectors.view(1, -1, 1).expand_as(wav2vec.quantizer.codevectors)
            wav2vec.quantizer.codevectors.data = codevectors.float()
            self.wav2vec = wav2vec
            self.audio_alignment = 2
            self.audio_vocab_size = 640
            self.audio_classifier = nn.Linear(768, self.audio_alignment * 2 * self.audio_vocab_size)
            self.audio_weight = args.audio_weight
            self.codec = "wav2vec2"
        
        if self.codec is not None:
            print(f"using {self.codec} neural audio codec")
        else:
            print("Inference purpose only, not using codec")
            print("To train with our method, you should set codec as 'wav2vec2' or 'vq'")
        
    def forward_audios(self, audios: torch.Tensor) -> torch.Tensor:
        # add extra 640 padding for audios to prevent mismatch error. Extra margin will be truncated later
        extra_padding = torch.zeros(audios.size(0), 8000).to(audios.device) # 0.5 sec
        audios = torch.cat([audios, extra_padding], axis=-1)
        if self.codec == None or "vq" in self.codec.lower():
            audio_tokens = self.wav2vec.feature_extractor(audios)
            audio_tokens = self.wav2vec.vector_quantizer.forward_idx(audio_tokens)[1]
            return audio_tokens
        elif "wav2vec2" in self.codec.lower():
            # extract features from raw waveform.
            feats = self.wav2vec.wav2vec2.feature_extractor(audios).transpose(1, 2)
            _, feats = self.wav2vec.wav2vec2.feature_projection(feats)
            indices = self.wav2vec.quantizer(feats)[0].unflatten(-1, (2, -1))[..., 0].long()
            return indices

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def forward(self, x, lengths, audios, label):
        if self.transformer_input_layer == "conv1d":
            lengths = torch.div(lengths, 640, rounding_mode="trunc")
        padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)

        x, _ = self.encoder(x, padding_mask) # x: batch x seq_len(164) x dim 

        if self.codec is not None:
            # audio loss
            audio_tokens = self.forward_audios(audios.permute(1,0,2).squeeze(0))
            audio_tokens = audio_tokens[:, : x.size(1) * self.audio_alignment]

            logits_audio = self.audio_classifier(x)
            logits_audio = logits_audio.float() # converting into float type before the loss calculation
            logits_audio = logits_audio.unflatten(2, (-1, self.audio_vocab_size))
            loss_audio = F.cross_entropy(logits_audio.flatten(0, 2),audio_tokens.flatten())
        else:
            loss_audio = None
        
        # ctc loss
        if self.mtlalpha > 0.0:
            loss_ctc, ys_hat = self.ctc(x.float(), lengths, label)
        else:
            loss_ctc = 0
        if self.proj_decoder:
            x = self.proj_decoder(x)

        # decoder loss
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, _ = self.decoder(ys_in_pad, ys_mask, x, padding_mask)
        loss_att = self.criterion(pred_pad.float(), ys_out_pad)
        loss = self.mtlalpha * loss_ctc + (1 - self.mtlalpha) * loss_att
        
        if self.codec is not None:
            loss = loss + loss_audio * self.audio_weight

        acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )

        return loss, loss_ctc, loss_att, loss_audio, acc
