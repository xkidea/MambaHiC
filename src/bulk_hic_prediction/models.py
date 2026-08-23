from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:  # pragma: no cover - depends on local CUDA build.
    Mamba = None


def require_mamba() -> None:
    if Mamba is None:
        raise ImportError(
            "mamba_ssm is required for Mamba models. Install causal-conv1d and mamba-ssm "
            "in the active PyTorch/CUDA environment."
        )


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, activation: str = "relu"):
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            act,
        )

    def forward(self, x):
        return self.block(x)


class SequenceMambaEncoder(nn.Module):
    """Sequence encoder used by the shared main/cross-cell/ablation weights."""

    def __init__(self, in_channel: int = 4, output_channels: int = 128, output_len: int = 512):
        super().__init__()
        require_mamba()
        self.downsampler = nn.Sequential(
            DownsampleBlock(in_channel, 48, kernel_size=5, stride=5),
            DownsampleBlock(48, 96, kernel_size=5, stride=5),
            DownsampleBlock(96, 128, kernel_size=5, stride=5),
        )
        self.mamba_fw = Mamba(d_model=128, d_state=16, d_conv=4, expand=2)
        self.mamba_bw = Mamba(d_model=128, d_state=16, d_conv=4, expand=2)
        self.pooler = nn.AdaptiveAvgPool1d(output_len)
        self.projection = nn.Conv1d(128, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.downsampler(x).transpose(1, 2)
        x_fw = self.mamba_fw(x)
        x_bw = torch.flip(self.mamba_bw(torch.flip(x, dims=[1])), dims=[1])
        x = (x_fw + x_bw).transpose(1, 2)
        return self.projection(self.pooler(x))


class OmicsMambaEncoder(nn.Module):
    def __init__(self, num_omics_features: int = 4, d_model: int = 128):
        super().__init__()
        require_mamba()
        self.projection = nn.Conv1d(num_omics_features, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fw = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_bw = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)

    def forward(self, x):
        x = self.norm(self.projection(x).transpose(1, 2))
        x_fw = self.mamba_fw(x)
        x_bw = torch.flip(self.mamba_bw(torch.flip(x, dims=[1])), dims=[1])
        return (x_fw + x_bw).transpose(1, 2)


class SequenceMambaEncoderBlocks(nn.Module):
    """Feature-ablation encoder variant with ModuleList Mamba blocks."""

    def __init__(self, in_channel: int = 4, output_channels: int = 128, output_len: int = 512, num_blocks: int = 1):
        super().__init__()
        require_mamba()
        self.downsampler = nn.Sequential(
            DownsampleBlock(in_channel, 48, kernel_size=5, stride=5),
            DownsampleBlock(48, 96, kernel_size=5, stride=5),
            DownsampleBlock(96, 128, kernel_size=5, stride=5),
        )
        self.mamba_fw_blocks = nn.ModuleList([Mamba(d_model=128, d_state=16, d_conv=4, expand=2) for _ in range(num_blocks)])
        self.mamba_bw_blocks = nn.ModuleList([Mamba(d_model=128, d_state=16, d_conv=4, expand=2) for _ in range(num_blocks)])
        self.pooler = nn.AdaptiveAvgPool1d(output_len)
        self.projection = nn.Conv1d(128, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.downsampler(x).transpose(1, 2)
        x_fw = x
        for mamba_layer in self.mamba_fw_blocks:
            x_fw = mamba_layer(x_fw)
        x_bw = torch.flip(x, dims=[1])
        for mamba_layer in self.mamba_bw_blocks:
            x_bw = mamba_layer(x_bw)
        x_bw = torch.flip(x_bw, dims=[1])
        return self.projection(self.pooler((x_fw + x_bw).transpose(1, 2)))


class OmicsMambaEncoderBlocks(nn.Module):
    """Feature-ablation omics encoder variant with ModuleList Mamba blocks."""

    def __init__(self, num_omics_features: int = 4, d_model: int = 128, num_blocks: int = 1):
        super().__init__()
        require_mamba()
        self.projection = nn.Conv1d(num_omics_features, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fw_blocks = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(num_blocks)])
        self.mamba_bw_blocks = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.norm(self.projection(x).transpose(1, 2))
        x_fw = x
        for mamba_layer in self.mamba_fw_blocks:
            x_fw = mamba_layer(x_fw)
        x_bw = torch.flip(x, dims=[1])
        for mamba_layer in self.mamba_bw_blocks:
            x_bw = mamba_layer(x_bw)
        x_bw = torch.flip(x_bw, dims=[1])
        return (x_fw + x_bw).transpose(1, 2)


class MultiModalEncoder(nn.Module):
    def __init__(
        self,
        seq_in_channels: int = 4,
        num_omics_features: int = 4,
        encoder_out_dim: int = 128,
        output_len: int = 512,
        ablate_dna: bool = False,
        ablate_omics_all: bool = False,
        encoder_variant: str = "single",
    ):
        super().__init__()
        if ablate_dna and ablate_omics_all:
            raise ValueError("Cannot ablate both DNA and all omics.")
        self.ablate_dna = ablate_dna
        self.ablate_omics_all = ablate_omics_all
        sequence_encoder_cls = SequenceMambaEncoderBlocks if encoder_variant == "blocks" else SequenceMambaEncoder
        omics_encoder_cls = OmicsMambaEncoderBlocks if encoder_variant == "blocks" else OmicsMambaEncoder
        if not self.ablate_dna:
            self.seq_encoder = sequence_encoder_cls(seq_in_channels, output_channels=encoder_out_dim, output_len=output_len)
        if not self.ablate_omics_all:
            self.omics_encoder = omics_encoder_cls(num_omics_features=num_omics_features, d_model=encoder_out_dim)
        self.fused_channels = encoder_out_dim

    def forward(self, seq_input, omics_input):
        if self.ablate_dna:
            return self.omics_encoder(omics_input)
        if self.ablate_omics_all:
            return self.seq_encoder(seq_input)
        seq_embedding = self.seq_encoder(seq_input)
        omics_embedding = self.omics_encoder(omics_input)
        if seq_embedding.shape[-1] != omics_embedding.shape[-1]:
            raise ValueError(
                f"Sequence and omics lengths differ after encoding: {seq_embedding.shape[-1]} vs {omics_embedding.shape[-1]}. "
                "Set sequence_output_len to match the omics bin count."
            )
        return seq_embedding + omics_embedding


class ResBlockDilated(nn.Module):
    def __init__(self, size: int, hidden: int = 64, dil: int = 2):
        super().__init__()
        kernel_size = size if size % 2 else size + 1
        pad_len = dil * (kernel_size // 2)
        self.res = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size, padding=pad_len, dilation=dil),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size, padding=pad_len, dilation=dil),
            nn.BatchNorm2d(hidden),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.res(x) + x)


class Decoder(nn.Module):
    def __init__(self, in_channel: int, hidden: int = 256, filter_size: int = 3, num_blocks: int = 5):
        super().__init__()
        self.filter_size = filter_size
        self.conv_start = nn.Sequential(
            nn.Conv2d(in_channel, hidden, 3, 1, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlockDilated(self.filter_size, hidden=hidden, dil=2 ** (i + 1)) for i in range(num_blocks)]
        )
        self.conv_end = nn.Conv2d(hidden, 1, 1)

    def forward(self, x):
        return self.conv_end(self.res_blocks(self.conv_start(x)))


class EfficientDecoder(nn.Module):
    def __init__(self, in_features: int, hidden_2d_channels: int = 256, num_blocks: int = 5, filter_size: int = 3):
        super().__init__()
        self.projection = nn.Conv1d(in_features, hidden_2d_channels, kernel_size=1)
        self.norm_relu = nn.Sequential(nn.BatchNorm2d(hidden_2d_channels), nn.ReLU(inplace=True))
        self.res_blocks = nn.Sequential(
            *[ResBlockDilated(filter_size, hidden=hidden_2d_channels, dil=2 ** (i + 1)) for i in range(num_blocks)]
        )
        self.conv_end = nn.Conv2d(hidden_2d_channels, 1, kernel_size=1)

    def forward(self, x):
        x_proj = self.projection(x)
        feature_map = x_proj.unsqueeze(3) + x_proj.unsqueeze(2)
        feature_map = self.norm_relu(feature_map)
        return self.conv_end(self.res_blocks(feature_map))


class MultiModalSeq2HiCModel(nn.Module):
    """Model matching weights/main, weights/cross_cell, and weights/ablation_ddp."""

    def __init__(
        self,
        seq_in_channels: int = 4,
        num_omics_features: int = 4,
        encoder_out_dim: int = 128,
        sequence_output_len: int = 512,
        ablate_dna: bool = False,
        ablate_omics_all: bool = False,
        encoder_variant: str = "single",
    ):
        super().__init__()
        self.encoder = MultiModalEncoder(
            seq_in_channels=seq_in_channels,
            num_omics_features=num_omics_features,
            encoder_out_dim=encoder_out_dim,
            output_len=sequence_output_len,
            ablate_dna=ablate_dna,
            ablate_omics_all=ablate_omics_all,
            encoder_variant=encoder_variant,
        )
        decoder_in_channels = encoder_out_dim * 2
        self.decoder = Decoder(in_channel=decoder_in_channels, hidden=decoder_in_channels)

    @staticmethod
    def diagonalize(x):
        _, _, length = x.shape
        x_i = x.unsqueeze(3).repeat(1, 1, 1, length)
        x_j = x.unsqueeze(2).repeat(1, 1, length, 1)
        return torch.cat([x_i, x_j], dim=1)

    def forward(self, seq_one_hot, omics_signals):
        encoded_features = self.encoder(seq_one_hot, omics_signals)
        feature_map = self.diagonalize(encoded_features)
        return self.decoder(feature_map).squeeze(1)


class EfficientMultiModalSeq2HiCModel(nn.Module):
    """DDP ablation model matching weights/ablation_ddp."""

    def __init__(
        self,
        seq_in_channels: int = 4,
        num_omics_features: int = 4,
        encoder_out_dim: int = 128,
        sequence_output_len: int = 512,
        ablate_dna: bool = False,
        ablate_omics_all: bool = False,
        encoder_variant: str = "single",
    ):
        super().__init__()
        self.encoder = MultiModalEncoder(
            seq_in_channels=seq_in_channels,
            num_omics_features=num_omics_features,
            encoder_out_dim=encoder_out_dim,
            output_len=sequence_output_len,
            ablate_dna=ablate_dna,
            ablate_omics_all=ablate_omics_all,
            encoder_variant=encoder_variant,
        )
        self.decoder = EfficientDecoder(in_features=encoder_out_dim, hidden_2d_channels=encoder_out_dim * 2, num_blocks=5)

    def forward(self, seq_one_hot, omics_signals):
        encoded_features = self.encoder(seq_one_hot, omics_signals)
        return self.decoder(encoded_features).squeeze(1)


class DeepFeatureResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class HybridCNNMambaEncoder(nn.Module):
    def __init__(self, in_channels: int, d_model: int = 128, num_cnn_blocks: int = 3, num_mamba_layers: int = 3, is_sequence: bool = True):
        super().__init__()
        require_mamba()
        self.is_sequence = is_sequence
        if self.is_sequence:
            self.downsampler = nn.Sequential(
                DownsampleBlock(in_channels, 48, kernel_size=5, stride=5, activation="gelu"),
                DownsampleBlock(48, 96, kernel_size=5, stride=5, activation="gelu"),
                DownsampleBlock(96, d_model, kernel_size=5, stride=5, activation="gelu"),
                DownsampleBlock(d_model, d_model, kernel_size=5, stride=4, activation="gelu"),
                DownsampleBlock(d_model, d_model, kernel_size=5, stride=4, activation="gelu"),
            )
        else:
            self.downsampler = nn.Sequential(nn.Conv1d(in_channels, d_model, kernel_size=1), nn.GELU())
        self.cnn_feature_extractor = nn.Sequential(*[DeepFeatureResBlock(d_model) for _ in range(num_cnn_blocks)])
        self.mamba_layers = nn.ModuleList([Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(num_mamba_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.downsampler(x)
        x = self.cnn_feature_extractor(x).transpose(1, 2)
        x = self.norm(x)
        for mamba_layer in self.mamba_layers:
            x_fw = mamba_layer(x)
            x_bw = torch.flip(mamba_layer(torch.flip(x, dims=[1])), dims=[1])
            x = x + x_fw + x_bw
        return x.transpose(1, 2)


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.omics_cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.seq_cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.fusion_conv = nn.Sequential(nn.Conv1d(d_model * 2, d_model, 1), nn.BatchNorm1d(d_model), nn.GELU())

    def forward(self, seq_emb, omics_emb):
        seq_t, omics_t = seq_emb.transpose(1, 2), omics_emb.transpose(1, 2)
        omics_up, _ = self.omics_cross_attn(query=omics_t, key=seq_t, value=seq_t)
        omics_t = self.norm1(omics_t + omics_up)
        seq_up, _ = self.seq_cross_attn(query=seq_t, key=omics_t, value=omics_t)
        seq_t = self.norm2(seq_t + seq_up)
        combined = torch.cat([seq_t.transpose(1, 2), omics_t.transpose(1, 2)], dim=1)
        return self.fusion_conv(combined)


class SimpleFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(in_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, seq_emb, omics_emb):
        return self.fusion_conv(torch.cat([seq_emb, omics_emb], dim=1))


class ConvResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class PracticalOuterProductDecoder(nn.Module):
    def __init__(self, in_channels: int = 128, bottleneck_channels: int = 16, num_blocks: int = 4):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1)
        self.initial_conv = nn.Conv2d(bottleneck_channels * 2, 64, 1)
        self.res_blocks = nn.Sequential(*[ConvResBlock(64) for _ in range(num_blocks)])
        self.final_conv = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, fused_embedding):
        _, _, length = fused_embedding.shape
        bottleneck_emb = self.bottleneck(fused_embedding)
        x1 = bottleneck_emb.unsqueeze(3).expand(-1, -1, length, length)
        x2 = bottleneck_emb.unsqueeze(2).expand(-1, -1, length, length)
        x = self.initial_conv(torch.cat([x1, x2], dim=1))
        x = self.final_conv(self.res_blocks(x))
        return (x + x.transpose(-2, -1)) / 2


class CNNMambaHybridModelV2(nn.Module):
    """Benchmark Mamba v2 model matching weights/benchmark/gm12878_mamba_v2_best_model_v2.pth."""

    def __init__(
        self,
        seq_in_channels: int = 4,
        num_omics_features: int = 4,
        d_model: int = 128,
        fusion_out_dim: int = 128,
        num_cnn_blocks: int = 3,
        num_mamba_layers: int = 3,
        decoder_bottleneck_channels: int = 48,
        decoder_num_blocks: int = 6,
    ):
        super().__init__()
        self.seq_encoder = HybridCNNMambaEncoder(seq_in_channels, d_model, num_cnn_blocks, num_mamba_layers)
        self.omics_encoder = HybridCNNMambaEncoder(num_omics_features, d_model, num_cnn_blocks, num_mamba_layers, is_sequence=False)
        self.fusion_module = CrossAttentionFusion(d_model=d_model)
        self.decoder = PracticalOuterProductDecoder(fusion_out_dim, decoder_bottleneck_channels, decoder_num_blocks)

    def forward(self, seq, omics):
        seq_emb = self.seq_encoder(seq)
        omics_emb = self.omics_encoder(omics)
        return self.decoder(self.fusion_module(seq_emb, omics_emb)).squeeze(1)


class CNNMambaHybridModelLegacy(nn.Module):
    """Original GM12878 Figure 04 model matching output_GM12878_train_64stride/best_model.pth."""

    def __init__(
        self,
        seq_in_channels: int = 4,
        num_omics_features: int = 4,
        d_model: int = 128,
        fusion_out_dim: int = 128,
        num_cnn_blocks: int = 3,
        num_mamba_layers: int = 3,
        decoder_bottleneck_channels: int = 48,
        decoder_num_blocks: int = 6,
    ):
        super().__init__()
        self.seq_encoder = HybridCNNMambaEncoder(seq_in_channels, d_model, num_cnn_blocks, num_mamba_layers, is_sequence=True)
        self.omics_encoder = HybridCNNMambaEncoder(num_omics_features, d_model, num_cnn_blocks, num_mamba_layers, is_sequence=False)
        self.fusion_module = SimpleFusion(in_channels=d_model, out_channels=fusion_out_dim)
        self.decoder = PracticalOuterProductDecoder(fusion_out_dim, decoder_bottleneck_channels, decoder_num_blocks)

    def encode_fused(self, seq, omics):
        seq_emb = self.seq_encoder(seq)
        omics_emb = self.omics_encoder(omics)
        return self.fusion_module(seq_emb, omics_emb)

    def forward(self, seq, omics):
        return self.decoder(self.encode_fused(seq, omics)).squeeze(1)


class ComparisonConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.scale = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.res = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        scaled = self.scale(x)
        return self.relu(self.res(scaled) + scaled)


class ComparisonResBlockDilated(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 2):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.res(x) + x)


class AdaptedEncoderSplit(nn.Module):
    def __init__(self, seq_in_channels: int, omics_in_channels: int, out_dim: int = 128):
        super().__init__()
        self.seq_encoder = nn.Sequential(
            ComparisonConvBlock(seq_in_channels, 48, 5, 5),
            ComparisonConvBlock(48, 96, 5, 5),
            ComparisonConvBlock(96, 128, 5, 5),
            ComparisonConvBlock(128, 128, 5, 4),
            ComparisonConvBlock(128, out_dim, 5, 4),
        )
        self.omics_encoder = nn.Sequential(
            ComparisonConvBlock(omics_in_channels, 64, 5, 1),
            ComparisonConvBlock(64, out_dim, 5, 1),
        )

    def forward(self, seq_x, omics_x):
        return torch.cat([self.seq_encoder(seq_x), self.omics_encoder(omics_x)], dim=1)


class AdaptedDecoder(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 256, num_blocks: int = 5, kernel_size: int = 3):
        super().__init__()
        self.conv_start = nn.Sequential(nn.Conv2d(in_channels, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU())
        self.res_blocks = nn.Sequential(*[ComparisonResBlockDilated(hidden, kernel_size, 2**i) for i in range(num_blocks)])
        self.conv_end = nn.Conv2d(hidden, 1, 1)

    def forward(self, x):
        return self.conv_end(self.res_blocks(self.conv_start(x)))


class ReproducedConvModel(nn.Module):
    """CNN benchmark model matching weights/benchmark/gm12878_reproduced_cnn_best_model.pth."""

    def __init__(self, seq_in_channels: int = 4, num_omics_features: int = 4, encoder_out_dim: int = 128, decoder_hidden_dim: int = 256):
        super().__init__()
        self.encoder = AdaptedEncoderSplit(seq_in_channels, num_omics_features, encoder_out_dim)
        self.decoder = AdaptedDecoder(encoder_out_dim * 2 * 2, decoder_hidden_dim)

    @staticmethod
    def diagonalize(x):
        _, _, length = x.shape
        x_i = x.unsqueeze(2).repeat(1, 1, length, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, length)
        return torch.cat([x_i, x_j], dim=1)

    def forward(self, seq, omics):
        pred = self.decoder(self.diagonalize(self.encoder(seq, omics))).squeeze(1)
        return (pred + pred.transpose(-2, -1)) / 2


def build_model(
    model_type: str,
    num_omics_features: int,
    target_len: int = 512,
    *,
    ablate_dna: bool = False,
    ablate_omics_all: bool = False,
    decoder_variant: str = "diagonal",
    encoder_variant: str = "single",
):
    if model_type in {"main", "cross_cell", "ablation_ddp", "seq2hic"}:
        if decoder_variant == "efficient":
            return EfficientMultiModalSeq2HiCModel(
                num_omics_features=num_omics_features,
                sequence_output_len=target_len,
                ablate_dna=ablate_dna,
                ablate_omics_all=ablate_omics_all,
                encoder_variant=encoder_variant,
            )
        return MultiModalSeq2HiCModel(
            num_omics_features=num_omics_features,
            sequence_output_len=target_len,
            ablate_dna=ablate_dna,
            ablate_omics_all=ablate_omics_all,
            encoder_variant=encoder_variant,
        )
    if model_type in {"mamba_v2", "benchmark_mamba"}:
        return CNNMambaHybridModelV2(num_omics_features=num_omics_features)
    if model_type in {"gm12878_legacy", "mamba_legacy", "figure04_legacy"}:
        return CNNMambaHybridModelLegacy(num_omics_features=num_omics_features)
    if model_type in {"cnn", "reproduced_cnn", "benchmark_cnn"}:
        return ReproducedConvModel(num_omics_features=num_omics_features)
    raise ValueError(f"Unknown model_type: {model_type}")
