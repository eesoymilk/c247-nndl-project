# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert num_features % channels == 0, (
                "block_channels must evenly divide num_features"
            )
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class RotationInvariantTCN(nn.Module):
    """A rotation-invariant Temporal Convolutional Network (TCN) module.

    This applies 1D convolutions over time while rotating the electrode channels
    and pooling over rotations for invariance.

    Args:
        in_channels (int): Number of input electrode channels * frequency bins.
        num_filters (Sequence[int]): List of feature sizes for each TCN layer.
        kernel_size (int): Kernel size for the TCN layers.
        dilation_base (int): Base dilation rate (e.g., 2).
        pooling (str): Either "mean" or "max" pooling over rotated versions.
        offsets (Sequence[int]): Rotations applied to electrode channels.
    """

    def __init__(
        self,
        in_channels: int,  # Number of electrode channels * frequency bins
        num_filters: Sequence[int] = (24, 24, 24, 24),
        kernel_size: int = 3,
        dilation_base: int = 2,
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ):
        super().__init__()

        assert len(num_filters) > 0
        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"

        self.offsets = offsets if len(offsets) > 0 else (0,)
        self.pooling = pooling

        layers = []

        # Create multiple TCN layers with increasing dilation
        for i, out_channels in enumerate(num_filters):
            # Exponential dilation
            dilation = dilation_base**i

            # Ensure proper padding
            theoretical_padding = (kernel_size - 1) * dilation / 2
            padding: int | list[int] = int(theoretical_padding)
            if not theoretical_padding.is_integer():
                padding = [padding, padding + 1]

            layers.extend(
                [
                    nn.ConstantPad1d(padding, 0),
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels

        self.tcn = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Rotation-Invariant TCN.

        Args:
            inputs (torch.Tensor): Shape (T, N, electrode_channels, freq)

        Returns:
            torch.Tensor: Shape (T, N, num_filters[-1])
        """
        T, N, C, F = inputs.shape  # Time, Batch, Electrode Channels, Frequency

        # Merge electrode channels and frequency into a single feature vector per time step
        inputs = inputs.view(T, N, C * F)  # Shape: (T, N, in_channels)

        # Rotate electrode channels
        # Shape: (T, N, rotations, in_channels)
        rotated_inputs = torch.stack(
            [inputs.roll(offset, dims=2) for offset in self.offsets], dim=2
        )

        # Merge "rotations" into batch dimension: (T, N * rotations, in_channels)
        T, N, rotations, D = rotated_inputs.shape
        rotated_inputs = rotated_inputs.view(T, N * rotations, D)

        # Transpose for 1D CNN: (N * rotations, in_channels, T)
        rotated_inputs = rotated_inputs.permute(1, 2, 0)

        # Apply TCN over time
        x = self.tcn(rotated_inputs)  # (N * rotations, num_filters[-1], T)

        # Reshape back to (T, N, rotations, num_filters[-1])
        x = x.permute(2, 0, 1).view(T, N, rotations, -1)

        # Pool over rotations to ensure rotation invariance
        if self.pooling == "max":
            return x.max(dim=2).values  # (T, N, num_filters[-1])
        else:
            return x.mean(dim=2)  # (T, N, num_filters[-1])


class MultiBandRotationInvariantTCN(nn.Module):
    """A rotation-invariant MultiBand Temporal Convolutional Network (TCN).

    Each frequency band is processed separately through its own Rotation-Invariant TCN.
    The outputs are then pooled or concatenated across bands.

    Args:
        in_channels (int): Number of electrode channels.
        num_filters (Sequence[int]): Feature sizes per TCN layer.
        num_bands (int): Number of frequency bands.
        kernel_size (int): Kernel size for TCN layers.
        dilation_base (int): Base dilation rate (e.g., 2).
        pooling (str): Either "mean" or "max" pooling over rotated versions.
        band_merge_mode (str): "concat" to concatenate band outputs or "mean" to average.
        offsets (Sequence[int]): List of rotations for electrode channels.
    """

    def __init__(
        self,
        in_channels: int,  # Number of electrode channels
        num_filters: Sequence[int],  # Feature map sizes per layer
        kernel_size: int = 3,
        dilation_base: int = 2,
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ):
        super().__init__()

        assert len(num_filters) > 0
        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"

        self.num_bands = num_bands
        self.stack_dim = stack_dim
        self.pooling = pooling
        self.offsets = offsets if len(offsets) > 0 else (0,)

        # Create a separate TCN for each band
        self.band_tcns = nn.ModuleList(
            [
                RotationInvariantTCN(
                    in_channels=in_channels,
                    num_filters=num_filters,
                    kernel_size=kernel_size,
                    dilation_base=dilation_base,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MultiBand Rotation-Invariant TCN.

        Args:
            inputs (torch.Tensor): Shape (T, N, num_bands, electrode_channels, freq_per_band)

        Returns:
            torch.Tensor: Shape (T, N, num_filters[-1])
        """
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        band_outputs = [
            tcn(_input) for tcn, _input in zip(self.band_tcns, inputs_per_band)
        ]
        return torch.stack(band_outputs, dim=self.stack_dim)


class LSTMGRU(nn.Module):
    """A `torch.nn.Module` that applies a single layer of LSTM and GRU
    over the input tensor.

    Args:
        in_features (int): Number of input features to the LSTM and GRU.
        lstm_layers (int): Number of LSTM layers.
        lstm_hidden_size (int): Number of features in the LSTM hidden state.
        lstm_dropout (float): Dropout probability for the LSTM.
        between_dropout (float): Dropout probability between LSTM and GRU.
        gru_layers (int): Number of GRU layers.
        gru_hidden_size (int): Number of features in the GRU hidden state.
        gru_dropout (float): Dropout probability for the GRU.
        num_bands (int): Number of frequency bands.
    """

    def __init__(
        self,
        in_features: int,
        lstm_layers: int = 3,
        lstm_hidden_size: int = 256,
        lstm_dropout: float = 0.0,
        between_dropout: float = 0.0,
        gru_layers: int = 2,
        gru_hidden_size: int = 256,
        gru_dropout: float = 0.0,
        gru_bidirectional: bool = False,
        num_bands: int = 2,
    ) -> None:
        super().__init__()

        self.num_bands = num_bands
        self.lstm = nn.LSTM(
            in_features, lstm_hidden_size, lstm_layers, dropout=lstm_dropout
        )
        self.dropout = nn.Dropout(between_dropout)
        self.gru = nn.GRU(
            lstm_hidden_size,
            gru_hidden_size,
            gru_layers,
            dropout=gru_dropout,
            bidirectional=gru_bidirectional,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LSTM-GRU module.

        Args:
            inputs (torch.Tensor): Shape (T, N, in_features)

        Returns:
            torch.Tensor: Shape (T, N, gru_hidden_size)
        """
        lstm_output, _ = self.lstm(inputs)
        lstm_output = self.dropout(lstm_output)
        gru_output, _ = self.gru(lstm_output)
        return gru_output


class MultiBandLSTMGRU(nn.Module):
    """A `torch.nn.Module` that applies a single layer of LSTM and GRU
    over the input tensor.

    Args:
        in_features (int): Number of input features to the LSTM and GRU.
        lstm_layers (int): Number of LSTM layers.
        lstm_hidden_size (int): Number of features in the LSTM hidden state.
        lstm_dropout (float): Dropout probability for the LSTM.
        between_dropout (float): Dropout probability between LSTM and GRU.
        gru_layers (int): Number of GRU layers.
        gru_hidden_size (int): Number of features in the GRU hidden state.
        gru_dropout (float): Dropout probability for the GRU.
        num_bands (int): Number of frequency bands.
    """

    def __init__(
        self,
        in_features: int,
        lstm_layers: int = 3,
        lstm_hidden_size: int = 256,
        lstm_dropout: float = 0.1,
        between_dropout: float = 0.1,
        gru_layers: int = 2,
        gru_hidden_size: int = 256,
        gru_dropout: float = 0.1,
        gru_bidirectional: bool = False,
        num_bands: int = 2,
    ) -> None:
        super().__init__()

        self.num_bands = num_bands
        self.lstm_grus = nn.ModuleList(
            [
                LSTMGRU(
                    in_features=in_features,
                    lstm_layers=lstm_layers,
                    lstm_hidden_size=lstm_hidden_size,
                    lstm_dropout=lstm_dropout,
                    between_dropout=between_dropout,
                    gru_layers=gru_layers,
                    gru_hidden_size=gru_hidden_size,
                    gru_dropout=gru_dropout,
                    gru_bidirectional=gru_bidirectional,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LSTM-GRU module.

        Args:
            inputs (torch.Tensor): Shape (T, N, num_bands, electrode_channels, freq_per_band)

        Returns:
            torch.Tensor: Shape (T, N, num_bands, hidden_size)
        """
        assert inputs.shape[2] == self.num_bands

        T, N, _, C, F = inputs.shape

        # Shape: (T, N, num_bands, electrode_channels * freq_per_band)
        inputs = inputs.view(T, N, self.num_bands, C * F)

        # Shape for each: (T, N, electrode_channels * freq_per_band)
        inputs_per_band = inputs.unbind(2)

        # Shape for each: (T, N, gru_hidden_size)
        outputs_per_band = [
            lstm_gru(_input)
            for lstm_gru, _input in zip(self.lstm_grus, inputs_per_band)
        ]

        return torch.stack(outputs_per_band, dim=2)
