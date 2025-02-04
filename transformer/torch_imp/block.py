import torch

from collections import OrderedDict

from .config import BaseConfig
from .utils import TransformerArgs

from attention.torch_imp import RooDecAttention


class RooDecAtBlock(torch.nn.Module):
    def __init__(
            self,
            transformer_args: TransformerArgs = TransformerArgs(),
    ):
        super().__init__()

        self.transformer_args = transformer_args

        self.roodec_attention = RooDecAttention(
            max_seq=transformer_args.max_input_seq,
            threshold=transformer_args.attention_threshold,
            d_model=transformer_args.d_model,
            d_attn=transformer_args.d_attn,
            num_heads=transformer_args.ma_heads_amount,
            dropout=transformer_args.dropout,
            use_bias=transformer_args.attention_bias,
            additional_attention=transformer_args.additional_attention
        )

        self.feedforward_network = torch.nn.Sequential(OrderedDict([
            ("Linear_1", torch.nn.Linear(
                in_features=transformer_args.d_model,
                out_features=transformer_args.d_model * BaseConfig.ff_multy_factor,
                bias=transformer_args.transformer_bias
            )),
            ("FF_Activation", torch.nn.LeakyReLU(
                negative_slope=BaseConfig.leaky_relu_value
            )),
            ("Lienear_2", torch.nn.Linear(
                in_features=transformer_args.d_model * BaseConfig.ff_multy_factor,
                out_features=transformer_args.d_model,
                bias=transformer_args.transformer_bias
            ))
        ]))

        self.layer_norm_1 = torch.nn.LayerNorm(
            normalized_shape=transformer_args.d_model,
            bias=transformer_args.transformer_bias
        )
        self.layer_norm_2 = torch.nn.LayerNorm(
            normalized_shape=transformer_args.d_model,
            bias=transformer_args.transformer_bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized_1 = self.layer_norm_1(x)
        attention_masked = self.roodec_attention(normalized_1)
        residual_connected_1 = x + attention_masked

        normalized_2 = self.layer_norm_2(residual_connected_1)
        forwarded = self.feedforward_network(normalized_2)
        residual_connected_2 = residual_connected_1 + forwarded

        return residual_connected_2