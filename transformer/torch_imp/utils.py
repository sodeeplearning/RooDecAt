import torch


class TransformerArgs:
    max_input_seq: int = 1024
    num_blocks: int = 16
    d_model: int = 2048
    d_attn: int = 256
    attention_threshold: float = 0.5
    ma_heads_amount: int = 8
    dropout: float = 0.0
    transformer_bias: bool = True
    attention_bias: bool = False
    additional_attention: torch.nn.Module = None


class GenerationArgs:
    max_output_seq: int = 512
    temperature: float = 1.0
    top_k: int = 5
    top_p: float = 0.6