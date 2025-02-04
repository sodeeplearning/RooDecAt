"""Torch implementation of Root Decomposition based attention mechanism."""

import torch

from attention.torch_imp.utils import UpdateModel


class RooDecAttention (torch.nn.Module):
    """Attention based on a root decomposition mechanism."""
    def __init__(
            self,
            max_seq: int = 1024,
            threshold: float = 0.5,
            d_model: int = 512,
            d_attn: int = 256,
            num_heads: int = 8,
            dropout: float = 0.0,
            use_bias: bool = False,
            additional_attention: torch.nn.Module = None
    ):
        """Constructor of RooDecAttention class.

        :param max_seq: Max length of input sequence.
        :param threshold: Confidence threshold (more - stricter)
        :param d_model: Size of embeddings inside model.
        :param d_attn: Size of key and query tensors.
        :param num_heads: Num of heads in Multi-Head attention.
        :param dropout: Dropout in multi head attention
        :param use_bias: 'True' if you need to use bias in Linear in Layernorm layers.
        :param additional_attention: Attention using with probability mask (default - Multi-Head attention).
        """
        super().__init__()

        self.max_seq = max_seq
        self.threshold = threshold
        self.d_attn = d_attn
        self.root = max_seq ** 0.5

        assert self.root % 1 == 0, "Max seq must be a square of some number."
        self.root = int(self.root)

        if additional_attention is None:
            additional_attention = torch.nn.MultiheadAttention(
                num_heads=num_heads,
                embed_dim=d_model,
                dropout=dropout,
                batch_first=True,
                bias=use_bias
            )
        self.additional_attention = additional_attention

        self.key_weights = torch.nn.Linear(
            in_features=d_model,
            out_features=d_attn,
            bias=use_bias
        )
        self.query_weights = torch.nn.Linear(
            in_features=d_model,
            out_features=d_attn,
            bias=use_bias
        )

        self.update_embedding = UpdateModel(
            d_model=d_model,
            max_seq=max_seq
        )

    def __process_prob_matrix(
            self,
            prob_matrix: torch.Tensor,
            input_sequence: torch.Tensor
    ) -> list[list]:

        prob_matrix = prob_matrix / (self.d_attn ** 0.5)
        prob_matrix = prob_matrix.softmax(-1)

        indexes_matrix = [[[] for _ in range(self.max_seq)] for _ in range(prob_matrix.shape[0])]

        for batch_ind, current_batch in enumerate(prob_matrix):
            for word_ind, current_probs in enumerate(current_batch):
                for block_ind, block_prob in enumerate(current_probs):
                    if block_prob >= self.threshold or word_ind // self.root == block_ind:
                        indexes_matrix[batch_ind][word_ind].append(block_ind)

        embeddings_matrix = [[[] for _ in range(self.max_seq)] for _ in range(prob_matrix.shape[0])]

        for batch_ind, current_batch in enumerate(indexes_matrix):
            for word_ind, word_block_indexes in enumerate(current_batch):
                for current_block_index in word_block_indexes:
                    embeddings_matrix[batch_ind][word_ind].append(
                        input_sequence[batch_ind][
                            current_block_index * self.root : (current_block_index + 1) * self.root
                        ]
                    )

                adding_tensor = torch.stack(embeddings_matrix[batch_ind][word_ind]).flatten(0, 1)
                embeddings_matrix[batch_ind][word_ind] = adding_tensor

        return embeddings_matrix


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        root_embeddings = self.update_embedding(x)
        key_matrix = self.key_weights(root_embeddings)
        query_matrix = self.query_weights(x)

        prob_matrix = query_matrix @ key_matrix.transpose(-1, -2)

        correlated_parts = self.__process_prob_matrix(
            prob_matrix=prob_matrix,
            input_sequence=x
        )

        output = [[None for _ in range(self.max_seq)] for _ in range(prob_matrix.shape[0])]

        for batch_ind, current_batch in enumerate(correlated_parts):
            for word_ind, correlated_words in enumerate(current_batch):
                current_word_embedding = x[batch_ind][word_ind].unsqueeze(0)
                attn_output, _ = self.additional_attention(
                    current_word_embedding,
                    correlated_words,
                    correlated_words
                )
                output[batch_ind][word_ind] = attn_output

        output = [torch.stack(current) for current in output]
        output = torch.stack(output).squeeze() + x
        return output
