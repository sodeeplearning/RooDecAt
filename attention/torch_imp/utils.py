import torch


class UpdateModel(torch.nn.Module):
    """Model for updating root vectors."""

    def __init__(
            self,
            max_seq: int,
            d_model: int
    ):
        """Constructor of UpdateModel class.

        :param max_seq: Max length of input sequence.
        :param d_model: Embedding size of the model.
        """
        super().__init__()

        self.update_matrix_1 = torch.nn.Linear(
            in_features=max_seq,
            out_features=d_model
        )
        self.update_matrix_2 = torch.nn.Linear(
            in_features=d_model,
            out_features=int(max_seq ** 0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transposed1 = x.transpose(-1, -2)
        update1 = self.update_matrix_1(transposed1)
        update2 = self.update_matrix_2(update1)
        transposed2 = update2.transpose(-1, -2)
        return transposed2
