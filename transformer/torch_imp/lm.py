import torch

from collections import OrderedDict

from .block import RooDecAtBlock
from .config import BaseConfig
from .utils import GenerationArgs, TransformerArgs


class RooDecAtLM(torch.nn.Module):
    def __init__(
            self,
            transformer_args: TransformerArgs = TransformerArgs(),
            generation_args: GenerationArgs = GenerationArgs(),
            embedding_model = None
    ):
        super().__init__()

        self.transformer_args = transformer_args
        self.generation_args = generation_args

        if embedding_model is None:
            embedding_model = torch.nn.Embedding(
                num_embeddings=BaseConfig.dict_size,
                embedding_dim=transformer_args.d_model
            )
        self.embedding_model = embedding_model

        self.decoder_sequence = torch.nn.Sequential(*[
            RooDecAtBlock(
                transformer_args=transformer_args
            )
            for _ in range(transformer_args.num_blocks)
        ])

        self.output_layer = torch.nn.Sequential(OrderedDict([
            ("Output_linear", torch.nn.Linear(
                in_features=transformer_args.d_model,
                out_features=BaseConfig.dict_size,
                bias=transformer_args.transformer_bias
            )),
            ("Softmax", torch.nn.Softmax(dim=-1))
        ]))

    def forward(self, x: torch.Tensor, tokens: bool = True) -> torch.Tensor:
        if tokens:
            x = x.to(dtype=torch.long)
            x = self.embedding_model(x)
        processed_embeddings = self.decoder_sequence(x)

        probs = self.output_layer(processed_embeddings[:, -1])
        return probs

    def __process_sample(self, sample: torch.Tensor) -> int:
        if self.generation_args.temperature == 0:
            return sample.argmax(dim=-1).item()

        sample = sample.log() / self.generation_args.temperature
        sample, indexes = torch.sort(sample)

        current_sum = 0
        p_bound = sample.shape[0]
        for ind, current_element in enumerate(sample, start=1):
            current_sum += current_element
            if current_sum >= self.generation_args.top_p:
                p_bound = ind
                break

        right_bound = min(p_bound, self.generation_args.top_k, sample.shape[0])
        sample = sample[:right_bound]
        chosen_index = torch.multinomial(
            input=sample.softmax(dim=-1),
            num_samples=1
        )
        chosen_token = indexes[chosen_index].item()
        return chosen_token

    def generate(
            self,
            input_seq: list | torch.Tensor,
            tokens: bool = True,
            use_cuda: bool = True
    ) -> list[int]:
        if isinstance(input_seq, list):
            input_seq = torch.tensor(input_seq)
        if len(input_seq.shape) == 1:
            input_seq = input_seq.unsqueeze(0)

        if tokens:
            assert input_seq.shape[-1] <= self.transformer_args.max_input_seq, \
                "Size of input sequence more than max value"
            input_seq = torch.cat((
                input_seq,
                torch.zeros(input_seq.shape[0], self.transformer_args.max_input_seq - input_seq.shape[-1])
            ), dim=-1)

        probs = self.forward(
            x=input_seq.to(device=torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")),
            tokens=tokens
        )

        answer_massive = []
        for current_sample in probs:
            answer_massive.append(self.__process_sample(sample=current_sample))

        return answer_massive
