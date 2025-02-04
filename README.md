# Root-Decomposition-based-Attention
Attention mechanism based on a root decomposition

## How does this work?

1. Sequence splits to N ** 0.5 blocks
2. Model gets an embedding of every block
3. Attention looks for a correlation between word and block
4. If the correlation is higher than some threshold, model will use Multy-Head attention with it
5. Otherwise model will ignore the block