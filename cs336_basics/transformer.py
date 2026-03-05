import torch
from einops import einsum, rearrange


class Linear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.w = torch.empty(self.out_features, self.in_features)
        std = (2 / (self.in_features + self.out_features)) ** (0.5)
        torch.nn.init.trunc_normal_(
            tensor=self.w,
            mean=0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
        self.W = torch.nn.Parameter(self.w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "d_out d_in, batch seq_len d_in -> batch seq_len d_out")


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.embedding_matrix = torch.empty(self.num_embeddings, self.embedding_dim)
        std = 1
        torch.nn.init.trunc_normal_(
            tensor=self.embedding_matrix,
            mean=0,
            std=std,
            a=-3,
            b=3,
        )
        self.embedding_matrix = torch.nn.Parameter(self.embedding_matrix)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]
