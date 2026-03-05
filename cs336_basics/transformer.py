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
    
class RMSNorm(torch.nn.Module):

    def __init__(self, d_model:int, eps: float = 1e-5, device=None, dtype=None):
        self.d_model = d_model
        self.eps = eps

        self.g = torch.empty(self.d_model)
        std = (2 / (self.d_model)) ** (0.5)
        torch.nn.init.trunc_normal_(
            tensor=self.g,
            mean=0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
        self.gain = torch.nn.Parameter(self.g)

    def _rms(self, x):
        to_sum = x.pow(2) + self.eps
        to_mul = to_sum.sum(dim=-1, keep_dim=True)
        to_sqrt = torch.div(to_mul, self.d_model)
        return to_sqrt.sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # code for RMSNorm
        rms = self._rms()
        first_result = torch.mul(self.gain, x)
        result = torch.div(first_result,rms)
        # x is (batch_size, seq_len, d_model), output is same

        return result.to(in_dtype)