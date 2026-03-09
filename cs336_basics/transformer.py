import torch
from einops import einsum, rearrange
import math

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
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

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
        to_sum = x.pow(2)
        to_mul = to_sum.sum(dim=-1, keepdim=True)
        to_sqrt = torch.div(to_mul, self.d_model) + self.eps
        return to_sqrt.sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # code for RMSNorm
        rms = self._rms(x)
        first_result = torch.mul(self.gain, x)
        result = torch.div(first_result, rms)
        # x is (batch_size, seq_len, d_model), output is same

        return result.to(in_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.device | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        self.d_ff = d_ff or int((8 / 3) * self.d_model)
        self.w1 = torch.empty(self.d_ff, self.d_model)
        self.w2 = torch.empty(self.d_model, self.d_ff)
        self.w3 = torch.empty(self.d_ff, self.d_model)
        std = (2 / (self.d_ff + self.d_model)) ** (0.5)
        torch.nn.init.trunc_normal_(
            tensor=self.w1,
            mean=0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
        torch.nn.init.trunc_normal_(
            tensor=self.w2,
            mean=0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
        torch.nn.init.trunc_normal_(
            tensor=self.w3,
            mean=0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
        self.W1 = torch.nn.Parameter(self.w1)
        self.W2 = torch.nn.Parameter(self.w2)
        self.W3 = torch.nn.Parameter(self.w3)

    def silu(x: torch.Tensor) -> torch.Tensor:
        return torch.mul(x, torch.sigmoid(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W1x = einsum(self.W1, x, "d_ff d_model, batch seq_len d_model -> batch seq_len d_ff")
        W3x = einsum(self.W3, x, "d_ff d_model, batch seq_len d_model -> batch seq_len d_ff")
        glu = torch.mul(SwiGLU.silu(W1x), W3x)
        return einsum(self.W2, glu, "d_model d_ff, batch seq_len d_ff -> batch seq_len d_model")

class RoPE(torch.nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device


        self.k = torch.arange(self.d_k//2)
        self.angle = self.theta**((2*self.k - 2)/self.d_k)

        self.positions = torch.arange(self.max_seq_len)

        thetas = torch.outer(self.positions, self.angle)

        self.register_buffer(name = "cos", tensor=torch.cos(thetas), persistent=False)
        self.register_buffer(name = "sin", tensor=torch.sin(thetas), persistent=False)

    # def small_r(self, i, k):
    #         angle = i/(self.theta**((2*k - 2)/self.d_k))
    #         return torch.Tensor([[math.cos(angle), -math.sin(angle)],[math.sin(angle), math.cos(angle)]])
    
    # def big_r(self, i: int):
    #     return torch.block_diag(*[self.small_r(i, k) for k in range(self.d_k/2)])
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x should be (..., seq_len, d_k) and output should be same
        # x should tolerate arbitrary num of batch dimensions

        # token_positions tensor: (..., seq_len)
        print(x.shape)
        print(token_positions)
        print(self.cos.shape)
        print(self.sin.shape)
        token_positions = token_positions.to(dtype=torch.int64, device = self.device)
        
        cos_positions = self.cos[token_positions]
        sin_positions = self.sin[token_positions]

        #split into even and odd
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        print(x_even.shape)
        print(x_odd.shape)

        first = cos_positions * x_even - sin_positions * x_even
        second = sin_positions * x_odd + cos_positions * x_odd

        out = torch.zeros_like(x)

        out[..., 0::2] = first
        out[..., 1::2] = second
    
        return out
