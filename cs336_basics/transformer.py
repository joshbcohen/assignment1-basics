import torch
from einops import einsum, rearrange

# todo: fix DEVICE!!!


class Linear(torch.nn.Module):
    """Simple linear layer (used to scale to dimension vocab_size)."""

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.w = torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype)
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
    """Takes a set of token id's and returns a corresponding set of embeddings."""

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

        self.embedding_matrix = torch.empty(
            self.num_embeddings, self.embedding_dim, device=self.device, dtype=self.dtype
        )
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
    """Computes RMS Norm on activations."""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.g = torch.ones(self.d_model, dtype=self.dtype, device=self.device)
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
    """Feed forward neural net with Swiglu activation, to be used in Transformer after MHA computation."""

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
        self.w1 = torch.empty(self.d_ff, self.d_model, device=self.device, dtype=self.dtype)
        self.w2 = torch.empty(self.d_model, self.d_ff, device=self.device, dtype=self.dtype)
        self.w3 = torch.empty(self.d_ff, self.d_model, device=self.device, dtype=self.dtype)
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
    """Add positional information over embeddings using RoPE."""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        self.k = torch.arange(1, self.d_k // 2 + 1, device=self.device)
        self.angle = 1 / (self.theta ** ((2 * self.k - 2) / self.d_k))

        self.positions = torch.arange(self.max_seq_len, device=self.device)

        thetas = torch.outer(self.positions, self.angle)

        self.register_buffer(name="cos", tensor=torch.cos(thetas), persistent=False)
        self.register_buffer(name="sin", tensor=torch.sin(thetas), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x should be (..., seq_len, d_k) and output should be same
        # x should tolerate arbitrary num of batch dimensions

        # token_positions tensor: (..., seq_len)
        token_positions = token_positions.to(dtype=torch.int64, device=self.device)

        cos_positions = self.cos[token_positions]
        sin_positions = self.sin[token_positions]

        # split into even and odd
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        first = cos_positions * x_even - sin_positions * x_odd
        second = sin_positions * x_even + cos_positions * x_odd

        out = torch.zeros_like(x)

        out[..., 0::2] = first
        out[..., 1::2] = second

        return out


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute softmax on specified dimension."""
    x_max = torch.amax(x, dim=dim, keepdim=True)
    x = torch.sub(x, x_max)
    exp = torch.exp(x)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    """Compute dot product attention."""

    d_k = K.shape[-1]
    # n -> num_queries
    # m -> num_keys/num_values
    qk = einsum(Q, K, "... n d_k, ... m d_k -> ... n m")
    attn = torch.mul(qk, d_k ** (-0.5))
    if mask is not None:
        attn = attn.masked_fill(~mask, value=float("-inf"))
    softmax_attn = softmax(attn, dim=-1)
    out = einsum(softmax_attn, V, "... n m, ... m d_v -> ... n d_v ")

    return out


class MultiheadSelfAttention(torch.nn.Module):
    """Apply RoPE, set W_K, W_Q, W_V parameter matrices, and compute MHSA."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        apply_rope: bool = False,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads
        self.w_query = torch.empty(self.num_heads * self.d_k, self.d_model, device=device)
        self.w_key = torch.empty(self.num_heads * self.d_k, self.d_model, device=device)
        self.w_value = torch.empty(self.num_heads * self.d_v, self.d_model, device=device)
        self.w_output = torch.empty(self.d_model, self.num_heads * self.d_v, device=device)

        std = (1 / (self.d_model)) ** (0.5)
        for w in [self.w_query, self.w_key, self.w_value, self.w_output]:
            torch.nn.init.trunc_normal_(
                    tensor=w,
                    mean=0,
                    std=std,  
                    a=-3 * std,
                    b=3 * std,
            )

        self.W_Q = torch.nn.Parameter(self.w_query)
        self.W_K = torch.nn.Parameter(self.w_key)
        self.W_V = torch.nn.Parameter(self.w_value)
        self.W_O = torch.nn.Parameter(self.w_output)

        self.apply_rope = apply_rope
        if apply_rope:
            self.RoPE = RoPE(theta, self.d_k, max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        Q = einsum(self.W_Q, x, "... h_d_k d_model, ... seq_len d_model -> ... h_d_k seq_len")
        K = einsum(self.W_K, x, "... h_d_k d_model, ... seq_len d_model -> ... h_d_k seq_len")
        V = einsum(self.W_V, x, "... h_d_v d_model, ... seq_len d_model -> ... h_d_v seq_len")
        num_queries = Q.shape[-1]
        num_keys = K.shape[-1]
        mask = torch.tril(torch.ones((num_queries, num_keys), dtype=torch.bool, device=Q.device))

        Q_head = rearrange(Q, " ... (h d_k) seq_len  -> ... h seq_len d_k", h=self.num_heads)
        K_head = rearrange(K, " ... (h d_k) seq_len  -> ... h seq_len d_k", h=self.num_heads)
        V_head = rearrange(V, " ... (h d_v) seq_len  -> ... h seq_len d_v", h=self.num_heads)

        if self.apply_rope:
            Q_head = self.RoPE(Q_head, token_positions)
            K_head = self.RoPE(K_head, token_positions)

        concatted_heads = scaled_dot_product_attention(Q=Q_head, K=K_head, V=V_head, mask=mask)
        mha = rearrange(concatted_heads, "... h seq_len d_v -> ... seq_len (h d_v)")
        multihead_self_attn = einsum(self.W_O, mha, " ... d_model h_d_v, ... h_d_v -> ... d_model")
        return multihead_self_attn


class Transformer(torch.nn.Module):
    """Transformer block w pre-norm attention and (swiglu) feed forward networks."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rms_norm_attention = RMSNorm(d_model=self.d_model, device=device)
        self.mha = MultiheadSelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            apply_rope=True,
            device=device,
        )  # Token Position
        self.rms_norm_swiglu = RMSNorm(d_model=self.d_model, device=device)
        self.swiglu = SwiGLU(d_model=self.d_model, d_ff=self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        # Attention calculation
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=x.device)
        normed_x = self.rms_norm_attention(x)
        new_x = self.mha(normed_x, token_positions=token_positions)
        x = x + new_x

        # FF NN
        normed_x = self.rms_norm_swiglu(x)
        new_x = self.swiglu(normed_x)
        x = x + new_x

        return x


class TransformerLM(torch.nn.Module):
    """Full Transformer Language Model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.theta = theta
        self.device = device
        self.dtype = dtype

        self.token_embedding = Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.d_model, device=self.device, dtype=self.dtype
        )
        self.transformers = torch.nn.ModuleList(
            [
                Transformer(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    theta=self.theta,
                    max_seq_len=self.context_length,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.rms_norm = RMSNorm(d_model=self.d_model, device=self.device, dtype=self.dtype)
        self.linear = Linear(
            in_features=self.d_model, out_features=self.vocab_size, device=self.device, dtype=self.dtype
        )

    def forward(self, x: torch.Tensor):  # input is tensor of token_ids
        embeddings = self.token_embedding(token_ids=x)

        # call transformer layers
        for t in self.transformers:
            embeddings = t(embeddings)

        # normed encodings takes in tensor of batch, seq_len, d_model
        normed_encodings = self.rms_norm(x=embeddings)

        # linearized must be of dimensions batch, seq_len, vocab_size (only next token?)
        linearized = self.linear(x=normed_encodings)

        return linearized
