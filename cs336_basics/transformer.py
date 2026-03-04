import torch
from einops import einsum, rearrange


class Linear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, device: torch.device | None =None, dtype: torch.dtype | None =None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.w = torch.empty(self.out_features, self.in_features)
        std = (2/(self.in_features + self.out_features))**(0.5)
        torch.nn.init.trunc_normal_(
            tensor=self.w,
            mean=0,
            std=std,
            a=-3*std, 
            b=3*std,
        )
        self.W = torch.nn.Parameter(self.w)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        print("x: ", x.shape)
        print("W: ", self.W.shape)
        
        y = einsum(self.W, x, "out in, batch seq_len in -> batch out" )

        return y