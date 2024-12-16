import torch
from torch import Tensor

# lib = torch.library.Library("autogptq", "FRAGMENT")
# lib.define("vecquant3matmul_faster_old_256(Tensor vec, Tensor mat, Tensor mul, Tensor scales, Tensor zeros, int groupsize, int vec_height) -> Tensor")

def vecquant3matmul_faster_old_256(vec: Tensor, mat: Tensor, mul: Tensor, scales: Tensor, zeros: Tensor, groupsize: int, vec_height: int) -> Tensor:
    # matvec
    # in_featuers = mat.shape[0] * 32 // 3
    # out_features = mat.shape[1]
    # mul = torch.zeros([vec.shape[0], out_features], dtype=torch.float, device=mat.device)
    import code; code.interact('vecquant3matmul_faster_old_256', local=dict(globals(), **locals()))

    torch.ops.autogptq.vecquant3matmul_faster_old_256.default(
        vec=vec,
        mat=mat,
        mul=mul,
        scales=scales,
        zeros=zeros,
        groupsize=groupsize,
        vec_height=vec_height,
    )
    
    return mul

@torch.library.register_fake("autogptq::vecquant3matmul_faster_old_256")
def _(vec: Tensor, mat: Tensor, mul: Tensor, scales: Tensor, zeros: Tensor, groupsize: int, vec_height: int) -> Tensor:
    out_features = mat.shape[1]
    return torch.empty(
        [vec.shape[0], out_features], device=vec.device, dtype=torch.float
    )