from math import isqrt

import torch
from e3nn.o3 import matrix_to_angles, wigner_D
from einops import einsum
from jaxtyping import Float
from torch import Tensor


def rotate_sh(
    sh_coefficients: Float[Tensor, "*#batch n"],
    rotations: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch n"]:
    device = sh_coefficients.device
    dtype = sh_coefficients.dtype
    basis_perm = torch.tensor(
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        dtype=dtype,
        device=device,
    )
    inverse_basis_perm = torch.tensor(
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        dtype=dtype,
        device=device,
    )
    permuted_rotation_matrix = inverse_basis_perm @ rotations @ basis_perm
    *_, n = sh_coefficients.shape
    alpha, beta, gamma = matrix_to_angles(permuted_rotation_matrix)
    result = []
    for degree in range(isqrt(n)):
        sh_rotations = wigner_D(degree, alpha, -beta, gamma).type(dtype)
        sh_rotated = einsum(
            sh_rotations,
            sh_coefficients[..., degree**2 : (degree + 1) ** 2],
            "... i j, ... j -> ... i",
        )
        result.append(sh_rotated)
    return torch.cat(result, dim=-1)
