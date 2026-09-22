import torch

from torchquantum.density import density_mat
from torchquantum.density import density_func
from torchquantum.functional import mat_dict

if __name__ == "__main__":
    mat = mat_dict["hadamard"]

    Xgatemat = mat_dict["paulix"]
    # Xgatemat = torch.tensor([[0., 1.], [1., 0.]])

    print(mat)
    D = density_mat.DensityMatrix(2, 2)

    rho = torch.zeros(2 ** 4,)
    rho = torch.reshape(rho, [4, 4])
    rho[0][0] = 1 / 2
    rho[0][3] = 1 / 2
    rho[3][0] = 1 / 2
    rho[3][3] = 1 / 2
    rho = torch.reshape(rho, [2, 2, 2, 2])
    # 添加批次维度
    rho = rho.unsqueeze(0)  # 形状变为 [1, 2, 2, 2, 2]
    D.update_matrix(rho)
    D.print_2d(0)
    print("Input density shape:", D._matrix.shape)
    newD = density_func.apply_unitary_density_bmm(D._matrix, Xgatemat, [1])
    print("Output density shape:", newD.shape)

    D.update_matrix(newD)

    D.print_2d(0)

