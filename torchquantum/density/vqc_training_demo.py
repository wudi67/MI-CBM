#!/usr/bin/env python3
"""
两量子比特密度矩阵VQC训练演示
展示如何创建可训练的变分量子电路并进行反向传播
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import sys
import os
import numpy as np

# 添加路径以便导入torchquantum
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torchquantum as tq


class DensityVQC(nn.Module):
    """两量子比特密度矩阵变分量子电路"""
    
    def __init__(self, n_wires=2, n_layers=3):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        
        # 可训练参数 - 每层包含RY和RZ旋转门参数
        self.ry_params = nn.Parameter(torch.randn(n_layers, n_wires) * 0.1)
        self.rz_params = nn.Parameter(torch.randn(n_layers, n_wires) * 0.1)
        
        print(f"初始化VQC: {n_layers}层, {n_wires}量子比特")
        print(f"RY参数形状: {self.ry_params.shape}")
        print(f"RZ参数形状: {self.rz_params.shape}")
        print(f"总参数数量: {self.ry_params.numel() + self.rz_params.numel()}")
    
    def forward(self, bsz=1, input_states=None):
        """前向传播构建VQC
        Args:
            bsz: 批次大小
            input_states: 输入量子态，如果提供则用于初始化量子设备
        """
        # 创建密度矩阵设备
        device = tq.NoiseDevice(n_wires=self.n_wires, bsz=bsz)
        
        # 如果提供了输入态，则将其编码到量子设备中
        if input_states is not None:
            # 将输入态编码为密度矩阵
            batch_densities = []
            for i in range(bsz):
                if i < len(input_states):
                    input_state = input_states[i]
                    # 创建密度矩阵 ρ = |ψ⟩⟨ψ|
                    rho = torch.outer(input_state.conj(), input_state)
                    # 重塑为设备期望的形状 [2, 2, 2, 2]
                    rho_reshaped = rho.reshape([2, 2, 2, 2])
                    batch_densities.append(rho_reshaped)
                else:
                    # 如果批次中没有足够的输入态，使用默认初始态
                    batch_densities.append(device.densities[0])
            
            # 堆叠所有密度矩阵并设置到设备
            if batch_densities:
                device.densities = torch.stack(batch_densities)
        
        # VQC层级结构
        for layer in range(self.n_layers):
            # 1. RY旋转门层
            for wire in range(self.n_wires):
                device.ry(wires=wire, params=self.ry_params[layer, wire])
            
            # 2. RZ旋转门层  
            for wire in range(self.n_wires):
                device.rz(wires=wire, params=self.rz_params[layer, wire])
            
            # 3. 纠缠层 - CNOT门
            if self.n_wires > 1:
                for wire in range(self.n_wires - 1):
                    device.cnot(wires=[wire, wire + 1])
                # 环形纠缠（最后一个和第一个相连）
                if self.n_wires > 2:
                    device.cnot(wires=[self.n_wires - 1, 0])
        
        return device
    
    def get_expectation_pauli_z(self, device, wire=0):
        """计算指定量子比特的Pauli-Z期望值"""
        # 获取密度矩阵并转换为2D形式
        bsz = device.densities.shape[0]
        rho_2d = device.densities.reshape(bsz, 2**self.n_wires, 2**self.n_wires)
        
        # Pauli-Z算子对应的对角矩阵 (对2比特系统)
        if self.n_wires == 2:
            if wire == 0:
                # Z⊗I: diag([1, 1, -1, -1])
                pauli_z = torch.diag(torch.tensor([1., 1., -1., -1.], 
                                                 dtype=torch.complex64, 
                                                 device=rho_2d.device))
            elif wire == 1:
                # I⊗Z: diag([1, -1, 1, -1])
                pauli_z = torch.diag(torch.tensor([1., -1., 1., -1.], 
                                                 dtype=torch.complex64, 
                                                 device=rho_2d.device))
        
        # 计算期望值 tr(ρ * Z) 对每个批次
        batch_expectations = []
        for i in range(bsz):
            expectation = torch.real(torch.trace(rho_2d[i] @ pauli_z))
            batch_expectations.append(expectation)
        
        return torch.stack(batch_expectations)[0] if bsz == 1 else torch.stack(batch_expectations)
    
    def get_fidelity_with_target(self, device, target_state):
        """计算与目标态的保真度"""
        bsz = device.densities.shape[0]
        rho = device.densities.reshape(bsz, 2**self.n_wires, 2**self.n_wires)
        
        # # 目标态的密度矩阵
        # target_rho = torch.outer(target_state.conj(), target_state)
        # target_rho = target_rho.to(rho.device)
        
        # 对于单个目标态与批次密度矩阵的保真度计算
        batch_fidelities = []
        for i in range(bsz):
            target_rho = torch.outer(target_state[i].conj(), target_state[i])
            fidelity = torch.real(torch.trace(rho[i] @ target_rho))
            batch_fidelities.append(fidelity)
        
        if bsz == 1:
            return batch_fidelities[0]
        else:
            return torch.stack(batch_fidelities)


def print_tensor_info(tensor, name):
    """打印张量信息"""
    print(f"{name}:")
    print(f"  值: {tensor.item() if tensor.numel() == 1 else tensor}")
    print(f"  梯度: {tensor.grad}")
    print(f"  requires_grad: {tensor.requires_grad}")


def demo_vqc_training():
    """演示VQC训练过程"""
    print("🚀 密度矩阵VQC训练演示")
    print("=" * 60)
    
    # 1. 创建VQC模型
    print("\n1. 创建VQC模型")
    vqc = DensityVQC(n_wires=2, n_layers=3)
    
    # 2. 生成随机训练数据
    print("\n2. 生成随机训练数据")
    n_samples = 100
    batch_size = 16
    
    # 生成随机量子态对 (X, Y)
    np.random.seed(42)  # 固定随机种子确保可重复性
    torch.manual_seed(42)
    
    # 生成输入量子态 X (100个两比特量子态)
    X_real = torch.randn(n_samples, 4) * 0.5
    X_imag = torch.randn(n_samples, 4) * 0.5
    X_states = torch.complex(X_real, X_imag)
    X_states = X_states / torch.norm(X_states, dim=1, keepdim=True)
    
    # 生成输出量子态 Y (100个两比特量子态)
    Y_real = torch.randn(n_samples, 4) * 0.5
    Y_imag = torch.randn(n_samples, 4) * 0.5
    Y_states = torch.complex(Y_real, Y_imag)
    Y_states = Y_states / torch.norm(Y_states, dim=1, keepdim=True)
    
    print(f"生成 {n_samples} 对随机量子态 (X, Y)")
    print(f"批处理大小: {batch_size}")
    print(f"输入态X形状: {X_states.shape}")
    print(f"输出态Y形状: {Y_states.shape}")
    print(f"前3对量子态示例:")
    for i in range(3):
        print(f"  样本 {i}:")
        print(f"    X态: {X_states[i]}")
        print(f"    Y态: {Y_states[i]}")
        print(f"    X模长: {torch.norm(X_states[i]).item():.6f}")
        print(f"    Y模长: {torch.norm(Y_states[i]).item():.6f}")
        print(f"    X·Y*: {torch.abs(torch.vdot(X_states[i], Y_states[i])).item():.6f}")  # 内积模长
    
    # 3. 设置优化器
    print("\n3. 设置优化器")
    optimizer = optim.Adam(vqc.parameters(), lr=0.01)  # 降低学习率适应批处理
    print(f"优化器: Adam, 学习率: 0.01")
    
    # 4. 创建数据加载器
    dataset = torch.utils.data.TensorDataset(X_states, Y_states)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"数据加载器: 批大小={batch_size}, 总批次={len(dataloader)}")
    print(f"每个批次包含: {batch_size}对(X,Y)量子态")
    
    # 5. 训练循环
    print("\n5. 开始训练")
    print("-" * 50)
    
    n_epochs = 20
    all_losses = []
    epoch_losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (batch_X, batch_Y) in enumerate(dataloader):
            current_batch_size = batch_X.size(0)
            
            optimizer.zero_grad()
            
            # 前向传播 - 使用输入态X初始化，然后经过VQC变换
            device = vqc.forward(bsz=current_batch_size, input_states=batch_X)
            
            # 计算批次损失 - VQC输出与目标态Y的保真度
                       
            batch_fidelities = vqc.get_fidelity_with_target(device, batch_Y)
                
            
            # 平均保真度和损失
            avg_fidelity = batch_fidelities.mean()
            loss = 1 - avg_fidelity
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # 每5个批次打印一次
            if batch_idx % 5 == 0:
                print(f"  Epoch {epoch:2d}, Batch {batch_idx:2d}: "
                      f"Loss={loss.item():.6f}, Fidelity={avg_fidelity.item():.6f}")
                if batch_idx == 0:  # 显示第一个批次的样例
                    print(f"    样例: X→Y 映射保真度: {batch_fidelities[0].item():.6f}")
        
        # 记录epoch平均损失
        avg_epoch_loss = epoch_loss / batch_count
        epoch_losses.append(avg_epoch_loss)
        all_losses.extend([avg_epoch_loss] * batch_count)
        
        print(f"Epoch {epoch:2d} 完成: 平均损失={avg_epoch_loss:.6f}")
    
    print("-" * 50)
    print(f"训练完成! 最终平均损失: {epoch_losses[-1]:.6f}")
    
    # 6. 测试阶段 - 在测试集上评估
    print("\n6. 测试阶段评估")
    
    # 生成测试数据对（与训练数据不同）
    test_X_real = torch.randn(10, 4) * 0.3
    test_X_imag = torch.randn(10, 4) * 0.3
    test_X = torch.complex(test_X_real, test_X_imag)
    test_X = test_X / torch.norm(test_X, dim=1, keepdim=True)
    
    test_Y_real = torch.randn(10, 4) * 0.3
    test_Y_imag = torch.randn(10, 4) * 0.3
    test_Y = torch.complex(test_Y_real, test_Y_imag)
    test_Y = test_Y / torch.norm(test_Y, dim=1, keepdim=True)
    
    with torch.no_grad():
        test_fidelities = []
        
        # 逐个测试每对(X,Y)
        for i in range(10):
            # 用输入态X[i]初始化VQC，产生输出
            test_device = vqc.forward(bsz=1, input_states=[test_X[i]])
            
            # 计算VQC输出与目标态Y[i]的保真度
            fidelity = vqc.get_fidelity_with_target(test_device, test_Y)
            if isinstance(fidelity, torch.Tensor):
                fidelity = fidelity.item()
            test_fidelities.append(fidelity)
        
        avg_test_fidelity = np.mean(test_fidelities)
        print(f"测试集平均保真度: {avg_test_fidelity:.6f}")
        print(f"测试集保真度标准差: {np.std(test_fidelities):.6f}")
        print(f"最佳测试保真度: {max(test_fidelities):.6f}")
        print(f"最差测试保真度: {min(test_fidelities):.6f}")
        
        # 显示几个具体的测试样例
        print(f"\n具体测试样例:")
        for i in range(3):
            print(f"  样例 {i}: X→Y 保真度 = {test_fidelities[i]:.6f}")
            print(f"    输入态模长: {torch.norm(test_X[i]).item():.6f}")
            print(f"    目标态模长: {torch.norm(test_Y[i]).item():.6f}")
    
    # 7. 分析一个具体的测试样例
    print("\n7. 分析具体测试样例")
    with torch.no_grad():
        # 选择第一个测试样例进行详细分析
        sample_X = test_X[0]
        sample_Y = test_Y[0]
        
        sample_device = vqc.forward(bsz=1, input_states=[sample_X])
        sample_fidelity = vqc.get_fidelity_with_target(sample_device, sample_Y)
        
        # 打印相关密度矩阵
        final_rho = sample_device.densities[0].reshape(4, 4)
        input_rho = torch.outer(sample_X.conj(), sample_X)
        target_rho = torch.outer(sample_Y.conj(), sample_Y)
        
        print("输入密度矩阵 ρ_X (实部):")
        print(input_rho.real.numpy())
        print("\nVQC输出密度矩阵 ρ_out (实部):")
        print(final_rho.real.numpy())
        print("\n目标密度矩阵 ρ_Y (实部):")
        print(target_rho.real.numpy())
        print(f"\n样例保真度 F(ρ_out, ρ_Y): {sample_fidelity.item():.6f}")
        
        # 计算输入与目标的直接保真度（作为参考）
        direct_fidelity = torch.real(torch.trace(input_rho @ target_rho))
        print(f"直接保真度 F(ρ_X, ρ_Y): {direct_fidelity.item():.6f}")
        print(f"VQC改善: {sample_fidelity.item() - direct_fidelity.item():.6f}")
        
        # 检查VQC输出密度矩阵性质
        trace = torch.trace(final_rho)
        purity = torch.trace(final_rho @ final_rho)
        eigenvals = torch.linalg.eigvals(final_rho)
        min_eigenval = torch.min(eigenvals.real)
        
        print(f"\nVQC密度矩阵性质:")
        print(f"迹: {trace.real.item():.6f} (应≈1)")
        print(f"纯度: {purity.real.item():.6f} (纯态≈1)")
        print(f"最小特征值: {min_eigenval.item():.6f} (应≥0)")
        print(f"正半定: {min_eigenval.item() >= -1e-6}")
    
    # 8. 训练统计
    print(f"\n8. 训练统计:")
    print(f"总训练样本: {n_samples}")
    print(f"总训练批次: {len(dataloader) * n_epochs}")
    print(f"初始平均损失: {epoch_losses[0]:.6f}")
    print(f"最终平均损失: {epoch_losses[-1]:.6f}")
    print(f"损失改善: {epoch_losses[0] - epoch_losses[-1]:.6f}")
    
    return vqc, epoch_losses, test_fidelities


def test_gradients():
    """测试梯度计算"""
    print("\n" + "="*60)
    print("🔬 梯度计算测试")
    print("="*60)
    
    vqc = DensityVQC(n_wires=2, n_layers=1)
    
    # 简单的前向传播
    device = vqc.forward(bsz=1)
    
    # 计算一个简单的损失函数：最小化第一个量子比特的Z期望值
    z_expectation = vqc.get_expectation_pauli_z(device, wire=0)
    loss = z_expectation**2  # 目标：使Z期望值接近0
    
    print(f"损失函数: ⟨Z₀⟩² = {loss.item():.6f}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print("\n梯度检查:")
    if vqc.ry_params.grad is not None:
        print(f"RY参数梯度范数: {torch.norm(vqc.ry_params.grad).item():.6f}")
        print(f"RY参数梯度: {vqc.ry_params.grad.flatten()}")
    else:
        print("RY参数梯度为None!")
        
    if vqc.rz_params.grad is not None:
        print(f"RZ参数梯度范数: {torch.norm(vqc.rz_params.grad).item():.6f}")
        print(f"RZ参数梯度: {vqc.rz_params.grad.flatten()}")
    else:
        print("RZ参数梯度为None!")


def main():
    """主函数"""
    # 运行梯度测试
    test_gradients()
    
    # 运行VQC训练演示
    vqc, epoch_losses, test_fidelities = demo_vqc_training()
    
    print("\n" + "="*60)
    print("🎉 演示完成!")
    print("="*60)
    print("✅ 成功验证:")
    print("  - 密度矩阵VQC构建")
    print("  - 随机数据生成 (100个样本)")
    print("  - 批处理训练 (batch_size=16)")
    print("  - 可训练参数定义")
    print("  - 反向传播计算")
    print("  - 梯度下降优化")
    print("  - 保真度损失函数")
    print("  - 期望值计算")
    print("  - 测试集评估")
    
    print(f"\n📊 最终结果:")
    print(f"  训练损失改善: {epoch_losses[0] - epoch_losses[-1]:.6f}")
    print(f"  测试集平均保真度: {np.mean(test_fidelities):.6f}")
    print(f"  测试集最佳保真度: {max(test_fidelities):.6f}")


if __name__ == "__main__":
    main()
