#！/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机结构生成器，生成满足特定数密度的原子结构（使用空间粗粒化优化）
@Time: 2025/11/05
@File: random_structure.py
@Author: Xuerui Wei
"""

import numpy as np
import random
import os
import sys
from typing import Tuple, List, Dict

# 常见元素的质量（原子质量单位）
ELEMENT_MASSES = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.811, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974, "S": 32.065,
    "Cl": 35.453, "Ar": 39.948, "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630,
    "As": 74.922, "Se": 78.96, "Br": 79.904, "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.96, "Tc": 98.0, "Ru": 101.07, "Rh": 102.906, "Pd": 106.42, "Ag": 107.868, "Cd": 112.411,
    "In": 114.818, "Sn": 118.710, "Sb": 121.760, "Te": 127.60, "I": 126.904, "Xe": 131.293, "Cs": 132.905, "Ba": 137.327,
    "La": 138.905, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242, "Pm": 145.0, "Sm": 150.36, "Eu": 151.964, "Gd": 157.25,
    "Tb": 158.925, "Dy": 162.500, "Ho": 164.930, "Er": 167.259, "Tm": 168.934, "Yb": 173.054, "Lu": 174.967, "Hf": 178.49,
    "Ta": 180.948, "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217, "Pt": 195.084, "Au": 196.967, "Hg": 200.59,
    "Tl": 204.383, "Pb": 207.2, "Bi": 208.980, "Po": 209.0, "At": 210.0, "Rn": 222.0, "Fr": 223.0, "Ra": 226.0,
    "Ac": 227.0, "Th": 232.038, "Pa": 231.036, "U": 238.029, "Np": 237.0, "Pu": 244.0, "Am": 243.0, "Cm": 247.0,
    "Bk": 247.0, "Cf": 251.0, "Es": 252.0, "Fm": 257.0, "Md": 258.0, "No": 259.0, "Lr": 262.0, "Rf": 267.0,
    "Db": 268.0, "Sg": 269.0, "Bh": 270.0, "Hs": 269.0, "Mt": 278.0, "Ds": 281.0, "Rg": 282.0, "Cn": 285.0,
    "Nh": 286.0, "Fl": 289.0, "Mc": 289.0, "Lv": 293.0, "Ts": 294.0, "Og": 294.0
}

def save_lammps_trajectory(filename: str, positions: np.ndarray, box_size: List[float], 
                          atom_types: List[int], element_types: List[str], step: int, 
                          num_atoms: int):
    """
    将原子结构保存为LAMMPS的轨迹文件格式(lammpstrj)的一帧
    
    参数:
    filename: 输出文件名
    positions: 原子坐标
    box_size: 盒子尺寸
    atom_types: 原子类型列表
    element_types: 元素类型名称列表
    step: 步数（当前原子数）
    num_atoms: 原子总数（目标原子数）
    """
    # 确保文件保存在当前目录下
    if not os.path.isabs(filename):
        filename = os.path.join(os.getcwd(), filename)
    
    # 使用UTF-8编码打开文件，避免编码错误
    mode = 'a' if os.path.exists(filename) and step > 1 else 'w'  # 第一帧用写入模式，后续帧用追加模式
    with open(filename, mode, encoding='utf-8') as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{step}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{step}\n")  # 使用当前实际原子数而不是总目标原子数
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"0.000000 {box_size[0]:.6f}\n")
        f.write(f"0.000000 {box_size[1]:.6f}\n")
        f.write(f"0.000000 {box_size[2]:.6f}\n")
        f.write("ITEM: ATOMS id type x y z\n")
        for i in range(step):  # 只写入当前实际存在的原子
            if i < len(atom_types):  # 确保不会越界
                f.write(f"{i+1} {atom_types[i]} {positions[i, 0]:.6f} {positions[i, 1]:.6f} {positions[i, 2]:.6f}\n")

def generate_random_structure(
    num_atoms: int,
    density: float,
    min_distance: float = 1.0,
    element_ratios: Dict[str, float] = None,
    generate_trajectory: bool = False,  # 添加轨迹生成开关参数
    use_ml_assisted_placement: bool = False  # 添加ML辅助放置开关
) -> Tuple[np.ndarray, List[float], List[int], List[str]]:
    """
    生成满足特定数密度的随机原子结构（使用空间粗粒化优化）
    
    参数:
    num_atoms: 原子数量
    density: 数密度 (atoms per cubic unit)
    min_distance: 原子间最小距离
    element_ratios: 元素类型及其比例，例如 {"Ti": 0.5, "Al": 0.5}
    generate_trajectory: 是否生成轨迹文件，默认为False
    use_ml_assisted_placement: 是否使用ML辅助原子放置，默认为False
    返回:
    positions: 原子坐标 (num_atoms x 3)
    box_size: 盒子边长 [lx, ly, lz]
    atom_types: 原子类型列表
    element_types: 元素类型名称列表
    """
    
    # 根据数密度计算盒子大小
    volume = num_atoms / density
    box_length = volume**(1/3)
    box_size = [box_length, box_length, box_length]
    
    # 初始化坐标数组
    positions = np.zeros((num_atoms, 3))
    atom_types = []  # 存储每个原子的类型（数字）
    element_types = []  # 存储每个原子的元素类型名称
    
    # 如果没有提供元素比例，则默认所有原子为Ti类型
    if element_ratios is None:
        element_ratios = {"Ti": 1.0}
    
    # 计算每个元素类型的原子数量
    elements = list(element_ratios.keys())
    ratios = list(element_ratios.values())
    
    # 创建原子类型的分布列表
    atom_distribution = []
    for i, element in enumerate(elements):
        count = round(num_atoms * ratios[i])
        atom_distribution.extend([element] * count)
    
    # 如果由于四舍五入导致总数不匹配，调整最后一个元素的数量
    if len(atom_distribution) != num_atoms:
        diff = num_atoms - len(atom_distribution)
        if diff > 0:
            # 添加缺少的原子到第一个元素类型
            atom_distribution.extend([elements[0]] * diff)
        else:
            # 删除多余的原子（从最后删除）
            atom_distribution = atom_distribution[:num_atoms]
    
    # 打乱原子分布顺序
    random.shuffle(atom_distribution)
    
    # 创建元素到类型编号的映射
    element_to_type = {element: i+1 for i, element in enumerate(elements)}
    
    # 空间粗粒化优化设置
    # 网格大小必须至少是min_distance，以确保所有可能影响当前网格的原子都在相邻网格中
    # 为了确保安全，我们使用min_distance * sqrt(3)作为网格大小，这样可以覆盖所有对角线方向的影响
    grid_size = min_distance * np.sqrt(3)
    nx = int(np.ceil(box_length / grid_size))
    ny = int(np.ceil(box_length / grid_size))
    nz = int(np.ceil(box_length / grid_size))
    
    # 初始化网格系统和密度统计
    grid = {}
    grid_density = np.zeros((nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                grid[(i, j, k)] = []
    
    # 获取网格坐标（考虑周期性边界条件）
    def get_grid_coords(pos):
        ix = int(pos[0] / grid_size) % nx
        iy = int(pos[1] / grid_size) % ny
        iz = int(pos[2] / grid_size) % nz
        return (ix, iy, iz)
    
    # 获取相邻网格（考虑周期性边界条件）
    # 扩大搜索范围以确保覆盖所有可能影响的原子
    def get_neighboring_grids(grid_coords):
        neighbors = []
        # 搜索范围需要根据网格大小和最小距离的关系来确定
        search_range = max(2, int(np.ceil(min_distance / grid_size)) + 1)
        for dx in range(-search_range, search_range + 1):
            for dy in range(-search_range, search_range + 1):
                for dz in range(-search_range, search_range + 1):
                    nx_coord = (grid_coords[0] + dx) % nx
                    ny_coord = (grid_coords[1] + dy) % ny
                    nz_coord = (grid_coords[2] + dz) % nz
                    neighbors.append((nx_coord, ny_coord, nz_coord))
        return list(set(neighbors))  # 去重
    
    # 添加第一个原子在盒子中心附近
    positions[0] = np.array([
        random.uniform(0, box_length),
        random.uniform(0, box_length),
        random.uniform(0, box_length)
    ])
    element_type = atom_distribution[0]
    atom_types.append(element_to_type[element_type])
    element_types.append(element_type)
    placed_atoms = 1
    
    # 更新第一个原子的网格信息
    first_grid_coord = get_grid_coords(positions[0])
    grid[first_grid_coord].append(0)
    grid_density[first_grid_coord] += 1
    
    # 如果需要生成轨迹，则保存第一个原子的构型作为轨迹的第一帧
    if generate_trajectory:
        save_lammps_trajectory("generation_trajectory.lammpstrj", 
                             positions[:placed_atoms], box_size, 
                             atom_types, element_types, placed_atoms, num_atoms)
    
    # 逐个添加原子，确保满足最小距离要求（考虑周期性边界条件和空间粗粒化优化）
    attempts = 0
    max_attempts = num_atoms * 1000  # 防止无限循环
    
    # 当放置过程变得困难时（尝试次数过多时），启用ML辅助
    ml_assisted_triggered = False
    
    while placed_atoms < num_atoms and attempts < max_attempts:
        # 检查是否需要启用ML辅助（当尝试次数过多时）
        if use_ml_assisted_placement and attempts > placed_atoms * 50:
            ml_assisted_triggered = True
        
        # 根据是否使用ML辅助来选择放置策略
        if ml_assisted_triggered:
            # 使用简单的低密度区域选择
            min_density = np.min(grid_density)
            low_density_indices = np.where(grid_density == min_density)
            if len(low_density_indices[0]) > 0:
                # 随机选择一个低密度网格
                idx = random.randint(0, len(low_density_indices[0])-1)
                grid_x = low_density_indices[0][idx]
                grid_y = low_density_indices[1][idx]
                grid_z = low_density_indices[2][idx]
            else:
                # 如果找不到，随机选择一个网格
                grid_x, grid_y, grid_z = random.randint(0, nx-1), random.randint(0, ny-1), random.randint(0, nz-1)
        else:
            # 优化策略：70%概率选择低密度区域，30%随机选择
            if random.random() < 0.7:
                # 找到密度最低的网格区域
                min_density = np.min(grid_density)
                low_density_indices = np.where(grid_density == min_density)
                if len(low_density_indices[0]) > 0:
                    # 随机选择一个低密度网格
                    idx = random.randint(0, len(low_density_indices[0])-1)
                    grid_x = low_density_indices[0][idx]
                    grid_y = low_density_indices[1][idx]
                    grid_z = low_density_indices[2][idx]
                else:
                    # 如果找不到，随机选择一个网格
                    grid_x, grid_y, grid_z = random.randint(0, nx-1), random.randint(0, ny-1), random.randint(0, nz-1)
            else:
                # 随机选择一个网格
                grid_x, grid_y, grid_z = random.randint(0, nx-1), random.randint(0, ny-1), random.randint(0, nz-1)
        
        # 在选定的网格内随机生成新原子位置
        x = random.uniform(grid_x * grid_size, min((grid_x + 1) * grid_size, box_length))
        y = random.uniform(grid_y * grid_size, min((grid_y + 1) * grid_size, box_length))
        z = random.uniform(grid_z * grid_size, min((grid_z + 1) * grid_size, box_length))
        new_pos = np.array([x, y, z])
        
        # 检查是否与相邻网格中的原子满足最小距离要求（考虑周期性边界条件）
        valid_position = True
        current_grid = get_grid_coords(new_pos)
        neighbor_grids = get_neighboring_grids(current_grid)
        
        for grid_coord in neighbor_grids:
            for atom_idx in grid.get(grid_coord, []):
                distance = calculate_periodic_distance(new_pos, positions[atom_idx], box_size)
                if distance < min_distance:
                    valid_position = False
                    break
            if not valid_position:
                break
        
        # 如果位置有效，则添加该原子
        if valid_position:
            positions[placed_atoms] = new_pos
            element_type = atom_distribution[placed_atoms]
            atom_types.append(element_to_type[element_type])
            element_types.append(element_type)
            placed_atoms += 1
            
            # 更新网格系统
            current_grid = get_grid_coords(new_pos)
            grid[current_grid].append(placed_atoms - 1)  # 存储索引
            grid_density[grid_x, grid_y, grid_z] += 1
            
            # 如果需要生成轨迹，则保存当前构型作为轨迹的一帧
            if generate_trajectory:
                save_lammps_trajectory("generation_trajectory.lammpstrj", 
                                     positions[:placed_atoms], box_size, 
                                     atom_types, element_types, placed_atoms, num_atoms)
            
            # 显示进度
            if placed_atoms % max(1, num_atoms // 100) == 0 or placed_atoms == num_atoms:
                progress = placed_atoms / num_atoms * 100
                sys.stdout.write(f'\r结构生成进度: {progress:.1f}% ({placed_atoms}/{num_atoms} 原子)')
                sys.stdout.flush()
            
        attempts += 1
    
    if placed_atoms < num_atoms:
        raise RuntimeError(f"无法在 {attempts} 次尝试内放置所有原子，请检查参数设置")
    
    print()  # 换行
    return positions, box_size, atom_types, element_types

def calculate_periodic_distance(pos1: np.ndarray, pos2: np.ndarray, box_size: List[float]) -> float:
    """
    计算考虑周期性边界条件的两点间距离
    
    参数:
    pos1: 第一个点的坐标
    pos2: 第二个点的坐标
    box_size: 盒子尺寸 [lx, ly, lz]
    
    返回:
    考虑周期性边界条件的最小距离
    """
    delta = np.abs(pos1 - pos2)
    
    # 应用周期性边界条件：如果距离超过盒子尺寸的一半，则使用镜像距离
    for i in range(3):
        if delta[i] > 0.5 * box_size[i]:
            delta[i] = box_size[i] - delta[i]
    
    return np.sqrt(np.sum(delta**2))

def save_lammps_data(filename: str, positions: np.ndarray, box_size: List[float], atom_types: List[int], element_types: List[str]):
    """
    将原子结构保存为LAMMPS的data格式
    
    参数:
    filename: 输出文件名
    positions: 原子坐标
    box_size: 盒子尺寸
    atom_types: 原子类型列表
    element_types: 元素类型名称列表
    """
    num_atoms = len(positions)
    unique_elements = list(dict.fromkeys(element_types))  # 保持顺序的去重
    unique_atom_types = list(set(atom_types))
    
    # 创建元素到类型编号的映射，保持与generate_random_structure中一致的顺序
    element_to_type = {}
    for i, element in enumerate(unique_elements):
        element_to_type[element] = i + 1
    
    # 确保文件保存在当前目录下
    if not os.path.isabs(filename):
        filename = os.path.join(os.getcwd(), filename)
    
    # 使用UTF-8编码打开文件，避免编码错误
    with open(filename, 'w', encoding='utf-8') as f:
        # 写入文件头
        f.write(f"LAMMPS data file generated by random-structure.py\n\n")
        
        # 写入原子和原子类型的数量
        f.write(f"{num_atoms} atoms\n")
        f.write(f"{len(unique_atom_types)} atom types\n\n")
        
        # 写入盒子边界
        f.write(f"0.000000 {box_size[0]:.6f} xlo xhi\n")
        f.write(f"0.000000 {box_size[1]:.6f} ylo yhi\n")
        f.write(f"0.000000 {box_size[2]:.6f} zlo zhi\n\n")
        
        # 写入质量
        f.write(f"Masses\n\n")
        for element, type_id in element_to_type.items():
            mass = ELEMENT_MASSES.get(element, 1.0)  # 如果找不到元素，使用默认质量1.0
            f.write(f"{type_id} {mass:.4f}\n")
        f.write("\n")
        
        # 写入原子坐标
        f.write(f"Atoms\n\n")
        for i in range(num_atoms):
            # 在LAMMPS中，原子编号从1开始
            f.write(f"{i+1} {atom_types[i]} {positions[i, 0]:.6f} {positions[i, 1]:.6f} {positions[i, 2]:.6f}\n")

if __name__ == "__main__":
    # 示例：生成包含50个原子的结构，数密度为0.05 atoms/unit³，最小距离为1.5
    # 包含两种元素：Ti占70%，Al占30%
    try:
        positions, box_size, atom_types, element_types = generate_random_structure(
            num_atoms=200,  # 使用较小的原子数进行测试
            density=0.0784,
            min_distance=1.5,
            element_ratios={"Fe": 0.8, "B": 0.2},
            generate_trajectory=False,  # 默认开启轨迹生成功能
            use_ml_assisted_placement=True  # 启用ML辅助放置
        )
        
        save_lammps_data("random_structure_FeB.data", positions, box_size, atom_types, element_types)
        
        # 打印一些统计信息
        actual_density = len(positions) / (box_size[0] * box_size[1] * box_size[2])
        print(f"成功生成随机结构!")
        print(f"原子数量: {len(positions)}")
        print(f"盒子尺寸: {box_size[0]:.3f} x {box_size[1]:.3f} x {box_size[2]:.3f}")
        print(f"实际数密度: {actual_density:.4f} atoms/unit³")
        print(f"LAMMPS数据文件保存路径: {os.path.join(os.getcwd(), 'random_structure.data')}")
        if os.path.exists(os.path.join(os.getcwd(), 'generation_trajectory.lammpstrj')):
            print(f"LAMMPS轨迹文件保存路径: {os.path.join(os.getcwd(), 'generation_trajectory.lammpstrj')}")
        else:
            print("未生成轨迹文件（根据设置）")
        
        # 统计各元素数量
        element_counts = {}
        for element in element_types:
            element_counts[element] = element_counts.get(element, 0) + 1
        print("元素分布:")
        for element, count in element_counts.items():
            print(f"  {element}: {count} 个原子 ({count/len(element_types)*100:.1f}%)")  
    except RuntimeError as e:
        print(f"错误: {e}")