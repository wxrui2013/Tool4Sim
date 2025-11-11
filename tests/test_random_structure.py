"""
测试random_structure模块
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import random_structure
import numpy as np

def test_generate_random_structure():
    """测试随机结构生成功能"""
    # 生成一个小型结构进行测试
    positions, box_size, atom_types, element_types = random_structure.generate_random_structure(
        num_atoms=10,
        density=0.05,
        min_distance=1.0,
        element_ratios={"Ti": 0.5, "Al": 0.5}
    )
    
    # 检查返回值的类型和形状
    assert isinstance(positions, np.ndarray)
    assert positions.shape == (10, 3)
    assert len(box_size) == 3
    assert len(atom_types) == 10
    assert len(element_types) == 10
    
    # 检查原子类型是否正确
    unique_elements = set(element_types)
    assert unique_elements.issubset({"Ti", "Al"})
    
    print("随机结构生成测试通过!")

def test_calculate_periodic_distance():
    """测试周期性边界条件距离计算"""
    box_size = [10.0, 10.0, 10.0]
    
    # 测试相同点
    pos1 = np.array([1.0, 1.0, 1.0])
    pos2 = np.array([1.0, 1.0, 1.0])
    distance = random_structure.calculate_periodic_distance(pos1, pos2, box_size)
    assert distance == 0.0
    
    # 测试简单距离
    pos1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([3.0, 4.0, 0.0])
    distance = random_structure.calculate_periodic_distance(pos1, pos2, box_size)
    assert abs(distance - 5.0) < 1e-10
    
    print("周期性距离计算测试通过!")

def test_lammps_mass_consistency():
    """测试LAMMPS数据文件中元素与质量的一致性"""
    # 生成一个测试结构
    positions, box_size, atom_types, element_types = random_structure.generate_random_structure(
        num_atoms=10,
        density=0.05,
        min_distance=1.0,
        element_ratios={"Ti": 0.5, "Al": 0.5}
    )
    
    # 保存为LAMMPS数据文件
    filename = "test_mass_consistency.data"
    random_structure.save_lammps_data(filename, positions, box_size, atom_types, element_types)
    
    # 读取文件检查质量
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 找到质量部分
    masses_start = -1
    for i, line in enumerate(lines):
        if line.strip() == "Masses":
            masses_start = i
            break
    
    assert masses_start != -1, "未找到Masses部分"
    
    # 检查质量是否正确
    # 第一个元素应该是Ti，质量为47.867
    # 第二个元素应该是Al，质量为26.982
    mass_line_1 = lines[masses_start + 2].strip().split()
    mass_line_2 = lines[masses_start + 3].strip().split()
    
    # 检查质量值是否正确
    assert abs(float(mass_line_1[1]) - 47.867) < 0.001 or abs(float(mass_line_1[1]) - 26.982) < 0.001
    assert abs(float(mass_line_2[1]) - 47.867) < 0.001 or abs(float(mass_line_2[1]) - 26.982) < 0.001
    
    # 清理测试文件
    os.remove(filename)
    
    print("LAMMPS质量一致性测试通过!")

if __name__ == "__main__":
    test_generate_random_structure()
    test_calculate_periodic_distance()
    test_lammps_mass_consistency()
    print("所有测试通过！")