"""
示例用法
展示如何使用 src/ 目录中的Python代码
"""

import sys
import os

# 将src目录添加到Python路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# 导入模块
import random_structure

if __name__ == "__main__":
    # 示例：生成包含50个原子的结构，数密度为0.05 atoms/unit³，最小距离为1.5
    # 包含两种元素：Ti占70%，Al占30%
    try:
        positions, box_size, atom_types, element_types = random_structure.generate_random_structure(
            num_atoms=20,  # 使用较小的原子数进行测试
            density=0.05,
            min_distance=1.0,
            element_ratios={"Ti": 0.7, "Al": 0.3} 
        )
        
        random_structure.save_lammps_data("examples/test_structure.data", positions, box_size, atom_types, element_types)
        
        # 打印一些统计信息
        actual_density = len(positions) / (box_size[0] * box_size[1] * box_size[2])
        print(f"成功生成随机结构!")
        print(f"原子数量: {len(positions)}")
        print(f"盒子尺寸: {box_size[0]:.3f} x {box_size[1]:.3f} x {box_size[2]:.3f}")
        print(f"实际数密度: {actual_density:.4f} atoms/unit³")
        print(f"LAMMPS数据文件保存路径: {os.path.join(os.getcwd(), 'test_structure.data')}")
        
        # 统计各元素数量
        element_counts = {}
        for element in element_types:
            element_counts[element] = element_counts.get(element, 0) + 1
        print("元素分布:")
        for element, count in element_counts.items():
            print(f"  {element}: {count} 个原子 ({count/len(element_types)*100:.1f}%)")  
    except RuntimeError as e:
        print(f"错误: {e}")