"""
示例用法
展示如何使用 src/ 目录中的Python代码
"""

import sys
import os

# 将src目录添加到Python路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from example import greet

if __name__ == "__main__":
    print(greet("Python Developer"))
    
    # 更多示例代码可以放在这里