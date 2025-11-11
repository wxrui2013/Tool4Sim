"""
测试示例文件
展示如何为Python代码编写测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from example import greet

def test_greet():
    """测试greet函数"""
    result = greet("World")
    assert "Hello, World!" in result
    assert "useful Python code repository" in result

def test_greet_empty_string():
    """测试空字符串输入"""
    result = greet("")
    assert "Hello, !" in result

if __name__ == "__main__":
    test_greet()
    test_greet_empty_string()
    print("所有测试通过！")