import sys
import os
__dir__=os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)  # 将当前文件夹添加到系统路径中
sys.path.append(os.path.abspath(os.path.join(__dir__,'..'))) # 将上一级文件夹添加到系统路径中
