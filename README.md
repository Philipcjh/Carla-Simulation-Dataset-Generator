# Carla-Simulation-Dataset-Generator

## 项目简介
本项目是针对在CARLA中自动生成和整理路侧仿真数据集而建立的，能够根据KITTI数据集格式自动生成和整理路侧数据集（RGB图像、激光雷达点云及其对应的标签文件）。

## 项目框架
本项目的框架如下图所示
![image](framework.jpg)

## 环境配置 
在环境为[Python>=3.6](https://www.python.org)和[carla >= 0.9.12](https://carla.readthedocs.io/en/0.9.12/)中进行下列安装：

    git clone https://github.com/Philipcjh/Carla-Simulation-Dataset-Generator  # clone
    cd Carla-Simulation-Dataset-Generator
    pip install -r requirements.txt  # install

其中carla包的安装需要从本地导入，具体方法参考[基础API的使用](https://zhuanlan.zhihu.com/p/340031078)第0部分。

## 使用方法
在CARLA中点击play，然后运行[main.py](main.py)文件。

## 参考代码
本项目是参考[mmmmaomao/DataGenerator](https://github.com/mmmmaomao/DataGenerator)的代码修改而得到的。