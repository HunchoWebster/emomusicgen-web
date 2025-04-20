#!/bin/bash

# 服务器端设置脚本

echo "=== 开始服务器设置 ==="

# 更新包列表
echo "正在更新包列表..."
apt update

# 安装Docker
echo "正在安装Docker..."
apt install -y docker.io

# 启动Docker服务
echo "正在启动Docker服务..."
systemctl enable docker
systemctl start docker

echo "Docker已安装并启动"
echo "请按照本地脚本输出的说明加载和运行Docker镜像"

echo "=== 服务器设置完成 ===" 