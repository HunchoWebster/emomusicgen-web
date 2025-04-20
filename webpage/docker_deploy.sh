#!/bin/bash

# 本地构建和部署脚本

echo "=== 开始Docker部署流程 ==="

# 构建Docker镜像
echo "正在构建Docker镜像..."
docker build -t emomusicgen-web .

# 提示用户选择部署方式
echo "请选择部署方式:"
echo "1. 保存镜像为文件并传输到服务器"
echo "2. 推送到Docker Hub再在服务器拉取"
read -p "请输入选项 (1/2): " deploy_option

if [ "$deploy_option" = "1" ]; then
    # 保存镜像为文件
    echo "正在保存镜像为文件..."
    docker save -o emomusicgen-web.tar emomusicgen-web
    
    # 传输文件到服务器
    echo "正在传输镜像到服务器..."
    scp emomusicgen-web.tar root@172.19.52.21:~/
    
    echo "镜像已传输到服务器，请在服务器执行以下命令:"
    echo "-------------------------------"
    echo "docker load -i ~/emomusicgen-web.tar"
    echo "docker run -d -p 7860:7860 --name emomusicgen emomusicgen-web"
    echo "-------------------------------"
    
elif [ "$deploy_option" = "2" ]; then
    # 登录Docker Hub
    echo "请登录Docker Hub..."
    docker login
    
    # 提示用户输入Docker Hub用户名
    read -p "请输入Docker Hub用户名: " docker_username
    
    # 标记镜像
    echo "正在标记镜像..."
    docker tag emomusicgen-web $docker_username/emomusicgen-web:latest
    
    # 推送镜像
    echo "正在推送镜像到Docker Hub..."
    docker push $docker_username/emomusicgen-web:latest
    
    echo "镜像已推送到Docker Hub，请在服务器执行以下命令:"
    echo "-------------------------------"
    echo "docker pull $docker_username/emomusicgen-web:latest"
    echo "docker run -d -p 7860:7860 --name emomusicgen $docker_username/emomusicgen-web:latest"
    echo "-------------------------------"
    
else
    echo "无效选项，部署已取消"
    exit 1
fi

echo "=== 本地部署步骤完成 ===" 