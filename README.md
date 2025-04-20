# EmoMusicGen Web

一个基于情感的音乐生成Web应用程序，利用人工智能技术根据用户的情感状态生成相应的音乐。

## 功能

- 情感音乐生成
- 音频分析
- Web界面交互

## 技术栈

- Python
- Flask
- 音频处理库
- AI模型

## 安装与使用

1. 克隆仓库
```bash
git clone https://github.com/HunchoWebster/emomusicgen-web.git
```

2. 安装依赖
```bash
pip install -r audiocraft/webpage/requirements.txt
```

3. 运行应用
```bash
python audiocraft/webpage/app.py
```

## Docker部署

可以使用提供的Dockerfile进行Docker部署：

```bash
cd audiocraft/webpage
sh docker_deploy.sh
``` 