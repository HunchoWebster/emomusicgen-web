FROM python:3.9-slim

WORKDIR /app

# 复制必要的文件
COPY app.py requirements.txt /app/
COPY static /app/static/
COPY templates /app/templates/
COPY *.py /app/

# 安装编译工具和依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装基本依赖
RUN pip install --no-cache-dir flask==2.0.1 soundfile numpy torch torchaudio librosa transformers scipy gunicorn

# 创建必要的目录
RUN mkdir -p /app/static/generated_music

# 暴露端口
EXPOSE 7860

# 启动应用
CMD ["python", "app.py"]