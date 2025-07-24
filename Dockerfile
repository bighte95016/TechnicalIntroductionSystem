FROM python:3.9-slim

# 設置工作目錄
WORKDIR /app

# 安裝系統依賴項
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1-dev \
    ffmpeg \
    espeak-ng \
    espeak-ng-data \
    libespeak-ng-dev \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 複製需求文件並安裝Python依賴項
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程序文件
COPY . .

# 創建必要的目錄
RUN mkdir -p ./bar_file/PDF \
    && mkdir -p ./chroma_db \
    && mkdir -p ./bm25_cache \
    && mkdir -p ./voice \
    && mkdir -p ./index_tts

# 設置環境變量
ENV PYTHONPATH=/app
ENV ANONYMIZED_TELEMETRY=False

# 暴露端口（如果需要）
EXPOSE 8000

# 設置啟動命令
CMD ["python", "roboticBar_printChunk_QAsplit_indexTTSclone_JP.py"] 