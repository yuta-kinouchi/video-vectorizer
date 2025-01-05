FROM python:3.9-slim

# 作業ディレクトリの設定
WORKDIR /app

# 必要なパッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY . .

# 環境変数の設定
ENV PORT 8080

# Gunicornを使用してサービスを起動
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app