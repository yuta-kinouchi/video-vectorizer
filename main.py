from flask import Flask, request, jsonify
from video_vectorizer import VideoVectorizer
import os
from typing import Dict, Any, Union

app = Flask(__name__)

# 環境変数から設定を読み込み
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT_ID")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "asia-northeast1")
VECTOR_INDEX_ID = os.environ.get("VECTOR_INDEX_ID")

# VideoVectorizerのインスタンスを作成
vectorizer = VideoVectorizer(
    project_id=PROJECT_ID, location=LOCATION, vector_index_id=VECTOR_INDEX_ID
)


def create_error_response(
    message: str, status_code: int = 400
) -> tuple[Dict[str, str], int]:
    """
    エラーレスポンスを作成する
    """
    return jsonify({"error": message}), status_code


@app.route("/health", methods=["GET"])
def health_check() -> Dict[str, str]:
    """
    ヘルスチェックエンドポイント
    """
    return jsonify({"status": "healthy"})


@app.route("/process-video", methods=["POST"])
def process_video() -> Union[Dict[str, Any], tuple[Dict[str, str], int]]:
    """
    動画処理エンドポイント

    Expected JSON payload:
    {
        "gcsUri": "gs://bucket-name/video.mp4",
        "videoId": "unique-video-id"
    }
    """
    try:
        if not request.is_json:
            return create_error_response("Content-Type must be application/json")

        data = request.get_json()

        # 必須パラメータのチェック
        if not data or "gcsUri" not in data or "videoId" not in data:
            return create_error_response(
                "Missing required parameters. 'gcsUri' and 'videoId' are required."
            )

        # パラメータの取り出し
        gcs_uri = data["gcsUri"]
        video_id = data["videoId"]

        # URIの形式チェック
        if not gcs_uri.startswith("gs://"):
            return create_error_response(
                "Invalid gcsUri format. Must start with 'gs://'"
            )

        # 動画の処理
        result = vectorizer.process_video(gcs_uri, video_id)

        # エラーチェック
        if result.get("status") == "error":
            return create_error_response(
                f"Error processing video: {result.get('error')}", 500
            )

        return jsonify(result)

    except Exception as e:
        return create_error_response(f"Unexpected error: {str(e)}", 500)


@app.route("/search", methods=["POST"])
def search_videos() -> Union[Dict[str, Any], tuple[Dict[str, str], int]]:
    """
    動画検索エンドポイント

    Expected JSON payload:
    {
        "query": "検索クエリ",
        "limit": 5  # オプション、デフォルト5
    }
    """
    try:
        if not request.is_json:
            return create_error_response("Content-Type must be application/json")

        data = request.get_json()

        # 必須パラメータのチェック
        if not data or "query" not in data:
            return create_error_response("Missing required parameter: 'query'")

        # パラメータの取り出し
        query = data["query"]
        limit = data.get("limit", 5)  # デフォルト値は5

        # 検索の実行
        query_embedding = vectorizer.generate_embeddings(query)
        results = vectorizer.search_videos(query_embedding, limit)

        return jsonify({"status": "success", "query": query, "results": results})

    except Exception as e:
        return create_error_response(f"Unexpected error: {str(e)}", 500)


@app.errorhandler(404)
def not_found(e) -> tuple[Dict[str, str], int]:
    """
    404エラーハンドラ
    """
    return create_error_response("Resource not found", 404)


@app.errorhandler(405)
def method_not_allowed(e) -> tuple[Dict[str, str], int]:
    """
    405エラーハンドラ
    """
    return create_error_response("Method not allowed", 405)


if __name__ == "__main__":
    # 環境変数のチェック
    if not all([PROJECT_ID, VECTOR_INDEX_ID]):
        raise ValueError(
            "Missing required environment variables. "
            "Please set GOOGLE_CLOUD_PROJECT_ID and VECTOR_INDEX_ID."
        )

    # サーバーの起動
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
