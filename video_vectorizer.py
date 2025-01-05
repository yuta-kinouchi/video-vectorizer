from google.cloud import videointelligence_v1
from google.cloud import storage
from google.cloud import aiplatform
from google.cloud.aiplatform import matching_engine
import json
import os
from typing import Dict, List, Any, Optional


class VideoVectorizer:
    def __init__(self, project_id: str, location: str, vector_index_id: str):
        """
        VideoVectorizerの初期化
        Args:
            project_id: Google Cloudプロジェクトのプロジェクトコード
            location: リージョン (例: asia-northeast1)
            vector_index_id: Vertex AI Vector Searchのインデックスコード
        """
        self.project_id = project_id
        self.location = location
        self.vector_index_id = vector_index_id

        # クライアントの初期化
        self.video_client = videointelligence_v1.VideoIntelligenceServiceClient()
        self.storage_client = storage.Client()

        # Vertex AIの初期化
        aiplatform.init(project=project_id, location=location)

        # Vector Searchインデックスの初期化
        self.index = matching_engine.MatchingEngineIndex(
            index_name=vector_index_id, project=project_id, location=location
        )

    def analyze_video(self, gcs_uri: str) -> Dict[str, Any]:
        """
        動画を分析し、ラベル、シーン、音声テキストを抽出
        Args:
            gcs_uri: Cloud Storage上の動画のURI (gs://bucket-name/video.mp4)
        Returns:
            Dict: 分析結果を含む辞書
        """
        features = [
            videointelligence_v1.Feature.LABEL_DETECTION,
            videointelligence_v1.Feature.SHOT_CHANGE_DETECTION,
            videointelligence_v1.Feature.SPEECH_TRANSCRIPTION,
        ]

        config = videointelligence_v1.SpeechTranscriptionConfig(
            language_code="ja-JP",
            enable_automatic_punctuation=True,
        )

        video_context = videointelligence_v1.VideoContext(
            speech_transcription_config=config
        )

        operation = self.video_client.annotate_video(
            request={
                "features": features,
                "input_uri": gcs_uri,
                "video_context": video_context,
            }
        )

        print(f"Processing video analysis for {gcs_uri}...")
        result = operation.result(timeout=600)  # 10分のタイムアウト

        return self._parse_video_analysis(result)

    def _parse_video_analysis(self, result: Any) -> Dict[str, Any]:
        """
        Video Intelligence APIの結果をパース
        Args:
            result: API レスポンス
        Returns:
            Dict: パース済みの分析結果
        """
        video_data = {
            "labels": [],
            "scenes": [],
            "transcript": "",
        }

        # ラベル情報の取得
        for label in result.annotation_results[0].shot_label_annotations:
            video_data["labels"].append(
                {
                    "description": label.entity.description,
                    "confidence": label.frames[0].confidence,
                }
            )

        # シーン情報の取得
        for shot in result.annotation_results[0].shot_annotations:
            start_time = shot.start_time_offset.total_seconds()
            end_time = shot.end_time_offset.total_seconds()
            video_data["scenes"].append(
                {"start_time": start_time, "end_time": end_time}
            )

        # 音声テキストの取得
        for transcript in result.annotation_results[0].speech_transcriptions:
            for alternative in transcript.alternatives:
                video_data["transcript"] += alternative.transcript + " "

        return video_data

    def generate_embeddings(self, text: str) -> List[float]:
        """
        テキストのベクトル埋め込みを生成
        Args:
            text: 入力テキスト
        Returns:
            List[float]: ベクトル埋め込み
        """
        model = aiplatform.TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
        embeddings = model.get_embeddings([text])
        return embeddings[0].values

    def store_vectors(self, vector_data: List[float], metadata: Dict[str, Any]) -> None:
        """
        Vertex AI Vector Searchにベクトルデータを保存
        Args:
            vector_data: ベクトルデータ
            metadata: 関連するメタデータ
        """
        self.index.upsert_embeddings(
            embeddings=[vector_data], ids=[metadata["video_id"]], parameters=metadata
        )

    def search_videos(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        ベクトル検索を実行
        Args:
            query_embedding: 検索クエリのベクトル
            limit: 返す結果の最大数
        Returns:
            List[Dict]: 検索結果のリスト
        """
        results = self.index.find_neighbors(
            query_embeddings=[query_embedding], num_neighbors=limit
        )

        return [
            {
                "video_id": result.id,
                "score": result.distance,
                "metadata": result.parameters,
            }
            for result in results[0]
        ]

    def process_video(self, gcs_uri: str, video_id: str) -> Dict[str, Any]:
        """
        動画処理のメインフロー
        Args:
            gcs_uri: Cloud Storage上の動画のURI
            video_id: 動画の一意識別子
        Returns:
            Dict: 処理結果
        """
        try:
            # 1. 動画分析
            video_data = self.analyze_video(gcs_uri)

            # 2. 検索用テキストの作成
            search_text = f"{' '.join([label['description'] for label in video_data['labels']])} {video_data['transcript']}"

            # 3. ベクトル埋め込みの生成
            embeddings = self.generate_embeddings(search_text)

            # 4. メタデータの作成
            metadata = {
                "video_id": video_id,
                "gcs_uri": gcs_uri,
                "labels": video_data["labels"],
                "scenes": video_data["scenes"],
                "transcript": video_data["transcript"],
            }

            # 5. ベクトルデータの保存
            self.store_vectors(embeddings, metadata)

            return {"status": "success", "video_id": video_id, "metadata": metadata}

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return {"status": "error", "video_id": video_id, "error": str(e)}
