from minio import Minio
import os
import logging
import io

from datetime import timedelta

logger = logging.getLogger("MinioStorage")

class MinioUploader:
    def __init__(self):
        self.endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = os.getenv("MINIO_ROOT_USER", "minioadmin")
        self.secret_key = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
        self.bucket = os.getenv("MINIO_BUCKET", "diarization")
        self.secure = os.getenv("MINIO_SECURE", "False").lower() == "true"
        
        try:
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure
            )
            self._ensure_bucket()
        except Exception as e:
            logger.error(f"Failed to initialize MinIO: {e}")
            self.client = None

    def _ensure_bucket(self):
        if not self.client: return
        try:
            if not self.client.bucket_exists(self.bucket):
                logger.info(f"Creating bucket: {self.bucket}")
                self.client.make_bucket(self.bucket)
        except Exception as e:
            logger.error(f"Bucket check failed: {e}")

    def upload_file(self, object_name, file_path, content_type="application/octet-stream"):
        if not self.client: 
            logger.warning("MinIO client not available. Skipping upload.")
            return False
            
        import time
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                result = self.client.fput_object(
                    self.bucket, 
                    object_name, 
                    file_path,
                    content_type=content_type
                )
                logger.info(f"Uploaded {object_name} to MinIO. Etag: {result.etag}")
                return True
            except Exception as e:
                wait_time = (2 ** attempt) * 0.5 # 0.5, 1.0, 2.0s
                logger.warning(f"Upload failed for {object_name} (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        logger.error(f"Upload failed for {object_name} after {max_retries} attempts.")
        return False
            
    def upload_bytes(self, object_name, data_bytes, content_type="application/octet-stream"):
        if not self.client:
            return False
            
        try:
            data_stream = io.BytesIO(data_bytes)
            length = len(data_bytes)
            self.client.put_object(
                self.bucket,
                object_name,
                data_stream,
                length,
                content_type=content_type
            )
            logger.info(f"Uploaded {object_name} ({length} bytes)")
            return True
        except Exception as e:
            logger.error(f"Bytes upload failed: {e}")
            return False

    def get_presigned_url(self, object_name, expires_hours=1):
        if not self.client: return None
        try:
            url = self.client.presigned_get_object(
                self.bucket,
                object_name,
                expires=timedelta(hours=expires_hours)
            )
            return url
        except Exception as e:
            logger.error(f"Presigned URL generation failed: {e}")
            return None

    def download_file(self, object_name, file_path):
        if not self.client: return False
        try:
            self.client.fget_object(self.bucket, object_name, file_path)
            logger.info(f"Downloaded {object_name} to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Download failed for {object_name}: {e}")
            return False
