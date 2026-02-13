"""OSS Client for accessing robot dataset"""

import logging
from typing import List, Optional, Iterator
import oss2
from io import BytesIO
import os


class OSSClient:
    """Wrapper for OSS operations"""

    def __init__(self, config: dict):
        """
        Initialize OSS client

        Args:
            config: OSS configuration with endpoint, bucket, access_key_id, access_key_secret, prefix
        """
        self.logger = logging.getLogger(__name__)
        if config['access_key_id'] == '':
            config['access_key_id'] = os.environ.get('ACCESS_KEY_ID', '')
        if config['access_key_secret'] == '':
            config['access_key_secret'] = os.environ.get('ACCESS_KEY_SECRET', '')

        auth = oss2.Auth(
            config['access_key_id'],
            config['access_key_secret']
        )

        self.bucket = oss2.Bucket(
            auth,
            config['endpoint'],
            config['bucket']
        )

        self.prefix = config.get('prefix', '')
        self.logger.info(f"OSS client initialized: bucket={config['bucket']}, prefix={self.prefix}")

    def list_tasks(self) -> List[str]:
        """
        List all task folders in OSS

        Returns:
            List of task folder names (without prefix)
        """
        self.logger.info("Listing all tasks from OSS...")

        tasks = set()
        prefix = self.prefix if self.prefix.endswith('/') else self.prefix + '/'

        # List all objects with prefix
        for obj in oss2.ObjectIterator(self.bucket, prefix=prefix, delimiter='/'):
            if obj.is_prefix():  # This is a directory
                task_name = obj.key[len(prefix):].rstrip('/')
                if task_name:
                    tasks.add(task_name)

        task_list = sorted(list(tasks))
        self.logger.info(f"Found {len(task_list)} tasks")
        return task_list

    def get_object(self, key: str) -> Optional[bytes]:
        """
        Get object content from OSS

        Args:
            key: Object key (relative to bucket root)

        Returns:
            Object content as bytes, or None if not found
        """
        try:
            result = self.bucket.get_object(key)
            return result.read()
        except oss2.exceptions.NoSuchKey:
            self.logger.warning(f"Object not found: {key}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting object {key}: {e}")
            return None

    def get_object_stream(self, key: str) -> Optional[BytesIO]:
        """
        Get object as stream

        Args:
            key: Object key

        Returns:
            BytesIO stream or None if not found
        """
        content = self.get_object(key)
        if content:
            return BytesIO(content)
        return None

    def object_exists(self, key: str) -> bool:
        """Check if object exists"""
        try:
            self.bucket.head_object(key)
            return True
        except:
            return False

    def list_objects(self, prefix: str) -> List[str]:
        """
        List all objects with given prefix

        Args:
            prefix: Object key prefix

        Returns:
            List of object keys
        """
        objects = []
        for obj in oss2.ObjectIterator(self.bucket, prefix=prefix):
            if not obj.is_prefix():
                objects.append(obj.key)
        return objects

    def get_task_path(self, task_name: str, relative_path: str = '') -> str:
        """
        Get full OSS path for a file in a task

        Args:
            task_name: Task folder name
            relative_path: Relative path within task folder

        Returns:
            Full OSS key path
        """
        prefix = self.prefix if self.prefix.endswith('/') else self.prefix + '/'
        if relative_path:
            return f"{prefix}{task_name}/{relative_path}"
        return f"{prefix}{task_name}"
    
    def upload_folder_non_recursive(self, prefix, local_folder_path):
        for file_name in os.listdir(local_folder_path):
            local_file_path = os.path.join(local_folder_path, file_name)
            if os.path.isfile(local_file_path):
                object_name = os.path.join(prefix, file_name).replace("\\", "/")
                self.bucket.put_object_from_file(object_name, local_file_path)
    
    def upload_file(self, prefix, local_file_path):
        if os.path.isfile(local_file_path):
            self.bucket.put_object_from_file(prefix, local_file_path)

