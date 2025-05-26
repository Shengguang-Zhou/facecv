"""SQLite 人脸数据库实现"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

from .abstract_facedb import AbstractFaceDB

logging.basicConfig(level=logging.INFO)


class SQLiteFaceDB(AbstractFaceDB):
    """SQLite 人脸数据库实现"""
    
    def __init__(self, db_path: str = "facecv.db"):
        """
        初始化 SQLite 数据库
        
        Args:
            db_path: 数据库文件路径
        """
        import os
        # 确保数据库目录存在
        if db_path.startswith('sqlite:///'):
            # 处理 SQLAlchemy 风格的连接字符串
            self.db_path = db_path.replace('sqlite:///', '')
        else:
            self.db_path = db_path
            
        # 创建目录（如果需要）
        if self.db_path != ":memory:":
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
        
        # 对于内存数据库，保持连接
        self._connection = None
        if self.db_path == ":memory:":
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            
        self._init_database()
        
    def _get_connection(self):
        """获取数据库连接"""
        if self._connection:
            return self._connection
        return sqlite3.connect(self.db_path)
    
    def _init_database(self):
        """初始化数据库表"""
        if self._connection:
            conn = self._connection
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP,
                    is_temporary INTEGER DEFAULT 0
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON faces(name)")
            conn.commit()
        else:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS faces (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP,
                        is_temporary INTEGER DEFAULT 0
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON faces(name)")
                conn.commit()
            
    def add_face(self, name: str, embedding: np.ndarray, metadata: Optional[Dict] = None) -> str:
        """添加人脸到数据库"""
        face_id = str(uuid.uuid4())
        embedding_bytes = embedding.tobytes()
        metadata_json = json.dumps(metadata) if metadata else None
        
        if self._connection:
            conn = self._connection
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO faces (id, name, embedding, metadata)
                VALUES (?, ?, ?, ?)
            """, (face_id, name, embedding_bytes, metadata_json))
            conn.commit()
        else:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO faces (id, name, embedding, metadata)
                    VALUES (?, ?, ?, ?)
                """, (face_id, name, embedding_bytes, metadata_json))
                conn.commit()
            
        logging.info(f"Added face {face_id} for {name}")
        return face_id
        
    def delete_face_by_id(self, face_id: str) -> bool:
        """根据 ID 删除人脸"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM faces WHERE id = ?", (face_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_face(self, face_id: str) -> bool:
        """Alias for delete_face_by_id for API compatibility"""
        return self.delete_face_by_id(face_id)
            
    def delete_face_by_name(self, name: str) -> int:
        """根据姓名删除所有相关人脸"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM faces WHERE name = ?", (name,))
            conn.commit()
            return cursor.rowcount
            
    def update_face(self, face_id: str, new_name: str, metadata: Optional[Dict] = None) -> bool:
        """更新人脸信息"""
        metadata_json = json.dumps(metadata) if metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if metadata is not None:
                cursor.execute("""
                    UPDATE faces 
                    SET name = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (new_name, metadata_json, face_id))
            else:
                cursor.execute("""
                    UPDATE faces 
                    SET name = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (new_name, face_id))
            conn.commit()
            return cursor.rowcount > 0
            
    def query_faces_by_name(self, name: str) -> List[Dict[str, Any]]:
        """根据姓名查询人脸"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, embedding, metadata, created_at, updated_at
                FROM faces WHERE name = ?
            """, (name,))
            
            results = []
            for row in cursor.fetchall():
                face_dict = dict(row)
                # 转换 embedding
                face_dict['embedding'] = np.frombuffer(
                    face_dict['embedding'], dtype=np.float32
                )
                # 解析 metadata
                if face_dict['metadata']:
                    face_dict['metadata'] = json.loads(face_dict['metadata'])
                results.append(face_dict)
                
            return results
            
    def query_faces_by_embedding(self, embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """根据特征向量查询相似人脸"""
        all_faces = self.get_all_faces()
        
        if not all_faces:
            return []
            
        # 计算相似度
        db_embeddings = np.array([face['embedding'] for face in all_faces])
        similarities = cosine_similarity(
            embedding.reshape(1, -1),
            db_embeddings
        )[0]
        
        # 排序并返回 top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            face = all_faces[idx].copy()
            face['similarity_score'] = float(similarities[idx])
            results.append(face)
            
        return results
        
    def search_similar_faces(self, embedding: List[float], threshold: float = 0.4, limit: int = 10) -> List[tuple]:
        """
        Search for similar faces in database
        
        Args:
            embedding: Query embedding as list of floats
            threshold: Similarity threshold (0-1)
            limit: Maximum number of results
            
        Returns:
            List of tuples: (face_dict, similarity_score)
        """
        # Convert list to numpy array
        query_embedding = np.array(embedding, dtype=np.float32)
        
        # Get similar faces using existing method
        similar_faces = self.query_faces_by_embedding(query_embedding, top_k=limit)
        
        # Filter by threshold and format as expected by ONNX recognizer
        results = []
        for face in similar_faces:
            if face['similarity_score'] >= threshold:
                # Format as tuple: (face_dict, similarity_score)
                face_dict = {
                    'id': face['id'],
                    'name': face['name'],
                    'metadata': face.get('metadata', {})
                }
                results.append((face_dict, face['similarity_score']))
        
        return results
        
    def get_face_by_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """根据 ID 获取人脸信息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, embedding, metadata, created_at, updated_at
                FROM faces WHERE id = ?
            """, (face_id,))
            
            row = cursor.fetchone()
            if row:
                face_dict = dict(row)
                face_dict['embedding'] = np.frombuffer(
                    face_dict['embedding'], dtype=np.float32
                )
                if face_dict['metadata']:
                    face_dict['metadata'] = json.loads(face_dict['metadata'])
                return face_dict
            return None
            
    def get_all_faces(self) -> List[Dict[str, Any]]:
        """获取所有人脸信息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, embedding, metadata, created_at, updated_at
                FROM faces
            """)
            
            results = []
            for row in cursor.fetchall():
                face_dict = dict(row)
                face_dict['embedding'] = np.frombuffer(
                    face_dict['embedding'], dtype=np.float32
                )
                if face_dict['metadata']:
                    face_dict['metadata'] = json.loads(face_dict['metadata'])
                results.append(face_dict)
                
            return results
            
    def get_all_faces_for_recognition(self) -> List[Dict[str, Any]]:
        """获取所有用于识别的人脸"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, embedding, metadata, created_at, updated_at
                FROM faces WHERE is_temporary = 0
            """)
            
            results = []
            for row in cursor.fetchall():
                face_dict = dict(row)
                face_dict['embedding'] = np.frombuffer(
                    face_dict['embedding'], dtype=np.float32
                )
                if face_dict['metadata']:
                    face_dict['metadata'] = json.loads(face_dict['metadata'])
                results.append(face_dict)
                
            return results
            
    def get_face_count(self) -> int:
        """获取数据库中的人脸总数"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM faces")
            return cursor.fetchone()[0]
            
    def clear_database(self) -> bool:
        """清空数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM faces")
            conn.commit()
            return True