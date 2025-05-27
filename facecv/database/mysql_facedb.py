"""MySQL 人脸数据库实现 - 修复版"""

import json
import uuid
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from concurrent.futures import ThreadPoolExecutor

try:
    import aiomysql
    import pymysql
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
    MYSQL_AVAILABLE = True
except ImportError as e:
    MYSQL_AVAILABLE = False
    logging.warning(f"MySQL依赖不可用: {e}")

from .abstract_facedb import AbstractFaceDB
from ..config.database import db_config

logging.basicConfig(level=logging.INFO)


class MySQLFaceDB(AbstractFaceDB):
    """MySQL 人脸数据库实现"""
    
    def __init__(self, config=None):
        """
        初始化 MySQL 数据库连接
        
        Args:
            config: 数据库配置对象，默认使用全局配置
        """
        if not MYSQL_AVAILABLE:
            raise ImportError("MySQL依赖不可用，请安装 pymysql 和 aiomysql")
            
        self.config = config or db_config
        self._engine = None
        self._pool = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # 初始化同步引擎
        self._init_sync_engine()
        # 初始化数据库表
        self._init_database()
    
    def _init_sync_engine(self):
        """初始化同步数据库引擎"""
        try:
            self._engine = create_engine(
                self.config.mysql_url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=False
            )
            logging.info("MySQL同步引擎初始化成功")
        except Exception as e:
            logging.error(f"MySQL引擎初始化失败: {e}")
            raise
    
    async def _get_async_pool(self):
        """获取异步连接池"""
        if self._pool is None:
            try:
                self._pool = await aiomysql.create_pool(
                    host=self.config.mysql_host,
                    port=self.config.mysql_port,
                    user=self.config.mysql_user,
                    password=self.config.mysql_password,
                    db=self.config.mysql_database,
                    charset='utf8mb4',
                    minsize=1,
                    maxsize=self.config.pool_size,
                    autocommit=True
                )
                logging.info("MySQL异步连接池创建成功")
            except Exception as e:
                logging.error(f"MySQL异步连接池创建失败: {e}")
                raise
        return self._pool
    
    def _init_database(self):
        """初始化数据库表"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS faces (
            id VARCHAR(36) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            embedding LONGBLOB NOT NULL,
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            is_temporary TINYINT(1) DEFAULT 0,
            INDEX idx_name (name),
            INDEX idx_created_at (created_at),
            INDEX idx_is_temporary (is_temporary)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        
        try:
            with self._engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
                logging.info("数据库表初始化成功")
        except SQLAlchemyError as e:
            logging.error(f"数据库表初始化失败: {e}")
            raise
    
    def _execute_sync(self, func, *args, **kwargs):
        """在线程池中执行同步数据库操作"""
        return self._executor.submit(func, *args, **kwargs).result()
    
    def add_face(self, name: str, embedding: np.ndarray, metadata: Optional[Dict] = None) -> str:
        """添加人脸到数据库"""
        def _add_face_sync():
            face_id = str(uuid.uuid4())
            embedding_bytes = embedding.tobytes()
            metadata_json = json.dumps(metadata) if metadata else None
            
            sql = """
            INSERT INTO faces (id, name, embedding, metadata)
            VALUES (:face_id, :name, :embedding, :metadata)
            """
            
            with self._engine.connect() as conn:
                conn.execute(text(sql), {
                    'face_id': face_id, 
                    'name': name, 
                    'embedding': embedding_bytes, 
                    'metadata': metadata_json
                })
                conn.commit()
                
            logging.info(f"添加人脸成功: {face_id} - {name}")
            return face_id
        
        return self._execute_sync(_add_face_sync)
    
    def delete_face_by_id(self, face_id: str) -> bool:
        """根据 ID 删除人脸"""
        def _delete_sync():
            sql = "DELETE FROM faces WHERE id = :face_id"
            with self._engine.connect() as conn:
                result = conn.execute(text(sql), {'face_id': face_id})
                conn.commit()
                return result.rowcount > 0
        
        return self._execute_sync(_delete_sync)
    
    def delete_face_by_name(self, name: str) -> int:
        """根据姓名删除所有相关人脸"""
        def _delete_sync():
            sql = "DELETE FROM faces WHERE name = :name"
            with self._engine.connect() as conn:
                result = conn.execute(text(sql), {'name': name})
                conn.commit()
                return result.rowcount
        
        return self._execute_sync(_delete_sync)
    
    def update_face(self, face_id: str, new_name: str, metadata: Optional[Dict] = None) -> bool:
        """更新人脸信息"""
        def _update_sync():
            metadata_json = json.dumps(metadata) if metadata else None
            
            if metadata is not None:
                sql = """
                UPDATE faces 
                SET name = :name, metadata = :metadata, updated_at = CURRENT_TIMESTAMP
                WHERE id = :face_id
                """
                params = {'name': new_name, 'metadata': metadata_json, 'face_id': face_id}
            else:
                sql = """
                UPDATE faces 
                SET name = :name, updated_at = CURRENT_TIMESTAMP
                WHERE id = :face_id
                """
                params = {'name': new_name, 'face_id': face_id}
            
            with self._engine.connect() as conn:
                result = conn.execute(text(sql), params)
                conn.commit()
                return result.rowcount > 0
        
        return self._execute_sync(_update_sync)
    
    def query_faces_by_name(self, name: str) -> List[Dict[str, Any]]:
        """根据姓名查询人脸"""
        def _query_sync():
            sql = """
            SELECT id, name, embedding, metadata, created_at, updated_at
            FROM faces WHERE name = :name
            """
            
            results = []
            with self._engine.connect() as conn:
                result = conn.execute(text(sql), {'name': name})
                
                for row in result:
                    face_dict = {
                        'id': row[0],
                        'name': row[1],
                        'embedding': np.frombuffer(row[2], dtype=np.float32),
                        'metadata': json.loads(row[3]) if row[3] else None,
                        'created_at': row[4],
                        'updated_at': row[5]
                    }
                    results.append(face_dict)
                    
            return results
        
        return self._execute_sync(_query_sync)
    
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
    
    def get_face_by_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """根据 ID 获取人脸信息"""
        def _get_sync():
            sql = """
            SELECT id, name, embedding, metadata, created_at, updated_at
            FROM faces WHERE id = :face_id
            """
            
            with self._engine.connect() as conn:
                result = conn.execute(text(sql), {'face_id': face_id})
                row = result.fetchone()
                
                if row:
                    return {
                        'id': row[0],
                        'name': row[1],
                        'embedding': np.frombuffer(row[2], dtype=np.float32),
                        'metadata': json.loads(row[3]) if row[3] else None,
                        'created_at': row[4],
                        'updated_at': row[5]
                    }
                return None
        
        return self._execute_sync(_get_sync)
    
    def get_all_faces(self) -> List[Dict[str, Any]]:
        """获取所有人脸信息"""
        def _get_all_sync():
            sql = """
            SELECT id, name, embedding, metadata, created_at, updated_at
            FROM faces ORDER BY created_at DESC
            """
            
            results = []
            with self._engine.connect() as conn:
                result = conn.execute(text(sql))
                
                for row in result:
                    face_dict = {
                        'id': row[0],
                        'name': row[1],
                        'embedding': np.frombuffer(row[2], dtype=np.float32),
                        'metadata': json.loads(row[3]) if row[3] else None,
                        'created_at': row[4],
                        'updated_at': row[5]
                    }
                    results.append(face_dict)
                    
            return results
        
        return self._execute_sync(_get_all_sync)
    
    def get_all_faces_for_recognition(self) -> List[Dict[str, Any]]:
        """获取所有用于识别的人脸"""
        def _get_recognition_sync():
            sql = """
            SELECT id, name, embedding, metadata, created_at, updated_at
            FROM faces WHERE is_temporary = 0 ORDER BY created_at DESC
            """
            
            results = []
            with self._engine.connect() as conn:
                result = conn.execute(text(sql))
                
                for row in result:
                    face_dict = {
                        'id': row[0],
                        'name': row[1],
                        'embedding': np.frombuffer(row[2], dtype=np.float32),
                        'metadata': json.loads(row[3]) if row[3] else None,
                        'created_at': row[4],
                        'updated_at': row[5]
                    }
                    results.append(face_dict)
                    
            return results
        
        return self._execute_sync(_get_recognition_sync)
    
    def get_face_count(self) -> int:
        """获取数据库中的人脸总数"""
        def _count_sync():
            sql = "SELECT COUNT(*) as count FROM faces"
            with self._engine.connect() as conn:
                result = conn.execute(text(sql))
                return result.fetchone()[0]
        
        return self._execute_sync(_count_sync)
    
    def search_similar_faces(self, embedding: List[float], threshold: float = 0.6, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索相似的人脸（兼容接口）"""
        # 转换为numpy数组
        embedding_np = np.array(embedding, dtype=np.float32)
        
        # 使用现有的query_faces_by_embedding方法
        results = self.query_faces_by_embedding(embedding_np, top_k=limit)
        
        # 过滤低于阈值的结果并格式化输出
        filtered_results = []
        for face in results:
            if face['similarity_score'] >= threshold:
                filtered_results.append({
                    'face_id': face['id'],
                    'person_name': face['name'],
                    'similarity': face['similarity_score'],
                    'distance': 1.0 - face['similarity_score'],
                    'metadata': face.get('metadata', {})
                })
        
        return filtered_results
    
    def search_faces(self, embedding: List[float], threshold: float = 0.6, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索相似的人脸（别名方法）"""
        return self.search_similar_faces(embedding, threshold, limit)
    
    def register_face(self, name: str, embedding: List[float], metadata: Optional[Dict] = None) -> str:
        """注册人脸（兼容接口）"""
        embedding_np = np.array(embedding, dtype=np.float32)
        return self.add_face(name, embedding_np, metadata)
    
    def get_faces_by_name(self, name: str) -> List[Dict[str, Any]]:
        """根据姓名获取人脸（兼容接口）"""
        faces = self.query_faces_by_name(name)
        # 格式化输出
        formatted_faces = []
        for face in faces:
            formatted_faces.append({
                'face_id': face['id'],
                'person_name': face['name'],
                'metadata': face.get('metadata', {}),
                'created_at': face['created_at'].isoformat() if face['created_at'] else None,
                'updated_at': face['updated_at'].isoformat() if face['updated_at'] else None
            })
        return formatted_faces
    
    def list_faces(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """列出人脸（带分页）"""
        def _list_sync():
            sql = """
            SELECT id, name, embedding, metadata, created_at, updated_at
            FROM faces 
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
            """
            
            results = []
            with self._engine.connect() as conn:
                result = conn.execute(text(sql), {'limit': limit, 'offset': offset})
                
                for row in result:
                    face_dict = {
                        'face_id': row[0],
                        'person_name': row[1],
                        'embedding': np.frombuffer(row[2], dtype=np.float32).tolist(),
                        'metadata': json.loads(row[3]) if row[3] else None,
                        'created_at': row[4].isoformat() if row[4] else None,
                        'updated_at': row[5].isoformat() if row[5] else None
                    }
                    results.append(face_dict)
                    
            return results
        
        return self._execute_sync(_list_sync)
    
    def clear_database(self) -> bool:
        """清空数据库"""
        def _clear_sync():
            sql = "DELETE FROM faces"
            with self._engine.connect() as conn:
                conn.execute(text(sql))
                conn.commit()
                return True
        
        return self._execute_sync(_clear_sync)
    
    def close(self):
        """关闭数据库连接"""
        if self._engine:
            self._engine.dispose()
        if self._pool:
            self._pool.close()
        if self._executor:
            self._executor.shutdown(wait=True)
        logging.info("数据库连接已关闭")