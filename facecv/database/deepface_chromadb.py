"""
DeepFace 专用的 ChromaDB 实现
使用独立的 ChromaDB 实例，避免与 InsightFace 的数据混淆
"""

import os
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import logging

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB 不可用，请安装: pip install chromadb")

from .abstract_facedb import AbstractFaceDB

logger = logging.getLogger(__name__)


class DeepFaceChromaDB(AbstractFaceDB):
    """DeepFace 专用的 ChromaDB 实现"""
    
    def __init__(self, collection_name: str = "deepface_faces", persist_directory: str = "./deepface_chroma_db"):
        """
        初始化 DeepFace ChromaDB
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB 不可用，请安装: pip install chromadb")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # 确保目录存在
        os.makedirs(persist_directory, exist_ok=True)
        
        # 创建 ChromaDB 客户端
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"使用现有 DeepFace ChromaDB 集合: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},  # 使用余弦距离
                embedding_function=None  # 我们自己管理embeddings
            )
            logger.info(f"创建新的 DeepFace ChromaDB 集合: {collection_name}")
    
    def register_face(
        self, 
        name: str, 
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        model_type: str = "deepface",
        model_name: str = "VGG-Face"
    ) -> Optional[str]:
        """
        注册人脸到DeepFace ChromaDB
        
        Args:
            name: 人名
            embedding: 特征向量
            metadata: 元数据
            model_type: 模型类型（固定为 "deepface"）
            model_name: DeepFace使用的具体模型名
            
        Returns:
            face_id: 注册成功返回ID，失败返回None
        """
        try:
            # 生成唯一ID
            face_id = f"deepface_{uuid.uuid4().hex}"
            
            # 准备元数据
            meta = {
                "person_name": name,
                "model_type": "deepface",
                "model_name": model_name,
                "created_at": datetime.now().isoformat(),
                "embedding_size": len(embedding)
            }
            
            if metadata:
                # 添加用户提供的元数据
                for k, v in metadata.items():
                    if k not in meta:  # 避免覆盖核心字段
                        # 确保值是可序列化的
                        if isinstance(v, (str, int, float, bool)):
                            meta[k] = v
                        elif isinstance(v, (list, tuple)):
                            meta[k] = str(v)
                        elif v is None:
                            meta[k] = ""
                        else:
                            meta[k] = str(v)
            
            # 添加到ChromaDB
            self.collection.add(
                ids=[face_id],
                embeddings=[embedding],
                metadatas=[meta]
            )
            
            logger.info(f"DeepFace: Registered face for {name} with ID {face_id} using model {model_name}")
            return face_id
            
        except Exception as e:
            logger.error(f"DeepFace: Error registering face: {e}")
            return None
    
    def search_faces(
        self, 
        embedding: List[float], 
        threshold: float = 0.6,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        搜索相似人脸
        
        Args:
            embedding: 查询特征向量
            threshold: 相似度阈值
            limit: 返回结果数量
            
        Returns:
            匹配结果列表
        """
        try:
            # 查询ChromaDB
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=limit,
                include=["metadatas", "distances", "embeddings"]
            )
            
            matches = []
            if results and results['ids'] and results['ids'][0]:
                for i, face_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    # 转换距离为相似度（余弦距离）
                    similarity = 1.0 - distance
                    
                    if similarity >= threshold:
                        metadata = results['metadatas'][0][i]
                        matches.append({
                            "face_id": face_id,
                            "person_name": metadata.get("person_name", "Unknown"),
                            "similarity": float(similarity),
                            "distance": float(distance),
                            "metadata": metadata,
                            "model_info": {
                                "model_type": metadata.get("model_type", "deepface"),
                                "model_name": metadata.get("model_name", "Unknown")
                            }
                        })
            
            return matches
            
        except Exception as e:
            logger.error(f"DeepFace: Error searching faces: {e}")
            return []
    
    def get_face(self, face_id: str) -> Optional[Dict[str, Any]]:
        """获取单个人脸信息"""
        try:
            result = self.collection.get(
                ids=[face_id],
                include=["metadatas", "embeddings"]
            )
            
            if result and result['ids']:
                metadata = result['metadatas'][0]
                return {
                    "face_id": face_id,
                    "person_name": metadata.get("person_name", "Unknown"),
                    "embedding": result['embeddings'][0],
                    "metadata": metadata
                }
            return None
            
        except Exception as e:
            logger.error(f"DeepFace: Error getting face {face_id}: {e}")
            return None
    
    def update_face(self, face_id: str, **kwargs) -> bool:
        """更新人脸信息"""
        try:
            # 获取现有数据
            existing = self.get_face(face_id)
            if not existing:
                return False
            
            # 更新元数据
            metadata = existing['metadata'].copy()
            
            if 'person_name' in kwargs:
                metadata['person_name'] = kwargs['person_name']
            
            if 'metadata' in kwargs and isinstance(kwargs['metadata'], dict):
                metadata.update(kwargs['metadata'])
            
            metadata['updated_at'] = datetime.now().isoformat()
            
            # 更新到ChromaDB
            self.collection.update(
                ids=[face_id],
                metadatas=[metadata]
            )
            
            logger.info(f"DeepFace: Updated face {face_id}")
            return True
            
        except Exception as e:
            logger.error(f"DeepFace: Error updating face {face_id}: {e}")
            return False
    
    def delete_face(self, face_id: str) -> bool:
        """删除人脸"""
        try:
            self.collection.delete(ids=[face_id])
            logger.info(f"DeepFace: Deleted face {face_id}")
            return True
        except Exception as e:
            logger.error(f"DeepFace: Error deleting face {face_id}: {e}")
            return False
    
    def delete_face_by_name(self, name: str) -> bool:
        """按姓名删除所有相关人脸"""
        try:
            # 查询所有匹配的人脸
            results = self.collection.get(
                where={"person_name": name}
            )
            
            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"DeepFace: Deleted {len(results['ids'])} faces for {name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"DeepFace: Error deleting faces by name {name}: {e}")
            return False
    
    def list_faces(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """列出所有人脸"""
        try:
            # 获取所有数据
            results = self.collection.get(
                limit=limit,
                include=["metadatas", "embeddings"]
            )
            
            faces = []
            if results and results['ids']:
                for i, face_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    faces.append({
                        "face_id": face_id,
                        "person_name": metadata.get("person_name", "Unknown"),
                        "metadata": metadata,
                        "embedding": results['embeddings'][i]
                    })
            
            return faces
            
        except Exception as e:
            logger.error(f"DeepFace: Error listing faces: {e}")
            return []
    
    def get_faces_by_name(self, name: str) -> List[Dict[str, Any]]:
        """按姓名获取所有相关人脸"""
        try:
            results = self.collection.get(
                where={"person_name": name},
                include=["metadatas", "embeddings"]
            )
            
            faces = []
            if results and results['ids']:
                for i, face_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    faces.append({
                        "face_id": face_id,
                        "person_name": name,
                        "metadata": metadata,
                        "embedding": results['embeddings'][i]
                    })
            
            return faces
            
        except Exception as e:
            logger.error(f"DeepFace: Error getting faces by name {name}: {e}")
            return []
    
    def clear_all(self) -> bool:
        """清空所有数据（谨慎使用）"""
        try:
            # 删除并重新创建集合
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None
            )
            logger.warning(f"DeepFace: Cleared all data in collection {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"DeepFace: Error clearing collection: {e}")
            return False
    
    # 实现抽象基类的必需方法
    def add_face(self, name: str, embedding: np.ndarray, metadata: Optional[Dict] = None) -> str:
        """添加人脸到数据库"""
        # 转换 numpy array 到 list
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        face_id = self.register_face(name, embedding, metadata)
        if face_id:
            return face_id
        else:
            raise ValueError("Failed to add face")
    
    def delete_face_by_id(self, face_id: str) -> bool:
        """根据 ID 删除人脸"""
        return self.delete_face(face_id)
    
    def query_faces_by_name(self, name: str) -> List[Dict[str, Any]]:
        """根据姓名查询人脸"""
        return self.get_faces_by_name(name)
    
    def query_faces_by_embedding(self, embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """根据特征向量查询相似人脸"""
        # 转换 numpy array 到 list
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        matches = self.search_faces(embedding, threshold=0.0, limit=top_k)
        
        # 添加必需的字段
        results = []
        for match in matches:
            result = {
                'face_id': match['face_id'],
                'name': match['person_name'],
                'distance': match['distance'],
                'similarity': match['similarity'],
                'metadata': match.get('metadata', {})
            }
            results.append(result)
        
        return results
    
    def get_face_by_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """根据 ID 获取人脸信息"""
        face = self.get_face(face_id)
        if face:
            return {
                'face_id': face['face_id'],
                'name': face['person_name'],
                'embedding': face['embedding'],
                'metadata': face.get('metadata', {})
            }
        return None
    
    def get_all_faces(self) -> List[Dict[str, Any]]:
        """获取所有人脸信息"""
        faces = self.list_faces(limit=10000)
        results = []
        for face in faces:
            results.append({
                'face_id': face['face_id'],
                'name': face['person_name'],
                'embedding': face['embedding'],
                'metadata': face.get('metadata', {})
            })
        return results
    
    def get_all_faces_for_recognition(self) -> List[Dict[str, Any]]:
        """获取所有用于识别的人脸"""
        # DeepFace 中所有人脸都用于识别
        return self.get_all_faces()
    
    def get_face_count(self) -> int:
        """获取数据库中的人脸总数"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting face count: {e}")
            return 0
    
    def clear_database(self) -> bool:
        """清空数据库"""
        return self.clear_all()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            total_faces = self.collection.count()
            
            # 获取所有元数据以统计
            results = self.collection.get(
                limit=total_faces,
                include=["metadatas"]
            )
            
            unique_persons = set()
            model_stats = {}
            
            if results and results['metadatas']:
                for metadata in results['metadatas']:
                    person_name = metadata.get("person_name", "Unknown")
                    unique_persons.add(person_name)
                    
                    model_name = metadata.get("model_name", "Unknown")
                    model_stats[model_name] = model_stats.get(model_name, 0) + 1
            
            return {
                "database_type": "deepface_chromadb",
                "collection_name": self.collection_name,
                "total_faces": total_faces,
                "unique_persons": len(unique_persons),
                "model_distribution": model_stats,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"DeepFace: Error getting stats: {e}")
            return {
                "database_type": "deepface_chromadb",
                "error": str(e)
            }