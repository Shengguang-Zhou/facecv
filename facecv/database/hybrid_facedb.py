"""Hybrid Face Database Implementation

This module implements a hybrid face database that combines MySQL for metadata storage
and ChromaDB for embedding vector storage and similarity search.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from facecv.database.abstract_facedb import AbstractFaceDB
from facecv.database.mysql_facedb import MySQLFaceDB
from facecv.database.chroma_facedb import ChromaFaceDB

logger = logging.getLogger(__name__)

class HybridFaceDB(AbstractFaceDB):
    """
    Hybrid Face Database implementation that combines MySQL and ChromaDB.
    
    MySQL is used for storing face metadata and ChromaDB is used for storing
    face embeddings and performing similarity search.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the hybrid database with both MySQL and ChromaDB backends.
        
        Args:
            **kwargs: Configuration parameters for both MySQL and ChromaDB
        """
        super().__init__()
        
        self.relational_db = MySQLFaceDB(**kwargs)
        
        # Extract only the parameters needed for ChromaFaceDB
        chroma_params = {}
        if 'persist_directory' in kwargs:
            chroma_params['persist_directory'] = kwargs['persist_directory']
        else:
            chroma_params['persist_directory'] = './chroma_db'
            
        if 'collection_name' in kwargs:
            chroma_params['collection_name'] = kwargs.get('collection_name', 'face_embeddings')
            
        logger.info(f"Initializing ChromaFaceDB with params: {chroma_params}")
        self.embedding_collection = ChromaFaceDB(**chroma_params)
        
        logger.info("Hybrid Face Database initialized with MySQL and ChromaDB backends")
    
    def add_face(self, name: str, embedding: np.ndarray, metadata: Optional[Dict] = None) -> str:
        """
        添加人脸到数据库
        
        Args:
            name: 人员姓名
            embedding: 人脸特征向量
            metadata: 额外的元数据
            
        Returns:
            人脸 ID
        """
        if metadata is None:
            metadata = {}
            
        # Generate a UUID for both MySQL and ChromaDB
        face_id = str(uuid.uuid4())
        
        mysql_result = self.relational_db.add_face(name, embedding, metadata)
        if mysql_result:
            face_id = mysql_result
            logger.info(f"Using MySQL-generated face ID: {face_id}")
        
        chroma_metadata = {
            'name': name,
            'created_at': datetime.now().isoformat(),
            'mysql_id': face_id  # Store the MySQL ID in ChromaDB metadata
        }
        
        for key, value in metadata.items():
            if key not in chroma_metadata:
                if isinstance(value, (str, int, float, bool)):
                    chroma_metadata[key] = str(value)
                else:
                    chroma_metadata[key] = str(value)
        
        embedding_list = []
        if isinstance(embedding, np.ndarray):
            if embedding.shape[0] == 1 and len(embedding.shape) > 1:
                embedding = embedding.flatten()
            embedding_list = embedding.tolist()
        elif isinstance(embedding, list):
            embedding_list = embedding
            if len(embedding_list) > 0 and isinstance(embedding_list[0], list):
                embedding_list = embedding_list[0]
        else:
            logger.warning(f"Unexpected embedding type: {type(embedding)}")
            try:
                embedding_list = list(embedding)
            except Exception as e:
                logger.error(f"Failed to convert embedding to list: {e}")
                embedding_list = []
        
        try:
            embedding_list = [float(x) for x in embedding_list]
            
            # Check if we need to pad or truncate the embedding
            if len(embedding_list) != 512:
                logger.warning(f"Embedding dimension mismatch: {len(embedding_list)} != 512")
                
                if len(embedding_list) < 512:
                    padding = [0.0] * (512 - len(embedding_list))
                    embedding_list.extend(padding)
                    logger.info(f"Padded embedding to length 512")
                else:
                    embedding_list = embedding_list[:512]
                    logger.info(f"Truncated embedding to length 512")
            
            self.embedding_collection.collection.add(
                ids=[face_id],
                embeddings=[embedding_list],
                metadatas=[chroma_metadata]
            )
            logger.info(f"Added face to ChromaDB with ID: {face_id}")
        except Exception as e:
            logger.error(f"Error adding face to ChromaDB: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info(f"Added face {face_id} to hybrid database")
        return face_id
    
    def delete_face_by_id(self, face_id: str) -> bool:
        """
        根据 ID 删除人脸
        
        Args:
            face_id: 人脸 ID
            
        Returns:
            是否删除成功
        """
        mysql_result = self.relational_db.delete_face_by_id(face_id)
        
        chroma_result = self.embedding_collection.delete_face_by_id(face_id)
        
        success = mysql_result and chroma_result
        if success:
            logger.info(f"Deleted face {face_id} from hybrid database")
        else:
            logger.warning(f"Partial deletion of face {face_id} from hybrid database")
            
        return success
    
    def delete_face_by_name(self, name: str) -> int:
        """
        根据姓名删除所有相关人脸
        
        Args:
            name: 人员姓名
            
        Returns:
            删除的人脸数量
        """
        faces = self.query_faces_by_name(name)
        face_ids = [face['id'] for face in faces]
        
        mysql_count = self.relational_db.delete_face_by_name(name)
        
        chroma_count = 0
        for face_id in face_ids:
            if self.embedding_collection.delete_face_by_id(face_id):
                chroma_count += 1
        
        if mysql_count != chroma_count:
            logger.warning(f"Inconsistent deletion counts: MySQL={mysql_count}, ChromaDB={chroma_count}")
            
        logger.info(f"Deleted {mysql_count} faces for name '{name}' from hybrid database")
        return mysql_count
    
    def update_face(self, face_id: str, new_name: str, metadata: Optional[Dict] = None) -> bool:
        """
        更新人脸信息
        
        Args:
            face_id: 人脸 ID
            new_name: 新的姓名
            metadata: 新的元数据
            
        Returns:
            是否更新成功
        """
        mysql_result = self.relational_db.update_face(face_id, new_name, metadata)
        
        chroma_metadata = {}
        if new_name is not None:
            chroma_metadata['name'] = new_name
        
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    chroma_metadata[key] = str(value)
        
        if new_name is not None or metadata is not None:
            chroma_result = self.embedding_collection.update_face(
                face_id, new_name, metadata=chroma_metadata
            )
        else:
            chroma_result = True
        
        success = mysql_result and chroma_result
        if success:
            logger.info(f"Updated face {face_id} in hybrid database")
        else:
            logger.warning(f"Partial update of face {face_id} in hybrid database")
            
        return success
    
    def get_face_by_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 ID 获取人脸信息
        
        Args:
            face_id: 人脸 ID
            
        Returns:
            人脸信息，如果不存在返回 None
        """
        return self.relational_db.get_face_by_id(face_id)
    
    def query_faces_by_name(self, name: str) -> List[Dict[str, Any]]:
        """
        根据姓名查询人脸
        
        Args:
            name: 人员姓名
            
        Returns:
            人脸信息列表
        """
        return self.relational_db.query_faces_by_name(name)
    
    def query_faces_by_embedding(self, embedding: np.ndarray, top_k: int = 10, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        根据特征向量查询相似人脸
        
        Args:
            embedding: 人脸特征向量
            top_k: 返回前 k 个最相似的结果
            threshold: 相似度阈值，只返回相似度大于此值的结果
            
        Returns:
            人脸信息列表，包含相似度分数
        """
        embedding_list = []
        if isinstance(embedding, np.ndarray):
            if embedding.shape[0] == 1 and len(embedding.shape) > 1:
                embedding = embedding.flatten()
            
            embedding_list = embedding.tolist()
            logger.info(f"Embedding is numpy array with shape: {embedding.shape}")
        elif isinstance(embedding, list):
            embedding_list = embedding
            logger.info(f"Embedding is list with length: {len(embedding)}")
            
            if len(embedding_list) > 0 and isinstance(embedding_list[0], list):
                embedding_list = embedding_list[0]
                logger.info(f"Flattened nested list to length: {len(embedding_list)}")
        else:
            logger.warning(f"Unexpected embedding type: {type(embedding)}")
            try:
                embedding_list = list(embedding)
                logger.info(f"Converted embedding to list with length: {len(embedding_list)}")
            except Exception as e:
                logger.error(f"Failed to convert embedding to list: {e}")
                return []
        
        try:
            logger.info(f"ChromaDB collection count: {self.embedding_collection.get_face_count()}")
            
            if isinstance(embedding_list, list) and len(embedding_list) > 0:
                embedding_list = [float(x) for x in embedding_list]
                
                logger.info(f"Final embedding length: {len(embedding_list)}")
                logger.info(f"Embedding sample (first 5 values): {embedding_list[:5]}")
                
                # Check if we need to pad or truncate the embedding
                if len(embedding_list) != 512:
                    logger.warning(f"Embedding dimension mismatch: {len(embedding_list)} != 512")
                    
                    if len(embedding_list) < 512:
                        padding = [0.0] * (512 - len(embedding_list))
                        embedding_list.extend(padding)
                        logger.info(f"Padded embedding to length 512")
                    else:
                        embedding_list = embedding_list[:512]
                        logger.info(f"Truncated embedding to length 512")
                
                actual_threshold = max(0.01, threshold)
                similar_faces = self.embedding_collection.query_faces_by_embedding(embedding_list, top_k, actual_threshold)
                logger.info(f"Found {len(similar_faces)} similar faces with threshold={actual_threshold}")
            else:
                logger.error(f"Invalid embedding format: {type(embedding_list)}")
                return []
        except Exception as e:
            logger.error(f"Error querying faces by embedding: {e}")
            return []
        
        enriched_results = []
        for face in similar_faces:
            face_id = face.get('id')
            mysql_id = face.get('metadata', {}).get('mysql_id', face_id)
            
            mysql_face = None
            if face_id:
                mysql_face = self.relational_db.get_face_by_id(face_id)
                
            if not mysql_face and mysql_id and mysql_id != face_id:
                mysql_face = self.relational_db.get_face_by_id(mysql_id)
                
            if mysql_face:
                mysql_face['similarity'] = face.get('similarity', 0.0)
                mysql_face['id'] = mysql_id  # Ensure we use the MySQL ID
                enriched_results.append(mysql_face)
            else:
                face_copy = face.copy()
                if mysql_id:
                    face_copy['id'] = mysql_id  # Use the MySQL ID from metadata
                enriched_results.append(face_copy)
        
        enriched_results.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
        
        if threshold > 0:
            enriched_results = [face for face in enriched_results if face.get('similarity', 0.0) >= threshold]
            
        return enriched_results
    
    def get_all_faces(self) -> List[Dict[str, Any]]:
        """
        获取所有人脸信息
        
        Returns:
            所有人脸信息列表
        """
        return self.relational_db.get_all_faces()
    
    def get_all_faces_for_recognition(self) -> List[Dict[str, Any]]:
        """
        获取所有用于识别的人脸（通常排除临时添加的人脸）
        
        Returns:
            人脸信息列表
        """
        return self.get_all_faces()
    
    def get_face_count(self) -> int:
        """
        获取数据库中的人脸总数
        
        Returns:
            人脸总数
        """
        return self.relational_db.get_face_count()
    
    def clear_database(self) -> bool:
        """
        清空数据库
        
        Returns:
            是否清空成功
        """
        mysql_result = self.relational_db.clear_database()
        chroma_result = self.embedding_collection.clear_database()
        
        return mysql_result and chroma_result
    
    def close(self) -> None:
        """Close database connections."""
        self.relational_db.close()
        try:
            if hasattr(self.embedding_collection, 'close'):
                self.embedding_collection.close()
        except Exception as e:
            logger.warning(f"Error closing ChromaDB connection: {e}")
