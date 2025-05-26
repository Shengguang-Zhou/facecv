"""人脸数据库抽象基类"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class AbstractFaceDB(ABC):
    """人脸数据库抽象基类"""
    
    @abstractmethod
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
        pass

    @abstractmethod
    def delete_face_by_id(self, face_id: str) -> bool:
        """
        根据 ID 删除人脸
        
        Args:
            face_id: 人脸 ID
            
        Returns:
            是否删除成功
        """
        pass

    @abstractmethod
    def delete_face_by_name(self, name: str) -> int:
        """
        根据姓名删除所有相关人脸
        
        Args:
            name: 人员姓名
            
        Returns:
            删除的人脸数量
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def query_faces_by_name(self, name: str) -> List[Dict[str, Any]]:
        """
        根据姓名查询人脸
        
        Args:
            name: 人员姓名
            
        Returns:
            人脸信息列表
        """
        pass

    @abstractmethod
    def query_faces_by_embedding(self, embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        根据特征向量查询相似人脸
        
        Args:
            embedding: 人脸特征向量
            top_k: 返回前 k 个最相似的结果
            
        Returns:
            人脸信息列表，包含相似度分数
        """
        pass

    @abstractmethod
    def get_face_by_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 ID 获取人脸信息
        
        Args:
            face_id: 人脸 ID
            
        Returns:
            人脸信息，如果不存在返回 None
        """
        pass

    @abstractmethod
    def get_all_faces(self) -> List[Dict[str, Any]]:
        """
        获取所有人脸信息
        
        Returns:
            所有人脸信息列表
        """
        pass

    @abstractmethod
    def get_all_faces_for_recognition(self) -> List[Dict[str, Any]]:
        """
        获取所有用于识别的人脸（通常排除临时添加的人脸）
        
        Returns:
            人脸信息列表
        """
        pass
    
    @abstractmethod
    def get_face_count(self) -> int:
        """
        获取数据库中的人脸总数
        
        Returns:
            人脸总数
        """
        pass
    
    @abstractmethod
    def clear_database(self) -> bool:
        """
        清空数据库
        
        Returns:
            是否清空成功
        """
        pass