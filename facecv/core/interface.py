"""统一的人脸识别接口"""

from typing import List, Dict, Optional, Union
import numpy as np
from PIL import Image
import logging

try:
    from facecv.models.insightface import InsightFaceRecognizer
except ImportError:
    # 如果 InsightFace 未安装，使用模拟版本
    logging.warning("Using mock InsightFaceRecognizer. Install insightface for real functionality.")
from facecv.database.sqlite_facedb import SQLiteFaceDB
from facecv.database.abstract_facedb import AbstractFaceDB
from facecv.schemas.face import RecognitionResult, VerificationResult
from facecv.config import get_settings

logging.basicConfig(level=logging.INFO)


class FaceRecognizer:
    """统一的人脸识别器接口"""
    
    def __init__(self, 
                 model: str = "insightface",
                 db_type: str = None,
                 db_connection: str = None,
                 **kwargs):
        """
        初始化人脸识别器
        
        Args:
            model: 模型类型，支持 "insightface" 或 "deepface"
            db_type: 数据库类型
            db_connection: 数据库连接字符串
            **kwargs: 其他模型特定参数
        """
        self.model_type = model
        settings = get_settings()
        
        # 初始化数据库
        db_type = db_type or settings.db_type
        db_connection = db_connection or settings.db_connection_string
        
        if db_type == "sqlite":
            self.face_db = SQLiteFaceDB(db_connection)
        else:
            # TODO: 支持其他数据库类型
            raise ValueError(f"Unsupported database type: {db_type}")
            
        # 初始化模型
        if model == "insightface":
            self.model = InsightFaceRecognizer(
                face_db=self.face_db,
                similarity_threshold=kwargs.get('similarity_threshold', 0.6),
                use_cuda=kwargs.get('use_cuda', settings.model_device == "cuda"),
                support_multiple=kwargs.get('support_multiple', True)
            )
        else:
            # TODO: 支持 DeepFace
            raise ValueError(f"Unsupported model type: {model}")
            
        logging.info(f"Initialized FaceRecognizer with {model} model and {db_type} database")
        
    def register(self, 
                 image: Union[str, np.ndarray, Image.Image],
                 name: str,
                 metadata: Dict = None) -> List[str]:
        """
        注册人脸
        
        Args:
            image: 图片路径、numpy数组或PIL图像
            name: 人员姓名
            metadata: 额外的元数据
            
        Returns:
            注册的人脸ID列表
        """
        # 转换 PIL 图像为 numpy 数组
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # 调用模型的 add_face 方法
        face_ids = self.model.add_face(image, name)
        
        # 如果有元数据，更新到数据库
        if metadata and face_ids:
            for face_id in face_ids:
                self.face_db.update_face(face_id, name, metadata)
                
        return face_ids
    
    def recognize(self,
                  image: Union[str, np.ndarray, Image.Image],
                  threshold: float = None) -> List[RecognitionResult]:
        """
        识别图片中的人脸
        
        Args:
            image: 图片路径、numpy数组或PIL图像
            threshold: 相似度阈值
            
        Returns:
            识别结果列表
        """
        # 转换 PIL 图像为 numpy 数组
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        return self.model.recognize(image, threshold)
    
    def verify(self,
               image1: Union[str, np.ndarray, Image.Image],
               image2: Union[str, np.ndarray, Image.Image],
               threshold: float = 0.6) -> VerificationResult:
        """
        验证两张人脸是否为同一人
        
        Args:
            image1: 第一张图片
            image2: 第二张图片
            threshold: 判定阈值
            
        Returns:
            验证结果
        """
        # 转换 PIL 图像为 numpy 数组
        if isinstance(image1, Image.Image):
            image1 = np.array(image1)
        if isinstance(image2, Image.Image):
            image2 = np.array(image2)
            
        similarity = self.model.verify(image1, image2)
        
        return VerificationResult(
            is_same_person=similarity >= threshold,
            similarity_score=similarity,
            threshold=threshold
        )
    
    def delete(self, face_id: str = None, name: str = None) -> bool:
        """
        删除已注册的人脸
        
        Args:
            face_id: 人脸ID
            name: 人员姓名
            
        Returns:
            是否删除成功
        """
        if face_id:
            return self.face_db.delete_face_by_id(face_id)
        elif name:
            count = self.face_db.delete_face_by_name(name)
            return count > 0
        else:
            raise ValueError("Must provide either face_id or name")
            
    def list_faces(self, name: str = None) -> List[Dict]:
        """
        列出人脸
        
        Args:
            name: 如果指定，只返回该姓名的人脸
            
        Returns:
            人脸信息列表
        """
        if name:
            return self.face_db.query_faces_by_name(name)
        else:
            return self.face_db.get_all_faces()
            
    def get_face_count(self) -> int:
        """获取人脸总数"""
        return self.face_db.get_face_count()