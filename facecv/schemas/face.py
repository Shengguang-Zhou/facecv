"""人脸识别相关数据模型"""

from typing import List, Union, Optional, Dict, Any
from datetime import datetime
import numpy as np
from pydantic import BaseModel, Field


class FaceCreate(BaseModel):
    """创建人脸请求"""
    name: str = Field(..., example="张三", description="人员姓名")
    metadata: Optional[Dict] = Field(None, description="额外的元数据")


class FaceUpdate(BaseModel):
    """更新人脸请求"""
    new_name: str = Field(..., example="李四", description="新的姓名")
    metadata: Optional[Dict] = Field(None, description="更新的元数据")


class FaceInfo(BaseModel):
    """人脸信息"""
    id: str = Field(..., description="人脸 ID")
    name: str = Field(..., description="人员姓名")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")
    metadata: Optional[Dict] = Field(None, description="元数据")
    
    class Config:
        from_attributes = True


class SimilarFace(BaseModel):
    """相似人脸结果"""
    face_id: str = Field(..., description="人脸 ID")
    name: str = Field(..., description="人员姓名")
    similarity_score: float = Field(..., ge=0, le=1, description="相似度分数")
    metadata: Optional[Dict] = Field(None, description="元数据")


class RecognitionResult(BaseModel):
    """人脸识别结果"""
    name: str = Field(..., description="识别到的姓名")
    confidence: float = Field(..., ge=0, le=1, description="相似度分数")
    bbox: List[int] = Field(..., description="人脸边界框 [x1, y1, x2, y2]")
    face_id: Optional[str] = Field(None, description="人脸 ID")
    detection_score: Optional[float] = Field(None, ge=0, le=1, description="检测置信度")
    metadata: Optional[Dict[str, Any]] = Field(None, description="附加元数据（年龄、性别等）")
    embedding: Optional[List[float]] = Field(None, description="人脸特征向量")
    
    
class VerificationResult(BaseModel):
    """人脸验证结果"""
    is_same_person: bool = Field(..., description="是否为同一人")
    confidence: float = Field(..., ge=0, le=1, description="相似度分数")
    distance: float = Field(..., ge=0, le=2, description="距离值（越小越相似）")
    threshold: float = Field(..., description="使用的阈值")
    message: Optional[str] = Field(None, description="附加信息")
    face1_bbox: Optional[List[int]] = Field(None, description="第一张图片中的人脸边界框")
    face2_bbox: Optional[List[int]] = Field(None, description="第二张图片中的人脸边界框")
    face1_quality: Optional[float] = Field(None, description="第一张图片的人脸质量")
    face2_quality: Optional[float] = Field(None, description="第二张图片的人脸质量")


class BatchRecognitionResult(BaseModel):
    """批量识别结果"""
    image_index: int = Field(..., description="图片索引")
    faces: List[RecognitionResult] = Field(..., description="识别到的人脸列表")
    processing_time: float = Field(..., description="处理时间（秒）")


class AttendanceRecord(BaseModel):
    """考勤记录"""
    face_id: str = Field(..., description="人脸 ID")
    name: str = Field(..., description="人员姓名")
    check_time: datetime = Field(..., description="打卡时间")
    check_type: str = Field(..., description="打卡类型", example="check_in")
    location: Optional[str] = Field(None, description="打卡地点")
    device_id: Optional[str] = Field(None, description="设备 ID")


class StrangerAlert(BaseModel):
    """陌生人警报"""
    alert_id: str = Field(..., description="警报 ID")
    detected_time: datetime = Field(..., description="检测时间")
    bbox: List[int] = Field(..., description="人脸边界框")
    image_url: Optional[str] = Field(None, description="抓拍图片 URL")
    camera_id: Optional[str] = Field(None, description="摄像头 ID")
    location: Optional[str] = Field(None, description="位置信息")


# ==================== DeepFace API 数据模型 ====================

class FaceRegisterRequest(BaseModel):
    """人脸注册请求"""
    name: str = Field(..., description="人员姓名")
    metadata: Optional[Dict[str, Any]] = Field(None, description="附加元数据")


class FaceRegisterResponse(BaseModel):
    """人脸注册响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    person_name: str = Field(..., description="注册的人员姓名")
    face_id: Optional[str] = Field(None, description="生成的人脸ID")


class FaceRecognitionRequest(BaseModel):
    """人脸识别请求"""
    threshold: Optional[float] = Field(0.6, description="识别阈值")
    return_all_candidates: bool = Field(False, description="是否返回所有候选结果")


class FaceRecognitionResponse(BaseModel):
    """人脸识别响应"""
    faces: List[Dict[str, Any]] = Field(..., description="识别到的人脸列表")
    total_faces: int = Field(..., description="检测到的人脸总数")
    processing_time: float = Field(..., description="处理时间（秒）")


class FaceVerificationRequest(BaseModel):
    """人脸验证请求"""
    threshold: Optional[float] = Field(0.6, description="验证阈值")
    model_name: str = Field("ArcFace", description="使用的模型")
    anti_spoofing: bool = Field(False, description="是否启用反欺骗检测")


class FaceVerificationResponse(BaseModel):
    """人脸验证响应"""
    verified: bool = Field(..., description="是否验证通过")
    confidence: float = Field(..., description="置信度")
    distance: float = Field(..., description="距离值")
    threshold: float = Field(..., description="使用的阈值")
    model: str = Field(..., description="使用的模型")


class FaceAnalysisRequest(BaseModel):
    """人脸分析请求"""
    actions: List[str] = Field(["emotion", "age", "gender", "race"], description="分析维度")
    detector_backend: str = Field("mtcnn", description="人脸检测器")


class FaceAnalysisResponse(BaseModel):
    """人脸分析响应"""
    faces: List[Dict[str, Any]] = Field(..., description="分析结果列表")
    total_faces: int = Field(..., description="检测到的人脸总数")
    
    class Config:
        """配置类"""
        json_encoders = {
            np.integer: lambda x: int(x),
            np.floating: lambda x: float(x),
            np.ndarray: lambda x: x.tolist(),
            np.bool_: lambda x: bool(x)
        }
        
    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        """自定义验证方法，确保numpy类型被正确转换"""
        if isinstance(obj, dict):
            obj = cls._convert_numpy_types(obj)
        return super().model_validate(obj, *args, **kwargs)
    
    @staticmethod
    def _convert_numpy_types(obj):
        """将numpy类型转换为Python原生类型，以便JSON序列化"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return FaceAnalysisResponse._convert_numpy_types(obj.tolist())
        elif isinstance(obj, list):
            return [FaceAnalysisResponse._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: FaceAnalysisResponse._convert_numpy_types(value) for key, value in obj.items()}
        else:
            return obj


class FaceListResponse(BaseModel):
    """人脸列表响应"""
    faces: List[Dict[str, Any]] = Field(..., description="人脸列表")
    total: int = Field(..., description="总数")


class FaceUpdateRequest(BaseModel):
    """人脸更新请求"""
    name: Optional[str] = Field(None, description="新姓名")
    metadata: Optional[Dict[str, Any]] = Field(None, description="新元数据")


class FaceDeleteResponse(BaseModel):
    """人脸删除响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")


class FaceDetection(BaseModel):
    """人脸检测结果"""
    bbox: List[int] = Field(..., description="边界框 [x1, y1, x2, y2]")
    confidence: float = Field(..., description="检测置信度")
    face_id: str = Field(..., description="人脸ID")
    landmarks: Optional[List[List[float]]] = Field(None, description="关键点坐标")
    age: Optional[int] = Field(None, description="年龄估计")
    gender: Optional[str] = Field(None, description="性别（Male/Female）")
    embedding: Optional[List[float]] = Field(None, description="人脸特征向量")
    quality_score: Optional[float] = Field(None, description="人脸质量分数")
    # Emotion recognition fields
    emotion: Optional[str] = Field(None, description="情绪（happy/sad/angry/neutral等）")
    emotion_confidence: Optional[float] = Field(None, description="情绪识别置信度")
    emotion_scores: Optional[Dict[str, float]] = Field(None, description="所有情绪的概率分数")
    # Mask detection fields
    has_mask: Optional[bool] = Field(None, description="是否戴口罩")
    mask_confidence: Optional[float] = Field(None, description="口罩检测置信度")


class FaceRecognitionResult(BaseModel):
    """人脸识别结果（单个）"""
    face_id: int = Field(..., description="人脸ID")
    person_name: str = Field(..., description="识别的人员姓名")
    confidence: float = Field(..., description="识别置信度")
    bbox: List[float] = Field(..., description="边界框")
    landmarks: Optional[Dict[str, Any]] = Field(None, description="关键点")
    candidates: Optional[List[Dict[str, Any]]] = Field(None, description="候选结果")


class FaceData(BaseModel):
    """人脸数据"""
    person_name: str = Field(..., description="人员姓名")
    face_id: str = Field(..., description="人脸ID")
    embedding: List[float] = Field(..., description="特征向量")
    metadata: Dict[str, Any] = Field(..., description="元数据")
