"""人脸质量评估模块"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import math

logging.basicConfig(level=logging.INFO)


class QualityLevel(Enum):
    """质量等级"""
    EXCELLENT = "excellent"  # 优秀 (90-100)
    GOOD = "good"           # 良好 (70-89)
    FAIR = "fair"           # 一般 (50-69)
    POOR = "poor"           # 较差 (30-49)
    VERY_POOR = "very_poor" # 很差 (0-29)


@dataclass
class QualityMetrics:
    """质量评估指标数据类"""
    
    # 整体质量分数 (0-100)
    overall_score: float = 0.0
    quality_level: QualityLevel = QualityLevel.POOR
    
    # 各项指标分数
    sharpness_score: float = 0.0      # 清晰度
    brightness_score: float = 0.0     # 亮度
    contrast_score: float = 0.0       # 对比度
    pose_score: float = 0.0           # 姿态
    size_score: float = 0.0           # 尺寸
    symmetry_score: float = 0.0       # 对称性
    occlusion_score: float = 0.0      # 遮挡
    
    # 详细信息
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # 人脸边界框
    face_size: Tuple[int, int] = (0, 0)                   # 人脸尺寸
    pitch: float = 0.0                                     # 俯仰角
    yaw: float = 0.0                                       # 偏航角
    roll: float = 0.0                                      # 翻滚角
    
    # 建议信息
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        
        # 计算整体质量等级
        if self.overall_score >= 90:
            self.quality_level = QualityLevel.EXCELLENT
        elif self.overall_score >= 70:
            self.quality_level = QualityLevel.GOOD
        elif self.overall_score >= 50:
            self.quality_level = QualityLevel.FAIR
        elif self.overall_score >= 30:
            self.quality_level = QualityLevel.POOR
        else:
            self.quality_level = QualityLevel.VERY_POOR
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'overall_score': round(self.overall_score, 2),
            'quality_level': self.quality_level.value,
            'metrics': {
                'sharpness': round(self.sharpness_score, 2),
                'brightness': round(self.brightness_score, 2),
                'contrast': round(self.contrast_score, 2),
                'pose': round(self.pose_score, 2),
                'size': round(self.size_score, 2),
                'symmetry': round(self.symmetry_score, 2),
                'occlusion': round(self.occlusion_score, 2)
            },
            'face_info': {
                'bbox': self.face_bbox,
                'size': self.face_size,
                'pose': {
                    'pitch': round(self.pitch, 2),
                    'yaw': round(self.yaw, 2),
                    'roll': round(self.roll, 2)
                }
            },
            'recommendations': self.recommendations
        }


class FaceQualityAssessor:
    """人脸质量评估器"""
    
    def __init__(self, 
                 min_face_size: int = 80,
                 max_pose_angle: float = 30.0,
                 min_brightness: float = 80,
                 max_brightness: float = 200):
        """
        初始化人脸质量评估器
        
        Args:
            min_face_size: 最小人脸尺寸
            max_pose_angle: 最大姿态角度
            min_brightness: 最小亮度值
            max_brightness: 最大亮度值
        """
        self.min_face_size = min_face_size
        self.max_pose_angle = max_pose_angle
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        
        # 加载人脸检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 尝试加载更精确的检测器
        try:
            self.face_detector = cv2.dnn.readNetFromTensorflow(
                cv2.samples.findFile('opencv_face_detector_uint8.pb'),
                cv2.samples.findFile('opencv_face_detector.pbtxt')
            )
            self.use_dnn = True
        except:
            self.use_dnn = False
            logging.info("DNN人脸检测器不可用，使用Haar级联检测器")
    
    def assess_quality(self, image: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> QualityMetrics:
        """
        评估人脸质量
        
        Args:
            image: 输入图像
            face_bbox: 人脸边界框 (x, y, w, h)，如果为None则自动检测
            
        Returns:
            质量评估结果
        """
        try:
            # 如果没有提供人脸框，自动检测
            if face_bbox is None:
                face_bbox = self._detect_face(image)
                if face_bbox is None:
                    return QualityMetrics(recommendations=["未检测到人脸"])
            
            # 提取人脸区域
            x, y, w, h = face_bbox
            face_roi = image[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return QualityMetrics(recommendations=["人脸区域无效"])
            
            # 创建质量评估结果
            metrics = QualityMetrics()
            metrics.face_bbox = face_bbox
            metrics.face_size = (w, h)
            
            # 评估各项指标
            metrics.sharpness_score = self._assess_sharpness(face_roi)
            metrics.brightness_score = self._assess_brightness(face_roi)
            metrics.contrast_score = self._assess_contrast(face_roi)
            metrics.size_score = self._assess_size(w, h)
            metrics.symmetry_score = self._assess_symmetry(face_roi)
            metrics.occlusion_score = self._assess_occlusion(face_roi)
            
            # 评估姿态
            pose_angles = self._assess_pose(face_roi)
            metrics.pitch, metrics.yaw, metrics.roll = pose_angles
            metrics.pose_score = self._calculate_pose_score(pose_angles)
            
            # 计算整体分数
            metrics.overall_score = self._calculate_overall_score(metrics)
            
            # 生成建议
            metrics.recommendations = self._generate_recommendations(metrics)
            
            return metrics
            
        except Exception as e:
            logging.error(f"评估人脸质量失败: {e}")
            return QualityMetrics(recommendations=[f"评估失败: {str(e)}"])
    
    def _detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """检测人脸"""
        try:
            if self.use_dnn:
                return self._detect_face_dnn(image)
            else:
                return self._detect_face_haar(image)
        except:
            return self._detect_face_haar(image)
    
    def _detect_face_haar(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """使用Haar级联检测人脸"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # 返回最大的人脸
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            return tuple(largest_face)
        
        return None
    
    def _detect_face_dnn(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """使用DNN检测人脸"""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        best_confidence = 0
        best_box = None
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5 and confidence > best_confidence:
                best_confidence = confidence
                
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                best_box = (x1, y1, x2 - x1, y2 - y1)
        
        return best_box
    
    def _assess_sharpness(self, face_roi: np.ndarray) -> float:
        """评估清晰度"""
        try:
            # 转换为灰度图
            if len(face_roi.shape) == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_roi
            
            # 使用拉普拉斯算子计算清晰度
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 归一化到0-100分
            sharpness_score = min(100, laplacian_var / 10)
            
            return sharpness_score
            
        except Exception as e:
            logging.error(f"评估清晰度失败: {e}")
            return 0.0
    
    def _assess_brightness(self, face_roi: np.ndarray) -> float:
        """评估亮度"""
        try:
            # 计算平均亮度
            if len(face_roi.shape) == 3:
                # RGB图像，转换为灰度
                gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_roi
            
            mean_brightness = np.mean(gray)
            
            # 计算亮度分数（最佳亮度范围）
            if self.min_brightness <= mean_brightness <= self.max_brightness:
                score = 100
            elif mean_brightness < self.min_brightness:
                score = max(0, (mean_brightness / self.min_brightness) * 100)
            else:
                score = max(0, 100 - ((mean_brightness - self.max_brightness) / 55) * 100)
            
            return score
            
        except Exception as e:
            logging.error(f"评估亮度失败: {e}")
            return 0.0
    
    def _assess_contrast(self, face_roi: np.ndarray) -> float:
        """评估对比度"""
        try:
            if len(face_roi.shape) == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_roi
            
            # 计算标准差作为对比度指标
            contrast = np.std(gray)
            
            # 归一化到0-100分
            contrast_score = min(100, (contrast / 64) * 100)
            
            return contrast_score
            
        except Exception as e:
            logging.error(f"评估对比度失败: {e}")
            return 0.0
    
    def _assess_size(self, width: int, height: int) -> float:
        """评估人脸尺寸"""
        try:
            # 计算人脸面积
            face_area = width * height
            
            # 理想尺寸范围
            ideal_min = self.min_face_size * self.min_face_size
            ideal_max = 300 * 300
            
            if face_area >= ideal_max:
                return 100
            elif face_area >= ideal_min:
                return 50 + (face_area - ideal_min) / (ideal_max - ideal_min) * 50
            else:
                return max(0, (face_area / ideal_min) * 50)
                
        except Exception as e:
            logging.error(f"评估尺寸失败: {e}")
            return 0.0
    
    def _assess_symmetry(self, face_roi: np.ndarray) -> float:
        """评估对称性"""
        try:
            if len(face_roi.shape) == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_roi
            
            h, w = gray.shape
            
            # 分割左右两半
            left_half = gray[:, :w//2]
            right_half = gray[:, w//2:]
            
            # 翻转右半部分
            right_half_flipped = cv2.flip(right_half, 1)
            
            # 调整尺寸使其匹配
            if left_half.shape[1] != right_half_flipped.shape[1]:
                min_width = min(left_half.shape[1], right_half_flipped.shape[1])
                left_half = left_half[:, :min_width]
                right_half_flipped = right_half_flipped[:, :min_width]
            
            # 计算相似度
            diff = cv2.absdiff(left_half, right_half_flipped)
            symmetry = 100 - (np.mean(diff) / 255 * 100)
            
            return max(0, symmetry)
            
        except Exception as e:
            logging.error(f"评估对称性失败: {e}")
            return 50.0  # 默认分数
    
    def _assess_occlusion(self, face_roi: np.ndarray) -> float:
        """评估遮挡程度"""
        try:
            # 简化的遮挡检测：检查边缘区域的完整性
            if len(face_roi.shape) == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_roi
            
            h, w = gray.shape
            
            # 检查四个边角区域的像素值变化
            corner_size = min(h, w) // 8
            
            corners = [
                gray[:corner_size, :corner_size],                    # 左上
                gray[:corner_size, -corner_size:],                   # 右上
                gray[-corner_size:, :corner_size],                   # 左下
                gray[-corner_size:, -corner_size:]                   # 右下
            ]
            
            total_variation = 0
            for corner in corners:
                total_variation += np.std(corner)
            
            # 归一化分数
            occlusion_score = min(100, (total_variation / 4) / 30 * 100)
            
            return occlusion_score
            
        except Exception as e:
            logging.error(f"评估遮挡失败: {e}")
            return 70.0  # 默认分数
    
    def _assess_pose(self, face_roi: np.ndarray) -> Tuple[float, float, float]:
        """评估人脸姿态角度"""
        try:
            # 简化的姿态估计：基于人脸特征点的几何分析
            if len(face_roi.shape) == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_roi
            
            h, w = gray.shape
            
            # 简单的姿态估计：分析人脸的重心和对称性
            # 计算水平方向的重心偏移（偏航角）
            horizontal_profile = np.mean(gray, axis=0)
            center_x = w // 2
            weighted_center = np.sum(np.arange(w) * horizontal_profile) / np.sum(horizontal_profile)
            yaw = (weighted_center - center_x) / center_x * 45  # 转换为角度
            
            # 计算垂直方向的重心偏移（俯仰角）
            vertical_profile = np.mean(gray, axis=1)
            center_y = h // 2
            weighted_center_y = np.sum(np.arange(h) * vertical_profile) / np.sum(vertical_profile)
            pitch = (weighted_center_y - center_y) / center_y * 45
            
            # 翻滚角度估计（基于眼睛线条的倾斜）
            # 简化处理，假设为0
            roll = 0.0
            
            return pitch, yaw, roll
            
        except Exception as e:
            logging.error(f"评估姿态失败: {e}")
            return 0.0, 0.0, 0.0
    
    def _calculate_pose_score(self, pose_angles: Tuple[float, float, float]) -> float:
        """计算姿态分数"""
        pitch, yaw, roll = pose_angles
        
        # 计算最大角度偏移
        max_angle = max(abs(pitch), abs(yaw), abs(roll))
        
        if max_angle <= self.max_pose_angle * 0.5:
            return 100
        elif max_angle <= self.max_pose_angle:
            return 50 + (self.max_pose_angle - max_angle) / (self.max_pose_angle * 0.5) * 50
        else:
            return max(0, 50 - (max_angle - self.max_pose_angle) / self.max_pose_angle * 50)
    
    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """计算整体分数"""
        # 权重配置
        weights = {
            'sharpness': 0.25,
            'brightness': 0.15,
            'contrast': 0.15,
            'pose': 0.20,
            'size': 0.15,
            'symmetry': 0.05,
            'occlusion': 0.05
        }
        
        overall_score = (
            metrics.sharpness_score * weights['sharpness'] +
            metrics.brightness_score * weights['brightness'] +
            metrics.contrast_score * weights['contrast'] +
            metrics.pose_score * weights['pose'] +
            metrics.size_score * weights['size'] +
            metrics.symmetry_score * weights['symmetry'] +
            metrics.occlusion_score * weights['occlusion']
        )
        
        return min(100, max(0, overall_score))
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if metrics.sharpness_score < 50:
            recommendations.append("图像模糊，请确保相机对焦清晰")
        
        if metrics.brightness_score < 50:
            if np.mean([metrics.brightness_score]) < 40:
                recommendations.append("图像过暗，请增加光照")
            else:
                recommendations.append("图像过亮，请减少光照")
        
        if metrics.contrast_score < 50:
            recommendations.append("对比度不足，请改善光照条件")
        
        if metrics.pose_score < 70:
            recommendations.append("人脸姿态偏离过大，请正对摄像头")
        
        if metrics.size_score < 50:
            recommendations.append("人脸尺寸过小，请靠近摄像头")
        
        if metrics.symmetry_score < 60:
            recommendations.append("人脸不够对称，可能存在遮挡")
        
        if metrics.occlusion_score < 60:
            recommendations.append("检测到可能的遮挡，请移除障碍物")
        
        if not recommendations:
            recommendations.append("图像质量良好")
        
        return recommendations