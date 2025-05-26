"""陌生人检测模块"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

from ..database import get_default_database, AbstractFaceDB

# 条件导入以支持测试
try:
    from ..models.insightface.recognizer import InsightFaceRecognizer
except ImportError:
    # 用于测试的mock基类
    class InsightFaceRecognizer:
        def recognize_faces(self, image):
            return []

logging.basicConfig(level=logging.INFO)


class AlertLevel(Enum):
    """警报级别枚举"""
    LOW = "low"          # 低级警报
    MEDIUM = "medium"    # 中级警报  
    HIGH = "high"        # 高级警报
    CRITICAL = "critical" # 严重警报


class StrangerAlert:
    """陌生人警报数据类"""
    
    def __init__(
        self,
        alert_id: str,
        timestamp: datetime,
        alert_level: AlertLevel,
        location: Optional[str] = None,
        image_path: Optional[str] = None,
        confidence: float = 0.0,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        self.alert_id = alert_id
        self.timestamp = timestamp
        self.alert_level = alert_level
        self.location = location
        self.image_path = image_path
        self.confidence = confidence
        self.description = description
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'alert_level': self.alert_level.value,
            'location': self.location,
            'image_path': self.image_path,
            'confidence': self.confidence,
            'description': self.description,
            'metadata': self.metadata
        }


class StrangerDetector:
    """陌生人检测器"""
    
    def __init__(
        self,
        recognizer: InsightFaceRecognizer,
        database: Optional[AbstractFaceDB] = None,
        stranger_threshold: float = 0.6,
        alert_cooldown_minutes: int = 10,
        max_alerts_per_hour: int = 20
    ):
        """
        初始化陌生人检测器
        
        Args:
            recognizer: 人脸识别器实例
            database: 数据库实例，默认使用系统默认数据库
            stranger_threshold: 陌生人判定阈值，低于此值认为是陌生人
            alert_cooldown_minutes: 警报冷却时间（分钟）
            max_alerts_per_hour: 每小时最大警报数量
        """
        self.recognizer = recognizer
        self.database = database or get_default_database()
        self.stranger_threshold = stranger_threshold
        self.alert_cooldown_minutes = alert_cooldown_minutes
        self.max_alerts_per_hour = max_alerts_per_hour
        
        # 警报记录存储
        self.stranger_alerts: List[StrangerAlert] = []
        
        # 最近检测记录（用于冷却时间判断）
        self.recent_detections: List[Dict] = []
        
        logging.info(f"陌生人检测器初始化完成，阈值: {stranger_threshold}")
    
    def detect_stranger(
        self,
        image: np.ndarray,
        location: Optional[str] = None,
        save_image: bool = True
    ) -> Dict[str, Any]:
        """
        检测陌生人
        
        Args:
            image: 输入图像
            location: 检测位置
            save_image: 是否保存图像
            
        Returns:
            检测结果
        """
        current_time = datetime.now()
        
        try:
            # 进行人脸识别
            recognition_results = self.recognizer.recognize_faces(image)
            
            if not recognition_results:
                return {
                    'is_stranger': False,
                    'reason': '未检测到人脸',
                    'timestamp': current_time.isoformat(),
                    'location': location
                }
            
            # 分析识别结果
            max_confidence = 0.0
            best_match = None
            
            for result in recognition_results:
                confidence = result.get('confidence', 0.0)
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_match = result
            
            # 判断是否为陌生人
            is_stranger = max_confidence < self.stranger_threshold
            
            if not is_stranger:
                return {
                    'is_stranger': False,
                    'recognized_person': best_match.get('name', 'Unknown'),
                    'confidence': max_confidence,
                    'timestamp': current_time.isoformat(),
                    'location': location
                }
            
            # 检查是否需要生成警报
            should_alert = self._should_generate_alert(location, current_time)
            
            result = {
                'is_stranger': True,
                'confidence': max_confidence,
                'threshold': self.stranger_threshold,
                'timestamp': current_time.isoformat(),
                'location': location,
                'alert_generated': should_alert
            }
            
            if should_alert:
                alert = self._generate_alert(
                    image=image,
                    confidence=max_confidence,
                    location=location,
                    timestamp=current_time,
                    save_image=save_image
                )
                result['alert'] = alert.to_dict()
            
            # 记录检测
            self._record_detection(location, current_time, is_stranger, max_confidence)
            
            return result
            
        except Exception as e:
            logging.error(f"陌生人检测失败: {e}")
            return {
                'is_stranger': False,
                'error': str(e),
                'timestamp': current_time.isoformat(),
                'location': location
            }
    
    def _should_generate_alert(self, location: Optional[str], current_time: datetime) -> bool:
        """
        判断是否应该生成警报
        
        Args:
            location: 检测位置
            current_time: 当前时间
            
        Returns:
            是否应该生成警报
        """
        # 检查冷却时间
        cooldown_threshold = current_time - timedelta(minutes=self.alert_cooldown_minutes)
        
        recent_alerts = [
            alert for alert in self.stranger_alerts
            if alert.timestamp > cooldown_threshold and alert.location == location
        ]
        
        if recent_alerts:
            return False
        
        # 检查每小时警报数量限制
        hour_threshold = current_time - timedelta(hours=1)
        hourly_alerts = [
            alert for alert in self.stranger_alerts
            if alert.timestamp > hour_threshold
        ]
        
        if len(hourly_alerts) >= self.max_alerts_per_hour:
            return False
        
        return True
    
    def _generate_alert(
        self,
        image: np.ndarray,
        confidence: float,
        location: Optional[str],
        timestamp: datetime,
        save_image: bool = True
    ) -> StrangerAlert:
        """
        生成陌生人警报
        
        Args:
            image: 检测到的图像
            confidence: 识别置信度
            location: 检测位置
            timestamp: 检测时间
            save_image: 是否保存图像
            
        Returns:
            陌生人警报对象
        """
        import uuid
        
        alert_id = str(uuid.uuid4())
        
        # 根据置信度确定警报级别
        if confidence < 0.2:
            alert_level = AlertLevel.CRITICAL
        elif confidence < 0.4:
            alert_level = AlertLevel.HIGH
        elif confidence < 0.5:
            alert_level = AlertLevel.MEDIUM
        else:
            alert_level = AlertLevel.LOW
        
        # 保存图像（可选）
        image_path = None
        if save_image:
            image_path = self._save_stranger_image(image, alert_id, timestamp)
        
        description = f"在{location or '未知位置'}检测到陌生人，置信度: {confidence:.3f}"
        
        alert = StrangerAlert(
            alert_id=alert_id,
            timestamp=timestamp,
            alert_level=alert_level,
            location=location,
            image_path=image_path,
            confidence=confidence,
            description=description,
            metadata={
                'recognition_confidence': confidence,
                'threshold_used': self.stranger_threshold
            }
        )
        
        # 保存警报
        self.stranger_alerts.append(alert)
        
        logging.warning(f"陌生人警报: {description}")
        
        return alert
    
    def _save_stranger_image(self, image: np.ndarray, alert_id: str, timestamp: datetime) -> str:
        """
        保存陌生人图像
        
        Args:
            image: 图像数据
            alert_id: 警报ID
            timestamp: 时间戳
            
        Returns:
            保存的图像路径
        """
        import cv2
        import os
        
        # 创建陌生人图像目录
        stranger_dir = "data/strangers"
        os.makedirs(stranger_dir, exist_ok=True)
        
        # 生成文件名
        filename = f"stranger_{alert_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        image_path = os.path.join(stranger_dir, filename)
        
        # 保存图像
        cv2.imwrite(image_path, image)
        
        return image_path
    
    def _record_detection(
        self,
        location: Optional[str],
        timestamp: datetime,
        is_stranger: bool,
        confidence: float
    ):
        """
        记录检测结果
        
        Args:
            location: 检测位置
            timestamp: 检测时间
            is_stranger: 是否为陌生人
            confidence: 置信度
        """
        detection_record = {
            'location': location,
            'timestamp': timestamp,
            'is_stranger': is_stranger,
            'confidence': confidence
        }
        
        self.recent_detections.append(detection_record)
        
        # 清理旧记录（保留最近1小时）
        hour_ago = timestamp - timedelta(hours=1)
        self.recent_detections = [
            record for record in self.recent_detections
            if record['timestamp'] > hour_ago
        ]
    
    def get_stranger_alerts(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        location: Optional[str] = None,
        alert_level: Optional[AlertLevel] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取陌生人警报记录
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            location: 位置过滤
            alert_level: 警报级别过滤
            limit: 返回数量限制
            
        Returns:
            警报记录列表
        """
        alerts = self.stranger_alerts
        
        # 按条件过滤
        if start_date:
            alerts = [a for a in alerts if a.timestamp >= start_date]
        
        if end_date:
            alerts = [a for a in alerts if a.timestamp <= end_date]
        
        if location:
            alerts = [a for a in alerts if a.location == location]
        
        if alert_level:
            alerts = [a for a in alerts if a.alert_level == alert_level]
        
        # 按时间排序
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        # 应用数量限制
        if limit:
            alerts = alerts[:limit]
        
        return [alert.to_dict() for alert in alerts]
    
    def get_detection_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        获取检测统计信息
        
        Args:
            hours: 统计时间范围（小时）
            
        Returns:
            统计信息
        """
        current_time = datetime.now()
        time_threshold = current_time - timedelta(hours=hours)
        
        # 统计检测记录
        recent_detections = [
            record for record in self.recent_detections
            if record['timestamp'] > time_threshold
        ]
        
        # 统计警报
        recent_alerts = [
            alert for alert in self.stranger_alerts
            if alert.timestamp > time_threshold
        ]
        
        stranger_count = len([r for r in recent_detections if r['is_stranger']])
        known_person_count = len([r for r in recent_detections if not r['is_stranger']])
        
        # 按位置统计
        location_stats = {}
        for record in recent_detections:
            location = record['location'] or 'unknown'
            if location not in location_stats:
                location_stats[location] = {'total': 0, 'strangers': 0}
            location_stats[location]['total'] += 1
            if record['is_stranger']:
                location_stats[location]['strangers'] += 1
        
        # 按警报级别统计
        alert_level_stats = {}
        for level in AlertLevel:
            alert_level_stats[level.value] = len([
                a for a in recent_alerts if a.alert_level == level
            ])
        
        return {
            'time_range_hours': hours,
            'total_detections': len(recent_detections),
            'stranger_detections': stranger_count,
            'known_person_detections': known_person_count,
            'total_alerts': len(recent_alerts),
            'stranger_rate': stranger_count / len(recent_detections) if recent_detections else 0,
            'location_statistics': location_stats,
            'alert_level_statistics': alert_level_stats,
            'threshold_used': self.stranger_threshold
        }
    
    def clear_alerts(self):
        """清空警报记录（仅用于测试）"""
        self.stranger_alerts.clear()
        self.recent_detections.clear()
        logging.info("陌生人警报记录已清空")
    
    def update_threshold(self, new_threshold: float):
        """
        更新陌生人判定阈值
        
        Args:
            new_threshold: 新的阈值
        """
        old_threshold = self.stranger_threshold
        self.stranger_threshold = new_threshold
        logging.info(f"陌生人检测阈值已更新: {old_threshold} -> {new_threshold}")