"""主处理器模块 - 整合所有核心功能"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np

from ..database import AbstractFaceDB, get_default_database
from .attendance import AttendanceSystem, AttendanceType
from .stranger import AlertLevel, StrangerDetector

# 条件导入以支持测试
try:
    from ..models.insightface.recognizer import InsightFaceRecognizer
except ImportError:
    # 用于测试的mock基类
    class InsightFaceRecognizer:
        def recognize_faces(self, image):
            return [{"name": "MockUser", "confidence": 0.9}]


logging.basicConfig(level=logging.INFO)


class ProcessingMode(Enum):
    """处理模式枚举"""

    ATTENDANCE_ONLY = "attendance_only"  # 仅考勤模式
    SECURITY_ONLY = "security_only"  # 仅安全模式（陌生人检测）
    FULL_MODE = "full_mode"  # 完整模式（考勤+安全）
    RECOGNITION_ONLY = "recognition_only"  # 仅识别模式


class ProcessingResult:
    """处理结果数据类"""

    def __init__(
        self,
        timestamp: datetime,
        mode: ProcessingMode,
        success: bool = True,
        message: str = "",
        recognition_results: Optional[List[Dict]] = None,
        attendance_result: Optional[Dict] = None,
        stranger_result: Optional[Dict] = None,
        image_path: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        self.timestamp = timestamp
        self.mode = mode
        self.success = success
        self.message = message
        self.recognition_results = recognition_results or []
        self.attendance_result = attendance_result
        self.stranger_result = stranger_result
        self.image_path = image_path
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "mode": self.mode.value,
            "success": self.success,
            "message": self.message,
            "recognition_results": self.recognition_results,
            "attendance_result": self.attendance_result,
            "stranger_result": self.stranger_result,
            "image_path": self.image_path,
            "metadata": self.metadata,
        }


class VideoProcessor:
    """视频处理器 - 统一的人脸识别处理核心"""

    def __init__(
        self,
        recognizer: Optional[InsightFaceRecognizer] = None,
        database: Optional[AbstractFaceDB] = None,
        processing_mode: ProcessingMode = ProcessingMode.FULL_MODE,
        **kwargs,
    ):
        """
        初始化视频处理器

        Args:
            recognizer: 人脸识别器实例
            database: 数据库实例
            processing_mode: 处理模式
            **kwargs: 其他配置参数
        """
        # 初始化组件
        self.recognizer = recognizer or InsightFaceRecognizer()
        self.database = database or get_default_database()
        self.processing_mode = processing_mode

        # 初始化子系统
        self.attendance_system = None
        self.stranger_detector = None

        if processing_mode in [ProcessingMode.ATTENDANCE_ONLY, ProcessingMode.FULL_MODE]:
            self.attendance_system = AttendanceSystem(
                recognizer=self.recognizer, database=self.database, **kwargs.get("attendance_config", {})
            )

        if processing_mode in [ProcessingMode.SECURITY_ONLY, ProcessingMode.FULL_MODE]:
            self.stranger_detector = StrangerDetector(
                recognizer=self.recognizer, database=self.database, **kwargs.get("stranger_config", {})
            )

        # 处理记录
        self.processing_history: List[ProcessingResult] = []

        # 回调函数
        self.on_face_detected: Optional[Callable] = None
        self.on_attendance_event: Optional[Callable] = None
        self.on_stranger_alert: Optional[Callable] = None

        logging.info(f"视频处理器初始化完成，模式: {processing_mode.value}")

    def set_callbacks(
        self,
        on_face_detected: Optional[Callable] = None,
        on_attendance_event: Optional[Callable] = None,
        on_stranger_alert: Optional[Callable] = None,
    ):
        """
        设置回调函数

        Args:
            on_face_detected: 人脸检测回调
            on_attendance_event: 考勤事件回调
            on_stranger_alert: 陌生人警报回调
        """
        self.on_face_detected = on_face_detected
        self.on_attendance_event = on_attendance_event
        self.on_stranger_alert = on_stranger_alert

    def process_image(
        self,
        image: np.ndarray,
        location: Optional[str] = None,
        attendance_type: Optional[AttendanceType] = None,
        notes: Optional[str] = None,
        save_image: bool = False,
    ) -> ProcessingResult:
        """
        处理单张图像

        Args:
            image: 输入图像
            location: 位置信息
            attendance_type: 考勤类型（仅考勤模式需要）
            notes: 备注信息
            save_image: 是否保存图像

        Returns:
            处理结果
        """
        current_time = datetime.now()

        try:
            # 基础人脸识别
            recognition_results = self.recognizer.recognize_faces(image)

            # 初始化结果
            result = ProcessingResult(
                timestamp=current_time,
                mode=self.processing_mode,
                recognition_results=recognition_results,
                metadata={"location": location, "notes": notes},
            )

            # 触发人脸检测回调
            if self.on_face_detected and recognition_results:
                try:
                    self.on_face_detected(recognition_results, location)
                except Exception as e:
                    logging.error(f"人脸检测回调执行失败: {e}")

            # 根据模式处理
            if self.processing_mode == ProcessingMode.RECOGNITION_ONLY:
                result.success = len(recognition_results) > 0
                result.message = f"检测到 {len(recognition_results)} 个人脸"

            elif self.processing_mode == ProcessingMode.ATTENDANCE_ONLY:
                if attendance_type and self.attendance_system:
                    attendance_result = self._process_attendance(image, attendance_type, notes)
                    result.attendance_result = attendance_result
                    result.success = attendance_result.get("success", False)
                    result.message = attendance_result.get("message", "")
                else:
                    result.success = False
                    result.message = "考勤模式需要指定考勤类型"

            elif self.processing_mode == ProcessingMode.SECURITY_ONLY:
                if self.stranger_detector:
                    stranger_result = self.stranger_detector.detect_stranger(image, location, save_image)
                    result.stranger_result = stranger_result
                    result.success = True
                    result.message = "陌生人检测完成"

                    # 触发陌生人警报回调
                    if (
                        stranger_result.get("is_stranger")
                        and stranger_result.get("alert_generated")
                        and self.on_stranger_alert
                    ):
                        try:
                            self.on_stranger_alert(stranger_result.get("alert"))
                        except Exception as e:
                            logging.error(f"陌生人警报回调执行失败: {e}")

            elif self.processing_mode == ProcessingMode.FULL_MODE:
                # 执行陌生人检测
                if self.stranger_detector:
                    stranger_result = self.stranger_detector.detect_stranger(image, location, save_image)
                    result.stranger_result = stranger_result

                    # 触发陌生人警报回调
                    if (
                        stranger_result.get("is_stranger")
                        and stranger_result.get("alert_generated")
                        and self.on_stranger_alert
                    ):
                        try:
                            self.on_stranger_alert(stranger_result.get("alert"))
                        except Exception as e:
                            logging.error(f"陌生人警报回调执行失败: {e}")

                # 如果指定了考勤类型，执行考勤处理
                if attendance_type and self.attendance_system:
                    attendance_result = self._process_attendance(image, attendance_type, notes)
                    result.attendance_result = attendance_result

                    # 触发考勤事件回调
                    if attendance_result.get("success") and self.on_attendance_event:
                        try:
                            self.on_attendance_event(attendance_result)
                        except Exception as e:
                            logging.error(f"考勤事件回调执行失败: {e}")

                result.success = True
                result.message = "完整模式处理完成"

            # 保存处理记录
            self.processing_history.append(result)

            # 清理历史记录（保留最近1000条）
            if len(self.processing_history) > 1000:
                self.processing_history = self.processing_history[-1000:]

            return result

        except Exception as e:
            logging.error(f"图像处理失败: {e}")
            error_result = ProcessingResult(
                timestamp=current_time,
                mode=self.processing_mode,
                success=False,
                message=f"处理失败: {str(e)}",
                metadata={"location": location, "error": str(e)},
            )
            self.processing_history.append(error_result)
            return error_result

    def _process_attendance(
        self, image: np.ndarray, attendance_type: AttendanceType, notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理考勤逻辑

        Args:
            image: 输入图像
            attendance_type: 考勤类型
            notes: 备注信息

        Returns:
            考勤处理结果
        """
        if not self.attendance_system:
            return {"success": False, "message": "考勤系统未初始化"}

        if attendance_type == AttendanceType.CHECK_IN:
            return self.attendance_system.check_in(image, notes)
        elif attendance_type == AttendanceType.CHECK_OUT:
            return self.attendance_system.check_out(image, notes)
        elif attendance_type == AttendanceType.BREAK_OUT:
            return self.attendance_system.break_out(image, notes)
        elif attendance_type == AttendanceType.BREAK_IN:
            return self.attendance_system.break_in(image, notes)
        else:
            return {"success": False, "message": f"不支持的考勤类型: {attendance_type}"}

    def process_stream(
        self,
        source: Union[str, int],
        on_frame_processed: Optional[Callable] = None,
        max_fps: int = 30,
        save_frames: bool = False,
    ) -> None:
        """
        处理视频流

        Args:
            source: 视频源（文件路径、RTSP地址或摄像头ID）
            on_frame_processed: 帧处理完成回调
            max_fps: 最大帧率
            save_frames: 是否保存帧
        """
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            logging.error(f"无法打开视频源: {source}")
            return

        frame_interval = 1.0 / max_fps
        last_process_time = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = datetime.now().timestamp()

                # 控制帧率
                if current_time - last_process_time < frame_interval:
                    continue

                # 处理帧
                result = self.process_image(frame, location=f"stream_{source}", save_image=save_frames)

                # 触发帧处理回调
                if on_frame_processed:
                    try:
                        on_frame_processed(frame, result)
                    except Exception as e:
                        logging.error(f"帧处理回调执行失败: {e}")

                last_process_time = current_time

        except KeyboardInterrupt:
            logging.info("视频流处理被用户中断")
        except Exception as e:
            logging.error(f"视频流处理失败: {e}")
        finally:
            cap.release()
            logging.info("视频流处理结束")

    def get_processing_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        获取处理统计信息

        Args:
            hours: 统计时间范围（小时）

        Returns:
            统计信息
        """
        from datetime import timedelta

        current_time = datetime.now()
        time_threshold = current_time - timedelta(hours=hours)

        # 筛选时间范围内的记录
        recent_results = [r for r in self.processing_history if r.timestamp > time_threshold]

        # 基本统计
        total_processed = len(recent_results)
        successful = len([r for r in recent_results if r.success])
        failed = total_processed - successful

        # 按模式统计
        mode_stats = {}
        for mode in ProcessingMode:
            mode_stats[mode.value] = len([r for r in recent_results if r.mode == mode])

        # 考勤统计
        attendance_stats = {"total_attendance": 0, "successful_attendance": 0, "by_type": {}}

        if self.attendance_system:
            attendance_records = self.attendance_system.get_attendance_records()
            attendance_stats["total_attendance"] = len(attendance_records)
            attendance_stats["successful_attendance"] = len(attendance_records)

            # 按类型统计
            for att_type in AttendanceType:
                attendance_stats["by_type"][att_type.value] = len(
                    [r for r in attendance_records if r["attendance_type"] == att_type.value]
                )

        # 陌生人检测统计
        stranger_stats = {}
        if self.stranger_detector:
            stranger_stats = self.stranger_detector.get_detection_statistics(hours)

        return {
            "time_range_hours": hours,
            "processing_statistics": {
                "total_processed": total_processed,
                "successful": successful,
                "failed": failed,
                "success_rate": successful / total_processed if total_processed > 0 else 0,
                "mode_statistics": mode_stats,
            },
            "attendance_statistics": attendance_stats,
            "stranger_statistics": stranger_stats,
            "current_mode": self.processing_mode.value,
        }

    def change_mode(self, new_mode: ProcessingMode, **kwargs):
        """
        切换处理模式

        Args:
            new_mode: 新的处理模式
            **kwargs: 新模式的配置参数
        """
        old_mode = self.processing_mode
        self.processing_mode = new_mode

        # 重新初始化子系统
        if new_mode in [ProcessingMode.ATTENDANCE_ONLY, ProcessingMode.FULL_MODE]:
            if not self.attendance_system:
                self.attendance_system = AttendanceSystem(
                    recognizer=self.recognizer, database=self.database, **kwargs.get("attendance_config", {})
                )

        if new_mode in [ProcessingMode.SECURITY_ONLY, ProcessingMode.FULL_MODE]:
            if not self.stranger_detector:
                self.stranger_detector = StrangerDetector(
                    recognizer=self.recognizer, database=self.database, **kwargs.get("stranger_config", {})
                )

        logging.info(f"处理模式已切换: {old_mode.value} -> {new_mode.value}")

    def clear_history(self):
        """清空处理历史记录"""
        self.processing_history.clear()
        logging.info("处理历史记录已清空")

    def get_recent_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的处理结果

        Args:
            limit: 返回数量限制

        Returns:
            最近的处理结果列表
        """
        recent = self.processing_history[-limit:] if limit else self.processing_history
        return [result.to_dict() for result in reversed(recent)]
