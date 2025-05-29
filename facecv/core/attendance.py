"""考勤系统模块"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..database import AbstractFaceDB, get_default_database

# 条件导入以支持测试
try:
    from ..models.insightface.recognizer import InsightFaceRecognizer
except ImportError:
    # 用于测试的mock基类
    class InsightFaceRecognizer:
        def recognize_faces(self, image):
            return []


logging.basicConfig(level=logging.INFO)


class AttendanceType(Enum):
    """考勤类型枚举"""

    CHECK_IN = "check_in"  # 签到
    CHECK_OUT = "check_out"  # 签退
    BREAK_OUT = "break_out"  # 外出
    BREAK_IN = "break_in"  # 回来


class AttendanceRecord:
    """考勤记录数据类"""

    def __init__(
        self,
        person_id: str,
        person_name: str,
        attendance_type: AttendanceType,
        timestamp: datetime,
        confidence: float,
        face_id: Optional[str] = None,
        image_path: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        self.person_id = person_id
        self.person_name = person_name
        self.attendance_type = attendance_type
        self.timestamp = timestamp
        self.confidence = confidence
        self.face_id = face_id
        self.image_path = image_path
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "person_id": self.person_id,
            "person_name": self.person_name,
            "attendance_type": self.attendance_type.value,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "face_id": self.face_id,
            "image_path": self.image_path,
            "notes": self.notes,
        }


class AttendanceSystem:
    """考勤系统实现"""

    def __init__(
        self,
        recognizer: InsightFaceRecognizer,
        database: Optional[AbstractFaceDB] = None,
        min_confidence: float = 0.8,
        duplicate_check_minutes: int = 5,
    ):
        """
        初始化考勤系统

        Args:
            recognizer: 人脸识别器实例
            database: 数据库实例，默认使用系统默认数据库
            min_confidence: 最小置信度阈值
            duplicate_check_minutes: 重复打卡检查时间间隔（分钟）
        """
        self.recognizer = recognizer
        self.database = database or get_default_database()
        self.min_confidence = min_confidence
        self.duplicate_check_minutes = duplicate_check_minutes

        # 考勤记录存储（内存中，实际应用中可存储到数据库）
        self.attendance_records: List[AttendanceRecord] = []

        # 初始化考勤表（如果需要持久化存储）
        self._init_attendance_table()

    def _init_attendance_table(self):
        """初始化考勤表（扩展数据库表结构）"""
        # 这里可以扩展数据库表结构以支持考勤记录
        # 目前使用内存存储，实际应用中建议存储到数据库
        logging.info("考勤系统初始化完成")

    def recognize_person(self, image: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
        """
        识别图像中的人员

        Args:
            image: 输入图像

        Returns:
            (person_id, person_name, confidence): 识别结果
        """
        try:
            # 使用识别器识别人脸
            results = self.recognizer.recognize_faces(image)

            if not results:
                return None, None, 0.0

            # 取置信度最高的结果
            best_result = max(results, key=lambda x: x.get("confidence", 0))

            person_name = best_result.get("name", "Unknown")
            confidence = best_result.get("confidence", 0.0)

            # 查询数据库获取person_id
            faces = self.database.query_faces_by_name(person_name)
            person_id = faces[0]["id"] if faces else None

            return person_id, person_name, confidence

        except Exception as e:
            logging.error(f"人员识别失败: {e}")
            return None, None, 0.0

    def _check_duplicate_attendance(
        self, person_id: str, attendance_type: AttendanceType, current_time: datetime
    ) -> bool:
        """
        检查是否重复打卡

        Args:
            person_id: 人员ID
            attendance_type: 考勤类型
            current_time: 当前时间

        Returns:
            True: 重复打卡, False: 非重复打卡
        """
        time_threshold = current_time - timedelta(minutes=self.duplicate_check_minutes)

        for record in self.attendance_records:
            if (
                record.person_id == person_id
                and record.attendance_type == attendance_type
                and record.timestamp > time_threshold
            ):
                return True

        return False

    def check_in(self, image: np.ndarray, notes: Optional[str] = None) -> Dict[str, Any]:
        """
        签到功能

        Args:
            image: 人脸图像
            notes: 备注信息

        Returns:
            考勤结果
        """
        return self._process_attendance(image, AttendanceType.CHECK_IN, notes)

    def check_out(self, image: np.ndarray, notes: Optional[str] = None) -> Dict[str, Any]:
        """
        签退功能

        Args:
            image: 人脸图像
            notes: 备注信息

        Returns:
            考勤结果
        """
        return self._process_attendance(image, AttendanceType.CHECK_OUT, notes)

    def break_out(self, image: np.ndarray, notes: Optional[str] = None) -> Dict[str, Any]:
        """
        外出功能

        Args:
            image: 人脸图像
            notes: 备注信息

        Returns:
            考勤结果
        """
        return self._process_attendance(image, AttendanceType.BREAK_OUT, notes)

    def break_in(self, image: np.ndarray, notes: Optional[str] = None) -> Dict[str, Any]:
        """
        回来功能

        Args:
            image: 人脸图像
            notes: 备注信息

        Returns:
            考勤结果
        """
        return self._process_attendance(image, AttendanceType.BREAK_IN, notes)

    def _process_attendance(
        self, image: np.ndarray, attendance_type: AttendanceType, notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理考勤流程

        Args:
            image: 人脸图像
            attendance_type: 考勤类型
            notes: 备注信息

        Returns:
            考勤处理结果
        """
        current_time = datetime.now()

        try:
            # 识别人员
            person_id, person_name, confidence = self.recognize_person(image)

            if not person_id or confidence < self.min_confidence:
                return {
                    "success": False,
                    "message": f"人脸识别失败或置信度过低 (confidence: {confidence:.3f})",
                    "confidence": confidence,
                    "timestamp": current_time.isoformat(),
                }

            # 检查重复打卡
            if self._check_duplicate_attendance(person_id, attendance_type, current_time):
                return {
                    "success": False,
                    "message": f"{person_name} 在 {self.duplicate_check_minutes} 分钟内已进行过{attendance_type.value}",
                    "person_name": person_name,
                    "confidence": confidence,
                    "timestamp": current_time.isoformat(),
                }

            # 创建考勤记录
            record = AttendanceRecord(
                person_id=person_id,
                person_name=person_name,
                attendance_type=attendance_type,
                timestamp=current_time,
                confidence=confidence,
                notes=notes,
            )

            # 保存记录
            self.attendance_records.append(record)

            logging.info(f"考勤记录: {person_name} {attendance_type.value} at {current_time}")

            return {
                "success": True,
                "message": f"{person_name} {attendance_type.value} 成功",
                "person_id": person_id,
                "person_name": person_name,
                "attendance_type": attendance_type.value,
                "confidence": confidence,
                "timestamp": current_time.isoformat(),
                "notes": notes,
            }

        except Exception as e:
            logging.error(f"考勤处理失败: {e}")
            return {"success": False, "message": f"考勤处理出错: {str(e)}", "timestamp": current_time.isoformat()}

    def get_attendance_records(
        self,
        person_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        attendance_type: Optional[AttendanceType] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取考勤记录

        Args:
            person_name: 人员姓名过滤
            start_date: 开始日期
            end_date: 结束日期
            attendance_type: 考勤类型过滤

        Returns:
            考勤记录列表
        """
        records = self.attendance_records

        # 按条件过滤
        if person_name:
            records = [r for r in records if r.person_name == person_name]

        if start_date:
            records = [r for r in records if r.timestamp >= start_date]

        if end_date:
            records = [r for r in records if r.timestamp <= end_date]

        if attendance_type:
            records = [r for r in records if r.attendance_type == attendance_type]

        # 按时间排序
        records.sort(key=lambda x: x.timestamp, reverse=True)

        return [record.to_dict() for record in records]

    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        获取每日考勤汇总

        Args:
            date: 查询日期，默认为今天

        Returns:
            每日汇总信息
        """
        if date is None:
            date = datetime.now()

        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        daily_records = [r for r in self.attendance_records if start_of_day <= r.timestamp < end_of_day]

        # 统计各类型考勤人数
        type_counts = {}
        for att_type in AttendanceType:
            type_counts[att_type.value] = len([r for r in daily_records if r.attendance_type == att_type])

        # 统计总人数
        unique_persons = set(r.person_name for r in daily_records)

        return {
            "date": date.strftime("%Y-%m-%d"),
            "total_records": len(daily_records),
            "unique_persons": len(unique_persons),
            "person_names": list(unique_persons),
            "type_counts": type_counts,
            "records": [record.to_dict() for record in daily_records],
        }

    def clear_records(self):
        """清空考勤记录（仅用于测试）"""
        self.attendance_records.clear()
        logging.info("考勤记录已清空")
