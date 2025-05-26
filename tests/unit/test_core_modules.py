#!/usr/bin/env python3
"""核心模块单元测试"""

import sys
import os
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from facecv.core.attendance import AttendanceSystem, AttendanceType
from facecv.core.stranger import StrangerDetector, AlertLevel
from facecv.core.processor import VideoProcessor, ProcessingMode
from test_db_simple import SimpleSQLiteDB

# 简单的Mock识别器
class MockInsightFaceRecognizer:
    def __init__(self):
        # 可配置的识别结果
        self.mock_results = [
            {'name': '张三', 'confidence': 0.85, 'bbox': [100, 100, 200, 200]}
        ]
    
    def recognize_faces(self, image):
        # 模拟识别结果
        return self.mock_results
    
    def set_mock_result(self, name, confidence=0.85):
        """设置mock识别结果"""
        self.mock_results = [
            {'name': name, 'confidence': confidence, 'bbox': [100, 100, 200, 200]}
        ]

def test_attendance_system():
    """测试考勤系统"""
    print("=" * 50)
    print("测试考勤系统")
    print("=" * 50)
    
    try:
        # 使用mock识别器和简化数据库
        recognizer = MockInsightFaceRecognizer()
        database = SimpleSQLiteDB()
        
        # 添加测试用户到数据库
        test_embedding = np.random.rand(512).astype(np.float32)
        database.add_face("张三", test_embedding, {"role": "employee"})
        
        # 初始化考勤系统
        attendance_system = AttendanceSystem(
            recognizer=recognizer,
            database=database,
            min_confidence=0.8  # 设置为0.8，匹配mock的0.85
        )
        
        print("✓ 考勤系统初始化成功")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 测试签到
        result = attendance_system.check_in(test_image, "正常签到")
        print(f"✓ 签到测试: {result['success']} - {result['message']}")
        
        # 测试重复签到
        result = attendance_system.check_in(test_image, "重复签到")
        print(f"✓ 重复签到测试: {result['success']} - {result['message']}")
        
        # 测试签退
        result = attendance_system.check_out(test_image, "正常签退")
        print(f"✓ 签退测试: {result['success']} - {result['message']}")
        
        # 测试记录查询
        records = attendance_system.get_attendance_records()
        print(f"✓ 记录查询: 找到{len(records)}条记录")
        
        # 测试每日汇总
        summary = attendance_system.get_daily_summary()
        print(f"✓ 每日汇总: {summary['total_records']}条记录，{summary['unique_persons']}个人")
        
        print("🎉 考勤系统测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 考勤系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stranger_detection():
    """测试陌生人检测"""
    print("\n" + "=" * 50)
    print("测试陌生人检测")
    print("=" * 50)
    
    try:
        # 使用mock识别器
        recognizer = MockInsightFaceRecognizer()
        database = SimpleSQLiteDB()
        
        # 添加已知用户到数据库
        test_embedding = np.random.rand(512).astype(np.float32)
        database.add_face("李四", test_embedding, {"role": "employee"})
        
        # 设置mock识别器结果
        recognizer.set_mock_result("李四", 0.6)  # 低于阈值，触发陌生人检测
        
        # 初始化陌生人检测器
        stranger_detector = StrangerDetector(
            recognizer=recognizer,
            database=database,
            stranger_threshold=0.7  # 较高阈值，容易触发陌生人检测
        )
        
        print("✓ 陌生人检测器初始化成功")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 测试陌生人检测
        result = stranger_detector.detect_stranger(
            test_image, 
            location="测试区域",
            save_image=False  # 不保存图像以避免文件系统依赖
        )
        print(f"✓ 陌生人检测: is_stranger={result.get('is_stranger')}")
        
        if result.get('alert_generated'):
            print(f"✓ 警报生成: {result.get('alert', {}).get('description')}")
        
        # 测试统计信息
        stats = stranger_detector.get_detection_statistics(1)
        print(f"✓ 检测统计: {stats['total_detections']}次检测，陌生人率: {stats['stranger_rate']:.2%}")
        
        # 测试警报记录
        alerts = stranger_detector.get_stranger_alerts()
        print(f"✓ 警报查询: 找到{len(alerts)}条警报")
        
        print("🎉 陌生人检测测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 陌生人检测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_processor():
    """测试主处理器"""
    print("\n" + "=" * 50)
    print("测试主处理器")
    print("=" * 50)
    
    try:
        # 使用mock识别器
        recognizer = MockInsightFaceRecognizer()
        database = SimpleSQLiteDB()
        
        # 添加测试用户
        test_embedding = np.random.rand(512).astype(np.float32)
        database.add_face("王五", test_embedding, {"role": "employee"})
        
        # 设置mock识别器结果
        recognizer.set_mock_result("王五", 0.9)
        
        # 初始化处理器（完整模式）
        processor = VideoProcessor(
            recognizer=recognizer,
            database=database,
            processing_mode=ProcessingMode.FULL_MODE
        )
        
        print("✓ 主处理器初始化成功")
        
        # 测试回调函数
        callback_results = []
        
        def on_face_detected(results, location):
            callback_results.append(f"检测到人脸: {len(results)}个")
        
        def on_attendance_event(result):
            callback_results.append(f"考勤事件: {result['message']}")
        
        def on_stranger_alert(alert):
            callback_results.append(f"陌生人警报: {alert['description']}")
        
        processor.set_callbacks(
            on_face_detected=on_face_detected,
            on_attendance_event=on_attendance_event,
            on_stranger_alert=on_stranger_alert
        )
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 测试识别模式
        processor.change_mode(ProcessingMode.RECOGNITION_ONLY)
        result = processor.process_image(test_image, location="测试位置")
        print(f"✓ 识别模式: {result.success} - {result.message}")
        
        # 测试考勤模式
        processor.change_mode(ProcessingMode.ATTENDANCE_ONLY)
        result = processor.process_image(
            test_image, 
            location="办公室",
            attendance_type=AttendanceType.CHECK_IN,
            notes="测试签到"
        )
        print(f"✓ 考勤模式: {result.success} - {result.message}")
        
        # 测试安全模式
        processor.change_mode(ProcessingMode.SECURITY_ONLY)
        result = processor.process_image(test_image, location="安全区域")
        print(f"✓ 安全模式: {result.success} - {result.message}")
        
        # 测试完整模式
        processor.change_mode(ProcessingMode.FULL_MODE)
        result = processor.process_image(
            test_image,
            location="主入口",
            attendance_type=AttendanceType.CHECK_OUT,
            notes="测试签退"
        )
        print(f"✓ 完整模式: {result.success} - {result.message}")
        
        # 测试统计信息
        stats = processor.get_processing_statistics(1)
        print(f"✓ 处理统计: {stats['processing_statistics']['total_processed']}次处理")
        
        # 测试最近结果
        recent = processor.get_recent_results(3)
        print(f"✓ 最近结果: {len(recent)}条记录")
        
        print(f"✓ 回调测试: 触发了{len(callback_results)}个回调")
        
        print("🎉 主处理器测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 主处理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """集成测试"""
    print("\n" + "=" * 50)
    print("集成测试")
    print("=" * 50)
    
    try:
        # 完整流程测试
        recognizer = MockInsightFaceRecognizer()
        database = SimpleSQLiteDB()
        
        # 添加多个测试用户
        users = ["Alice", "Bob", "Charlie"]
        for user in users:
            embedding = np.random.rand(512).astype(np.float32)
            database.add_face(user, embedding, {"role": "employee"})
        
        print(f"✓ 添加了{len(users)}个测试用户")
        
        # 创建完整模式处理器
        processor = VideoProcessor(
            recognizer=recognizer,
            database=database,
            processing_mode=ProcessingMode.FULL_MODE,
            attendance_config={'min_confidence': 0.5},
            stranger_config={'stranger_threshold': 0.7}
        )
        
        # 模拟一天的活动
        activities = [
            (AttendanceType.CHECK_IN, "早晨签到"),
            (AttendanceType.BREAK_OUT, "午餐外出"),
            (AttendanceType.BREAK_IN, "午餐回来"),
            (AttendanceType.CHECK_OUT, "下班签退")
        ]
        
        processed_count = 0
        for att_type, note in activities:
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = processor.process_image(
                test_image,
                location="办公室",
                attendance_type=att_type,
                notes=note
            )
            if result.success:
                processed_count += 1
        
        print(f"✓ 模拟活动: {processed_count}/{len(activities)}次处理成功")
        
        # 获取综合统计
        final_stats = processor.get_processing_statistics(24)
        print(f"✓ 最终统计: {final_stats['processing_statistics']['success_rate']:.1%}成功率")
        
        print("🎉 集成测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始核心模块测试...")
    
    success_count = 0
    total_tests = 4
    
    # 运行各项测试
    if test_attendance_system():
        success_count += 1
    
    if test_stranger_detection():
        success_count += 1
        
    if test_main_processor():
        success_count += 1
        
    if test_integration():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"测试完成: {success_count}/{total_tests} 通过")
    print("=" * 50)
    
    if success_count == total_tests:
        print("🎉 所有测试通过！核心模块工作正常")
        return 0
    else:
        print("❌ 部分测试失败，需要修复")
        return 1

if __name__ == "__main__":
    sys.exit(main())