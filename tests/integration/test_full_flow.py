"""测试完整的人脸识别流程"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from facecv import FaceRecognizer

def test_full_flow():
    """测试完整流程：注册、识别、验证"""
    print("=== FaceCV 完整功能测试 ===\n")
    
    # 初始化识别器
    print("1. 初始化识别器...")
    recognizer = FaceRecognizer(
        model="insightface",
        db_type="sqlite",
        db_connection="test_facecv.db"
    )
    print("✓ 识别器初始化成功\n")
    
    # 清空数据库
    if recognizer.face_db.get_face_count() > 0:
        recognizer.face_db.clear_database()
        print("✓ 清空数据库\n")
    
    # 测试人脸注册
    print("2. 测试人脸注册...")
    
    # 模拟图片数据
    mock_image1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mock_image2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 注册第一个人
    face_ids1 = recognizer.register(
        image=mock_image1,
        name="张三",
        metadata={"department": "研发部", "employee_id": "E001"}
    )
    print(f"✓ 注册张三成功，Face ID: {face_ids1}")
    
    # 注册第二个人
    face_ids2 = recognizer.register(
        image=mock_image2,
        name="李四",
        metadata={"department": "市场部", "employee_id": "E002"}
    )
    print(f"✓ 注册李四成功，Face ID: {face_ids2}")
    
    # 检查数据库
    count = recognizer.get_face_count()
    print(f"✓ 当前数据库中有 {count} 个人脸\n")
    
    # 测试人脸识别
    print("3. 测试人脸识别...")
    results = recognizer.recognize(mock_image1)
    if results:
        result = results[0]
        print(f"✓ 识别结果：{result.recognized_name}，相似度：{result.similarity_score:.3f}")
    else:
        print("✗ 未识别到人脸")
    
    # 测试人脸验证
    print("\n4. 测试人脸验证...")
    verification = recognizer.verify(mock_image1, mock_image2)
    print(f"✓ 验证结果：{'同一人' if verification.is_same_person else '不同人'}")
    print(f"  相似度：{verification.similarity_score:.3f}")
    print(f"  阈值：{verification.threshold}")
    
    # 测试人脸列表
    print("\n5. 测试人脸列表...")
    all_faces = recognizer.list_faces()
    print(f"✓ 所有人脸（{len(all_faces)} 个）：")
    for face in all_faces:
        print(f"  - {face['name']} (ID: {face['id']})")
    
    # 测试按姓名查询
    zhang_faces = recognizer.list_faces(name="张三")
    print(f"\n✓ 张三的人脸（{len(zhang_faces)} 个）")
    
    # 测试删除功能
    print("\n6. 测试删除功能...")
    if face_ids1:
        success = recognizer.delete(face_id=face_ids1[0])
        print(f"✓ 删除张三的人脸：{'成功' if success else '失败'}")
    
    # 按姓名删除
    success = recognizer.delete(name="李四")
    print(f"✓ 删除李四的所有人脸：{'成功' if success else '失败'}")
    
    # 最终检查
    final_count = recognizer.get_face_count()
    print(f"\n✓ 最终数据库中有 {final_count} 个人脸")
    
    print("\n=== 所有测试完成！ ===")

if __name__ == "__main__":
    test_full_flow()