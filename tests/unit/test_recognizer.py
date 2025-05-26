"""测试人脸识别器基本功能"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facecv import FaceRecognizer

def test_basic_functionality():
    """测试基本功能"""
    print("初始化人脸识别器...")
    
    try:
        # 初始化识别器
        recognizer = FaceRecognizer(
            model="insightface",
            db_type="sqlite",
            db_connection="test_facecv.db"
        )
        print("✓ 识别器初始化成功")
        
        # 获取人脸数量
        count = recognizer.get_face_count()
        print(f"✓ 当前数据库中有 {count} 个人脸")
        
        # 测试其他基本功能
        faces = recognizer.list_faces()
        print(f"✓ 列出人脸功能正常，返回 {len(faces)} 个结果")
        
        print("\n所有基本功能测试通过！")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_functionality()