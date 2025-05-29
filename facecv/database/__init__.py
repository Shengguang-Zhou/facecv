"""数据库模块"""

from .abstract_facedb import AbstractFaceDB
from .factory import FaceDBFactory, create_face_database, get_default_database, test_database_availability
from .sqlite_facedb import SQLiteFaceDB

# 尝试导入MySQL支持
try:
    from .mysql_facedb import MySQLFaceDB

    __all__ = [
        "AbstractFaceDB",
        "SQLiteFaceDB",
        "MySQLFaceDB",
        "FaceDBFactory",
        "create_face_database",
        "get_default_database",
        "test_database_availability",
    ]
except ImportError:
    __all__ = [
        "AbstractFaceDB",
        "SQLiteFaceDB",
        "FaceDBFactory",
        "create_face_database",
        "get_default_database",
        "test_database_availability",
    ]
