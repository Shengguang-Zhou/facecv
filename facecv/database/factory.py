"""数据库工厂类"""

import logging
from typing import Optional, Type
from .abstract_facedb import AbstractFaceDB
from .sqlite_facedb import SQLiteFaceDB
from ..config import get_db_config

from .hybrid_facedb import HybridFaceDB

db_config = get_db_config()

# 延迟导入以避免依赖问题
_mysql_facedb = None
_chromadb_facedb = None
_hybrid_facedb = None

logging.basicConfig(level=logging.INFO)


def get_mysql_facedb():
    """延迟导入MySQL数据库类"""
    global _mysql_facedb
    if _mysql_facedb is None:
        try:
            from .mysql_facedb import MySQLFaceDB
            _mysql_facedb = MySQLFaceDB
        except ImportError as e:
            logging.warning(f"MySQL数据库不可用: {e}")
            _mysql_facedb = False
    return _mysql_facedb if _mysql_facedb is not False else None


def get_chromadb_facedb():
    """延迟导入ChromaDB数据库类"""
    global _chromadb_facedb
    if _chromadb_facedb is None:
        try:
            from .chroma_facedb import ChromaFaceDB, CHROMADB_AVAILABLE
            if CHROMADB_AVAILABLE:
                _chromadb_facedb = ChromaFaceDB
                logging.info("ChromaDB is available and ready to use")
            else:
                logging.error("ChromaDB not installed. Install with: pip install chromadb")
                _chromadb_facedb = False
        except ImportError as e:
            logging.error(f"ChromaDB数据库不可用: {e}")
            _chromadb_facedb = False
    return _chromadb_facedb if _chromadb_facedb is not False else None


def get_hybrid_facedb():
    """延迟导入Hybrid数据库类"""
    global _hybrid_facedb
    if _hybrid_facedb is None:
        try:
            from .hybrid_facedb import HybridFaceDB
            _hybrid_facedb = HybridFaceDB
            logging.info("Hybrid database is available and ready to use")
        except ImportError as e:
            logging.error(f"Hybrid数据库不可用: {e}")
            _hybrid_facedb = False
    return _hybrid_facedb if _hybrid_facedb is not False else None


class FaceDBFactory:
    """人脸数据库工厂类"""
    
    _db_classes = {
        'sqlite': SQLiteFaceDB,
        'mysql': get_mysql_facedb,
        'chromadb': get_chromadb_facedb,
        'hybrid': get_hybrid_facedb,
    }
    
    @classmethod
    def create_database(
        self, 
        db_type: Optional[str] = None, 
        config=None, 
        **kwargs
    ) -> AbstractFaceDB:
        """
        创建数据库实例
        
        Args:
            db_type: 数据库类型 ('sqlite', 'mysql', 'chromadb')
            config: 数据库配置对象
            **kwargs: 额外的数据库参数
            
        Returns:
            数据库实例
            
        Raises:
            ValueError: 不支持的数据库类型
            ImportError: 缺少必要的依赖
        """
        if db_type is None:
            db_type = (config or db_config).db_type
        
        if db_type not in self._db_classes:
            available_types = list(self._db_classes.keys())
            raise ValueError(f"不支持的数据库类型: {db_type}。支持的类型: {available_types}")
        
        db_class = self._db_classes[db_type]
        
        # 处理延迟导入的情况
        if callable(db_class) and not isinstance(db_class, type):
            db_class = db_class()
            if db_class is None:
                raise ImportError(f"无法导入 {db_type} 数据库驱动")
        
        try:
            if db_type == 'sqlite':
                # SQLite只需要路径参数
                db_path = kwargs.get('db_path') or (config or db_config).get_sqlite_path()
                return db_class(db_path=db_path)
            
            elif db_type == 'mysql':
                # MySQL需要配置对象
                return db_class(config=config or db_config)
            
            elif db_type == 'chromadb':
                # ChromaDB需要持久化目录（可选）和集合名
                persist_directory = kwargs.get('persist_directory')
                collection_name = kwargs.get('collection_name', 'face_embeddings')
                return db_class(persist_directory=persist_directory, collection_name=collection_name)
            
            elif db_type == 'hybrid':
                return db_class(config=config or db_config)
            
            
            else:
                # 通用情况，传递所有参数
                return db_class(config=config or db_config, **kwargs)
                
        except Exception as e:
            logging.error(f"创建 {db_type} 数据库实例失败: {e}")
            raise
    
    @classmethod
    def get_available_databases(cls) -> dict:
        """
        获取可用的数据库类型
        
        Returns:
            可用数据库类型及其可用状态
        """
        available = {}
        
        for db_type, db_class in cls._db_classes.items():
            try:
                if callable(db_class) and not isinstance(db_class, type):
                    # 延迟导入的情况
                    imported_class = db_class()
                    available[db_type] = imported_class is not None
                else:
                    # 直接可用的类
                    available[db_type] = True
            except Exception:
                available[db_type] = False
        
        return available
    
    @classmethod
    def register_database(cls, db_type: str, db_class: Type[AbstractFaceDB]):
        """
        注册新的数据库类型
        
        Args:
            db_type: 数据库类型名称
            db_class: 数据库类
        """
        if not issubclass(db_class, AbstractFaceDB):
            raise ValueError(f"数据库类必须继承自 AbstractFaceDB")
        
        cls._db_classes[db_type] = db_class
        logging.info(f"注册数据库类型: {db_type}")


# 便捷函数
def create_face_database(db_type: Optional[str] = None, connection_string: Optional[str] = None, **kwargs) -> AbstractFaceDB:
    """
    创建人脸数据库实例的便捷函数
    
    Args:
        db_type: 数据库类型 ('sqlite', 'mysql', 'chromadb')
        connection_string: 连接字符串，用于简化配置
        **kwargs: 额外参数
        
    Examples:
        # SQLite
        db = create_face_database('sqlite', 'sqlite:///path/to/db.sqlite')
        
        # MySQL  
        db = create_face_database('mysql', 'mysql://user:pass@host:port/db')
        
        # ChromaDB
        db = create_face_database('chromadb', 'chromadb:///path/to/persist')
        db = create_face_database('chromadb')  # In-memory
    """
    if connection_string:
        if connection_string.startswith('sqlite://'):
            db_type = 'sqlite'
            kwargs['db_path'] = connection_string
        elif connection_string.startswith('mysql://'):
            db_type = 'mysql'
            # MySQL factory will handle the connection string
        elif connection_string.startswith('chromadb://'):
            db_type = 'chromadb'
            path = connection_string.replace('chromadb://', '')
            kwargs['persist_directory'] = path if path else None
    
    return FaceDBFactory.create_database(db_type=db_type, **kwargs)


def get_default_database() -> AbstractFaceDB:
    """获取默认配置的数据库实例"""
    return FaceDBFactory.create_database()


# 测试可用性
def test_database_availability():
    """测试各种数据库的可用性"""
    available = FaceDBFactory.get_available_databases()
    logging.info("数据库可用性测试结果:")
    for db_type, is_available in available.items():
        status = "✓ 可用" if is_available else "✗ 不可用"
        logging.info(f"  {db_type}: {status}")
    return available
