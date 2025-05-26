#!/usr/bin/env python3
"""简化的数据库测试类"""

import sqlite3
import json
import uuid
from typing import Dict, List, Optional, Any
import numpy as np


class SimpleSQLiteDB:
    """简化的SQLite测试数据库"""
    
    def __init__(self):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表"""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP,
                is_temporary INTEGER DEFAULT 0
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON faces(name)")
        self.conn.commit()
    
    def add_face(self, name: str, embedding: np.ndarray, metadata: Optional[Dict] = None) -> str:
        """添加人脸到数据库"""
        face_id = str(uuid.uuid4())
        embedding_bytes = embedding.tobytes()
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO faces (id, name, embedding, metadata)
            VALUES (?, ?, ?, ?)
        """, (face_id, name, embedding_bytes, metadata_json))
        self.conn.commit()
        return face_id
    
    def query_faces_by_name(self, name: str) -> List[Dict[str, Any]]:
        """根据姓名查询人脸"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, name, embedding, metadata, created_at, updated_at
            FROM faces WHERE name = ?
        """, (name,))
        
        results = []
        for row in cursor.fetchall():
            face_dict = {
                'id': row[0],
                'name': row[1],
                'embedding': np.frombuffer(row[2], dtype=np.float32),
                'metadata': json.loads(row[3]) if row[3] else None,
                'created_at': row[4],
                'updated_at': row[5]
            }
            results.append(face_dict)
        return results
    
    def get_face_count(self) -> int:
        """获取数据库中的人脸总数"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM faces")
        return cursor.fetchone()[0]
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()