#!/usr/bin/env python
"""Run migration script to transfer data from MySQL to ChromaDB"""

import logging
import numpy as np
from datetime import datetime
import os
import sys

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("migration")

from facecv.database.factory import create_face_database
from facecv.config.database import DatabaseConfig

print("Starting migration from MySQL to ChromaDB...")

try:
    db = create_face_database('hybrid')
    print(f"Created hybrid database instance: {type(db).__name__}")
    
    mysql_faces = db.relational_db.get_all_faces()
    print(f"Found {len(mysql_faces)} faces in MySQL database")

    success_count = 0
    error_count = 0

    for face in mysql_faces:
        if 'embedding' in face and face['embedding'] is not None:
            embedding = face['embedding']
            embedding_list = []
            
            if isinstance(embedding, np.ndarray):
                embedding_list = embedding.tolist()
            elif isinstance(embedding, list):
                embedding_list = embedding
            else:
                try:
                    embedding_list = list(embedding)
                except Exception as e:
                    print(f"Could not convert embedding to list for face {face.get('id', 'unknown')}: {e}")
                    error_count += 1
                    continue
            
            try:
                embedding_list = [float(x) for x in embedding_list]
            except Exception as e:
                print(f"Could not convert embedding values to float for face {face.get('id', 'unknown')}: {e}")
                error_count += 1
                continue

            created_at = face.get('created_at', '')
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()

            metadata = {
                'name': str(face.get('name', 'Unknown')),
                'created_at': str(created_at),
                'mysql_id': str(face.get('id', ''))
            }
            
            for key, value in face.items():
                if key not in ['id', 'name', 'embedding', 'created_at'] and value is not None:
                    metadata[key] = str(value)

            try:
                db.embedding_collection.add_face(
                    face.get('name', 'Unknown'),
                    embedding_list,
                    metadata
                )
                success_count += 1
                print(f"Migrated: {face.get('name', 'Unknown')} (ID: {face.get('id', 'unknown')})")
            except Exception as e:
                print(f"Error migrating {face.get('name', 'Unknown')}: {e}")
                error_count += 1
        else:
            print(f"Skipping face {face.get('id', 'unknown')}: No embedding found")

    print(f"Migration completed: {success_count} faces migrated successfully, {error_count} errors")
    
    try:
        chroma_count = db.embedding_collection.get_face_count()
        mysql_count = len(mysql_faces)
        print(f"ChromaDB count: {chroma_count}")
        print(f"MySQL count: {mysql_count}")
        
        if chroma_count >= success_count:
            print("Migration verification successful")
        else:
            print("Migration verification failed: ChromaDB count doesn't match migrated count")
    except Exception as e:
        print(f"Failed to verify migration: {e}")
    
except Exception as e:
    print(f"Migration failed: {e}")
