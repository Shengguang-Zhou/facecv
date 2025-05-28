"""
Script to migrate faces from MySQL to ChromaDB.
This script ensures all faces in MySQL have corresponding embeddings in ChromaDB.
"""

import os
import sys
import numpy as np
from datetime import datetime

os.environ["FACECV_DB_TYPE"] = "hybrid"

def migrate_faces_to_chromadb():
    """Migrate faces from MySQL to ChromaDB."""
    try:
        from facecv.database.factory import create_face_database
        
        db = create_face_database('hybrid')
        print(f"Database type: {type(db).__name__}")
        
        mysql_faces = db.relational_db.get_all_faces()
        print(f"MySQL faces count: {len(mysql_faces)}")
        
        chroma_count = db.embedding_collection.count()
        print(f"ChromaDB faces count: {chroma_count}")
        
        migrated_count = 0
        error_count = 0
        
        for face in mysql_faces:
            if 'embedding' in face and face['embedding'] is not None:
                embedding = face['embedding']
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                created_at = face.get('created_at', '')
                if isinstance(created_at, datetime):
                    created_at = created_at.isoformat()
                
                metadata = {
                    'name': face['name'],
                    'created_at': str(created_at),
                    'mysql_id': face['id']
                }
                
                try:
                    try:
                        existing = db.embedding_collection.collection.get(
                            ids=[face['id']],
                            include=["metadatas"]
                        )
                        
                        if existing and existing['ids'] and len(existing['ids']) > 0:
                            print(f"Face already exists in ChromaDB: {face['name']} (ID: {face['id']})")
                            continue
                    except Exception:
                        pass
                    
                    db.embedding_collection.add(
                        ids=[face['id']],
                        embeddings=[embedding],
                        metadatas=[metadata]
                    )
                    print(f"✓ Migrated: {face['name']} (ID: {face['id']})")
                    migrated_count += 1
                except Exception as e:
                    print(f"✗ Error migrating {face['name']} (ID: {face['id']}): {e}")
                    error_count += 1
            else:
                print(f"✗ No embedding for {face['name']} (ID: {face['id']})")
                error_count += 1
        
        new_chroma_count = db.embedding_collection.count()
        print(f"\nMigration summary:")
        print(f"- MySQL faces: {len(mysql_faces)}")
        print(f"- ChromaDB faces before: {chroma_count}")
        print(f"- ChromaDB faces after: {new_chroma_count}")
        print(f"- Successfully migrated: {migrated_count}")
        print(f"- Errors: {error_count}")
        
        return True
    except Exception as e:
        print(f"Error during migration: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("=== Migrating Faces from MySQL to ChromaDB ===")
    success = migrate_faces_to_chromadb()
    
    if success:
        print("\nMigration completed successfully!")
        sys.exit(0)
    else:
        print("\nMigration failed!")
        sys.exit(1)
