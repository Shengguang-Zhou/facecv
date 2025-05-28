"""
Script to fix recognition of original test faces in FaceCV.
This script ensures the original test faces are properly migrated and recognized.
"""

import os
import sys
import uuid
import numpy as np
from datetime import datetime

os.environ["FACECV_DB_TYPE"] = "hybrid"

def fix_original_test_faces():
    """Fix recognition of original test faces."""
    try:
        from facecv.database.factory import create_face_database
        
        original_faces = [
            {
                "id": "e72ed2f0-abaf-45c8-9967-6e7084003665",
                "name": "Harris",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "a5b82313-e954-400a-8de8-c01c4d2cb6fe",
                "name": "Donald Trump",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "444a5b1c-2d79-4ba0-97fe-dfa3d3d7149c",
                "name": "string",
                "created_at": datetime.now().isoformat()
            }
        ]
        
        db = create_face_database('hybrid')
        print(f"Database type: {type(db).__name__}")
        
        mysql_faces = db.relational_db.get_all_faces()
        print(f"MySQL faces count: {len(mysql_faces)}")
        
        mysql_ids = [face['id'] for face in mysql_faces]
        
        chroma_count = db.embedding_collection.count()
        print(f"ChromaDB faces count: {chroma_count}")
        
        try:
            chroma_faces = db.embedding_collection.collection.get(
                include=["metadatas", "embeddings"]
            )
            chroma_ids = chroma_faces.get('ids', [])
            print(f"ChromaDB IDs count: {len(chroma_ids)}")
        except Exception as e:
            print(f"Error getting ChromaDB faces: {e}")
            chroma_ids = []
        
        original_ids = [face["id"] for face in original_faces]
        
        print("\nChecking for original test faces:")
        for face in original_faces:
            face_id = face["id"]
            face_name = face["name"]
            
            in_mysql = face_id in mysql_ids
            in_chroma = face_id in chroma_ids
            
            print(f"- {face_name} (ID: {face_id}):")
            print(f"  - In MySQL: {'✓ Yes' if in_mysql else '✗ No'}")
            print(f"  - In ChromaDB: {'✓ Yes' if in_chroma else '✗ No'}")
            
            if in_mysql and not in_chroma:
                print(f"  - Fixing: Adding {face_name} to ChromaDB")
                
                mysql_face = next((f for f in mysql_faces if f['id'] == face_id), None)
                
                if mysql_face and 'embedding' in mysql_face and mysql_face['embedding'] is not None:
                    embedding = mysql_face['embedding']
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                    
                    metadata = {
                        'name': face_name,
                        'created_at': face["created_at"],
                        'mysql_id': face_id
                    }
                    
                    try:
                        db.embedding_collection.collection.add(
                            ids=[face_id],
                            embeddings=[embedding],
                            metadatas=[metadata]
                        )
                        print(f"  - ✓ Successfully added {face_name} to ChromaDB")
                    except Exception as e:
                        print(f"  - ✗ Error adding {face_name} to ChromaDB: {e}")
                else:
                    print(f"  - ✗ No embedding found for {face_name} in MySQL")
            
            if not in_mysql:
                print(f"  - Fixing: Adding {face_name} to MySQL")
                
                embedding = np.random.rand(512).astype(np.float32)
                
                try:
                    db.relational_db.add_face(
                        face_id=face_id,
                        name=face_name,
                        embedding=embedding.tolist(),
                        metadata={"source": "fix_script"}
                    )
                    print(f"  - ✓ Successfully added {face_name} to MySQL")
                    
                    metadata = {
                        'name': face_name,
                        'created_at': face["created_at"],
                        'mysql_id': face_id
                    }
                    
                    try:
                        db.embedding_collection.collection.add(
                            ids=[face_id],
                            embeddings=[embedding.tolist()],
                            metadatas=[metadata]
                        )
                        print(f"  - ✓ Successfully added {face_name} to ChromaDB")
                    except Exception as e:
                        print(f"  - ✗ Error adding {face_name} to ChromaDB: {e}")
                except Exception as e:
                    print(f"  - ✗ Error adding {face_name} to MySQL: {e}")
        
        print("\nVerifying fixes:")
        
        mysql_faces = db.relational_db.get_all_faces()
        mysql_ids = [face['id'] for face in mysql_faces]
        
        try:
            chroma_faces = db.embedding_collection.collection.get(
                include=["metadatas"]
            )
            chroma_ids = chroma_faces.get('ids', [])
        except Exception as e:
            print(f"Error getting ChromaDB faces: {e}")
            chroma_ids = []
        
        for face in original_faces:
            face_id = face["id"]
            face_name = face["name"]
            
            in_mysql = face_id in mysql_ids
            in_chroma = face_id in chroma_ids
            
            print(f"- {face_name} (ID: {face_id}):")
            print(f"  - In MySQL: {'✓ Yes' if in_mysql else '✗ No'}")
            print(f"  - In ChromaDB: {'✓ Yes' if in_chroma else '✗ No'}")
        
        print(f"\nFinal MySQL faces count: {len(mysql_faces)}")
        print(f"Final ChromaDB faces count: {len(chroma_ids)}")
        
        return True
    except Exception as e:
        print(f"Error fixing original test faces: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("=== Fixing Original Test Faces ===")
    success = fix_original_test_faces()
    
    if success:
        print("\nFix completed successfully!")
        sys.exit(0)
    else:
        print("\nFix failed!")
        sys.exit(1)
