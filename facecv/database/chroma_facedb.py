"""ChromaDB implementation for face database - Vector database support for DeepFace"""

import uuid
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

from facecv.database.abstract_facedb import AbstractFaceDB

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not installed. Install with: pip install chromadb")


class ChromaFaceDB(AbstractFaceDB):
    """ChromaDB implementation for face embeddings storage"""
    
    def __init__(self, persist_directory: Optional[str] = None, collection_name: str = "face_embeddings"):
        """
        Initialize ChromaDB face database
        
        Args:
            persist_directory: Directory to persist the database (None for in-memory)
            collection_name: Name of the collection to use
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory
            )
            logger.info(f"ChromaDB initialized with persistent storage at: {persist_directory}")
        else:
            self.client = chromadb.EphemeralClient()
            logger.info("ChromaDB initialized with in-memory storage")
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def _generate_id(self) -> str:
        """Generate unique ID for face"""
        return f"face_{uuid.uuid4().hex}"
    
    def _prepare_metadata(self, name: str, metadata: Optional[Dict] = None) -> Dict:
        """Prepare metadata for storage"""
        meta = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        if metadata:
            # ChromaDB requires all metadata values to be strings, ints, floats, or bools
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    meta[key] = str(value)
                else:
                    meta[key] = json.dumps(value)
        
        return meta
    
    def add_face(self, name: str, embedding: List[float], metadata: Optional[Dict] = None) -> str:
        """Add a face to the database"""
        face_id = self._generate_id()
        
        # Prepare metadata
        meta = self._prepare_metadata(name, metadata)
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=[embedding],
            metadatas=[meta],
            ids=[face_id]
        )
        
        logger.info(f"Added face {face_id} for {name}")
        return face_id
    
    def get_face(self, face_id: str) -> Optional[Dict]:
        """Get a face by ID"""
        try:
            result = self.collection.get(
                ids=[face_id],
                include=["embeddings", "metadatas"]
            )
            
            if result["ids"]:
                metadata = result["metadatas"][0]
                return {
                    "id": face_id,
                    "name": metadata.get("name", "Unknown"),
                    "embedding": result["embeddings"][0],
                    "metadata": metadata,
                    "created_at": metadata.get("created_at"),
                    "updated_at": metadata.get("updated_at")
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting face {face_id}: {e}")
            return None
    
    def get_faces_by_name(self, name: str) -> List[Dict]:
        """Get all faces for a person"""
        try:
            result = self.collection.get(
                where={"name": name},
                include=["embeddings", "metadatas"]
            )
            
            faces = []
            for i, face_id in enumerate(result["ids"]):
                metadata = result["metadatas"][i]
                faces.append({
                    "id": face_id,
                    "name": name,
                    "embedding": result["embeddings"][i],
                    "metadata": metadata,
                    "created_at": metadata.get("created_at"),
                    "updated_at": metadata.get("updated_at")
                })
            
            return faces
            
        except Exception as e:
            logger.error(f"Error getting faces for {name}: {e}")
            return []
    
    def search_similar_faces(self, embedding: List[float], threshold: float = 0.6, limit: int = 10) -> List[Tuple[Dict, float]]:
        """Search for similar faces using vector similarity"""
        try:
            # ChromaDB returns distances (lower is better for cosine)
            # Convert threshold to distance: distance = 1 - similarity
            max_distance = 1 - threshold
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=limit,
                include=["embeddings", "metadatas", "distances"]
            )
            
            similar_faces = []
            if results["ids"]:
                for i, face_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    
                    # Filter by threshold
                    if distance <= max_distance:
                        similarity = 1 - distance
                        metadata = results["metadatas"][0][i]
                        
                        face_data = {
                            "id": face_id,
                            "name": metadata.get("name", "Unknown"),
                            "embedding": results["embeddings"][0][i],
                            "metadata": metadata,
                            "created_at": metadata.get("created_at"),
                            "updated_at": metadata.get("updated_at")
                        }
                        
                        similar_faces.append((face_data, similarity))
            
            return similar_faces
            
        except Exception as e:
            logger.error(f"Error searching similar faces: {e}")
            return []
    
    def update_face(self, face_id: str, new_name: str, metadata: Optional[Dict] = None) -> bool:
        """Update a face in the database with new name and metadata"""
        try:
            # Get existing face
            existing = self.get_face(face_id)
            if not existing:
                return False
            
            # Prepare update
            update_metadata = existing["metadata"].copy()
            update_metadata["name"] = new_name
            update_metadata["updated_at"] = datetime.now().isoformat()
            
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        update_metadata[key] = value
                    else:
                        update_metadata[key] = json.dumps(value)
            
            # Update in ChromaDB (metadata only, keep same embedding)
            self.collection.update(
                ids=[face_id],
                metadatas=[update_metadata]
            )
            
            logger.info(f"Updated face {face_id} with new name: {new_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating face {face_id}: {e}")
            return False
    
    def delete_face(self, face_id: str) -> bool:
        """Delete a face from the database"""
        try:
            self.collection.delete(ids=[face_id])
            logger.info(f"Deleted face {face_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting face {face_id}: {e}")
            return False
    
    def delete_face_by_name(self, name: str) -> int:
        """Delete all faces for a person"""
        try:
            # Get all faces for the person
            faces = self.get_faces_by_name(name)
            
            if faces:
                face_ids = [face["id"] for face in faces]
                self.collection.delete(ids=face_ids)
                logger.info(f"Deleted {len(face_ids)} faces for {name}")
                return len(face_ids)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error deleting faces for {name}: {e}")
            return 0
    
    def get_all_faces(self, limit: int = 1000) -> List[Dict]:
        """Get all faces in the database"""
        try:
            result = self.collection.get(
                limit=limit,
                include=["embeddings", "metadatas"]
            )
            
            faces = []
            for i, face_id in enumerate(result["ids"]):
                metadata = result["metadatas"][i]
                faces.append({
                    "id": face_id,
                    "name": metadata.get("name", "Unknown"),
                    "embedding": result["embeddings"][i],
                    "metadata": metadata,
                    "created_at": metadata.get("created_at"),
                    "updated_at": metadata.get("updated_at")
                })
            
            return faces
            
        except Exception as e:
            logger.error(f"Error getting all faces: {e}")
            return []
    
    def get_face_count(self) -> int:
        """Get total number of faces"""
        try:
            # ChromaDB doesn't have a direct count method, so we get all IDs
            result = self.collection.get(include=[])
            return len(result["ids"])
            
        except Exception as e:
            logger.error(f"Error getting face count: {e}")
            return 0
    
    def clear_database(self) -> bool:
        """Clear all faces from the database"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Cleared ChromaDB database")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            face_count = self.get_face_count()
            
            # Get unique names
            all_faces = self.get_all_faces()
            unique_names = set(face["name"] for face in all_faces)
            
            # Calculate average embeddings per person
            name_counts = {}
            for face in all_faces:
                name = face["name"]
                name_counts[name] = name_counts.get(name, 0) + 1
            
            avg_per_person = sum(name_counts.values()) / len(name_counts) if name_counts else 0
            
            return {
                "total_faces": face_count,
                "unique_persons": len(unique_names),
                "average_faces_per_person": avg_per_person,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "backend": "ChromaDB"
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def optimize_index(self) -> bool:
        """Optimize the vector index for better search performance"""
        try:
            # ChromaDB automatically optimizes its HNSW index
            # We can trigger a persist to ensure data is saved
            if self.persist_directory:
                self.client.persist()
            
            logger.info("ChromaDB index optimized")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing index: {e}")
            return False
    
    def backup(self, backup_path: str) -> bool:
        """Backup the database"""
        try:
            if not self.persist_directory:
                logger.warning("Cannot backup in-memory database")
                return False
            
            # For persistent ChromaDB, we can copy the directory
            import shutil
            shutil.copytree(self.persist_directory, backup_path)
            
            logger.info(f"Database backed up to: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            return False
    
    def restore(self, backup_path: str) -> bool:
        """Restore the database from backup"""
        try:
            if not self.persist_directory:
                logger.warning("Cannot restore to in-memory database")
                return False
            
            # Copy backup to persist directory
            import shutil
            shutil.rmtree(self.persist_directory, ignore_errors=True)
            shutil.copytree(backup_path, self.persist_directory)
            
            # Close and reinitialize client to force reload from disk
            if hasattr(self, 'client'):
                del self.client
                
            # Create new client pointing to restored directory
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get the restored collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to get collection after restore: {e}")
                # If collection doesn't exist, create it (shouldn't happen with proper backup)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.warning(f"Created new collection: {self.collection_name}")
            
            logger.info(f"Database restored from: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring database: {e}")
            return False
    
    # Abstract method implementations
    def delete_face_by_id(self, face_id: str) -> bool:
        """Delete face by ID - alias for delete_face"""
        return self.delete_face(face_id)
    
    def get_all_faces_for_recognition(self) -> List[Dict[str, Any]]:
        """Get all faces for recognition"""
        return self.get_all_faces()
    
    def get_face_by_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Get face by ID - alias for get_face"""
        return self.get_face(face_id)
    
    def query_faces_by_embedding(self, embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Query faces by embedding vector"""
        # Convert embedding to list format for ChromaDB
        embedding_list = []
        if isinstance(embedding, np.ndarray):
            embedding_list = embedding.tolist()
        elif isinstance(embedding, list):
            embedding_list = embedding
        else:
            logger.warning(f"Unexpected embedding type: {type(embedding)}")
            try:
                embedding_list = list(embedding)
            except Exception as e:
                logger.error(f"Failed to convert embedding to list: {e}")
                return []
        
        try:
            embedding_list = [float(x) for x in embedding_list]
            # Use existing search method but return in expected format
            similar_faces = self.search_similar_faces(embedding_list, threshold=0.0, limit=top_k)
        except Exception as e:
            logger.error(f"Error querying faces by embedding: {e}")
            return []
        
        # Convert to expected format
        results = []
        for face_dict, similarity in similar_faces:
            face_copy = face_dict.copy()
            face_copy['similarity_score'] = similarity
            results.append(face_copy)
        
        return results
    
    def query_faces_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Query faces by name - alias for get_faces_by_name"""
        return self.get_faces_by_name(name)


