"""ArcFace Recognizer Implementation

This module implements face recognition using ArcFace models.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from facecv.database.abstract_facedb import AbstractFaceDB
from facecv.schemas.face import FaceDetection, RecognitionResult, VerificationResult

logger = logging.getLogger(__name__)

class ArcFaceRecognizer:
    """
    Face recognizer implementation using ArcFace models.
    
    This class provides face detection, recognition, verification, and registration
    capabilities using ArcFace models.
    """
    
    def __init__(
        self, 
        face_db: Optional[AbstractFaceDB] = None,
        model_name: str = 'buffalo_l_resnet50',
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        similarity_threshold: float = 0.4,
        enable_detection: bool = True,
        **kwargs
    ):
        """
        Initialize the ArcFace Recognizer.
        
        Args:
            face_db: Face database for storing and retrieving face data
            model_name: ArcFace model name ('buffalo_l_resnet50', 'buffalo_s_mobilefacenet')
            det_size: Detection size (width, height)
            det_thresh: Detection threshold
            similarity_threshold: Similarity threshold for face matching
            enable_detection: Whether to enable face detection
            **kwargs: Additional parameters
        """
        self.face_db = face_db
        self.model_name = model_name
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.similarity_threshold = similarity_threshold
        self.enable_detection = enable_detection
        
        try:
            from insightface.app import FaceAnalysis
            
            if 'mobilefacenet' in model_name:
                backbone = 'mobilefacenet'
                model_pack = 'buffalo_s'
            else:
                backbone = 'resnet50'
                model_pack = 'buffalo_l'
            
            prefer_gpu = kwargs.get('prefer_gpu', True)
            ctx_id = 0 if prefer_gpu else -1
            
            self.app = FaceAnalysis(
                name=model_pack,
                allowed_modules=['detection', 'recognition']
            )
            
            self.app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
            
            logger.info(f"Initialized ArcFaceRecognizer with model {model_name}")
            logger.info(f"  Backbone: {backbone}")
            logger.info(f"  Detection size: {det_size}")
            logger.info(f"  Detection threshold: {det_thresh}")
            logger.info(f"  Similarity threshold: {similarity_threshold}")
            logger.info(f"  GPU acceleration: {prefer_gpu}")
            
        except ImportError as e:
            logger.error(f"Failed to import InsightFace: {e}")
            self.app = None
        except Exception as e:
            logger.error(f"Error initializing ArcFace: {e}")
            self.app = None
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in the input image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of FaceDetection objects
        """
        if not self.enable_detection:
            logger.warning("Face detection is disabled")
            return []
        
        if self.app is None:
            logger.error("ArcFace app not initialized")
            return []
        
        try:
            faces = self.app.get(image)
            
            face_detections = []
            
            for face in faces:
                bbox = face.bbox.astype(int).tolist()
                landmarks = face.landmark_2d_106.astype(int).tolist() if hasattr(face, 'landmark_2d_106') else []
                
                quality_score = float(face.quality) if hasattr(face, 'quality') else 1.0
                
                face_detection = FaceDetection(
                    bbox=bbox,
                    confidence=float(face.det_score),
                    landmarks=landmarks[:5] if landmarks else [],  # Use first 5 landmarks
                    id=str(uuid.uuid4()),  # Generate temporary ID
                    name="Unknown",
                    quality_score=quality_score
                )
                
                face_detections.append(face_detection)
            
            logger.info(f"Detected {len(face_detections)} faces")
            return face_detections
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding vector from the input image.
        
        Args:
            image: Input face image as numpy array (BGR format)
            
        Returns:
            Face embedding vector as numpy array, or None if extraction fails
        """
        if self.app is None:
            logger.error("ArcFace app not initialized")
            return None
        
        try:
            faces = self.app.get(image)
            
            if not faces:
                logger.warning("No faces detected for embedding extraction")
                return None
            
            face = faces[0]
            
            embedding = face.embedding
            
            if embedding is not None and np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
    
    def recognize(self, image: np.ndarray, threshold: float = 0.4) -> List[RecognitionResult]:
        """
        Recognize faces in the input image against the database.
        
        Args:
            image: Input image as numpy array (BGR format)
            threshold: Similarity threshold for face matching
            
        Returns:
            List of RecognitionResult objects
        """
        if self.face_db is None:
            logger.error("No face database provided for recognition")
            return []
        
        face_detections = self.detect_faces(image)
        if not face_detections:
            logger.info("No faces detected for recognition")
            return []
        
        recognition_results = []
        
        for face in face_detections:
            x1, y1, x2, y2 = face.bbox
            face_img = image[y1:y2, x1:x2]
            
            embedding = self.extract_embedding(face_img)
            if embedding is None:
                logger.warning(f"Failed to extract embedding for face at {face.bbox}")
                continue
            
            similar_faces = self.face_db.query_faces_by_embedding(embedding, top_k=1)
            
            if similar_faces and similar_faces[0].get('similarity', 0) >= threshold:
                match = similar_faces[0]
                
                result = RecognitionResult(
                    bbox=face.bbox,
                    confidence=face.confidence,
                    id=match.get('id', ''),  # Use 'id' field from database
                    landmarks=face.landmarks,
                    quality_score=face.quality_score,
                    name=match.get('name', 'Unknown'),
                    similarity=match.get('similarity', 0.0)
                )
            else:
                result = RecognitionResult(
                    bbox=face.bbox,
                    confidence=face.confidence,
                    id='',  # Empty ID for unknown face
                    landmarks=face.landmarks,
                    quality_score=face.quality_score,
                    name='Unknown',
                    similarity=0.0
                )
            
            recognition_results.append(result)
        
        logger.info(f"Recognized {len(recognition_results)} faces")
        return recognition_results
    
    def verify(self, image1: np.ndarray, image2: np.ndarray, threshold: float = 0.4) -> VerificationResult:
        """
        Verify if two face images belong to the same person.
        
        Args:
            image1: First input image as numpy array (BGR format)
            image2: Second input image as numpy array (BGR format)
            threshold: Similarity threshold for face matching
            
        Returns:
            VerificationResult object
        """
        faces1 = self.detect_faces(image1)
        faces2 = self.detect_faces(image2)
        
        if not faces1 or not faces2:
            logger.warning("No faces detected in one or both images")
            return VerificationResult(
                is_same_person=False,
                confidence=0.0,
                face1_bbox=[0, 0, 0, 0],
                face2_bbox=[0, 0, 0, 0]
            )
        
        face1 = faces1[0]
        face2 = faces2[0]
        
        x1_1, y1_1, x2_1, y2_1 = face1.bbox
        face1_img = image1[y1_1:y2_1, x1_1:x2_1]
        
        x1_2, y1_2, x2_2, y2_2 = face2.bbox
        face2_img = image2[y1_2:y2_2, x1_2:x2_2]
        
        embedding1 = self.extract_embedding(face1_img)
        embedding2 = self.extract_embedding(face2_img)
        
        if embedding1 is None or embedding2 is None:
            logger.warning("Failed to extract embeddings for verification")
            return VerificationResult(
                is_same_person=False,
                confidence=0.0,
                face1_bbox=face1.bbox,
                face2_bbox=face2.bbox
            )
        
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        result = VerificationResult(
            is_same_person=similarity >= threshold,
            confidence=float(similarity),
            face1_bbox=face1.bbox,
            face2_bbox=face2.bbox
        )
        
        logger.info(f"Verification result: {result.is_same_person} (confidence: {result.confidence:.3f})")
        return result
    
    def register(self, image: np.ndarray, name: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Register face(s) in the database.
        
        Args:
            image: Input image as numpy array (BGR format)
            name: Name of the person
            metadata: Additional metadata
            
        Returns:
            List of registered face IDs
        """
        if self.face_db is None:
            logger.error("No face database provided for registration")
            return []
        
        if metadata is None:
            metadata = {}
        
        face_detections = self.detect_faces(image)
        if not face_detections:
            logger.warning("No faces detected for registration")
            return []
        
        face_ids = []
        
        for face in face_detections:
            x1, y1, x2, y2 = face.bbox
            face_img = image[y1:y2, x1:x2]
            
            embedding = self.extract_embedding(face_img)
            if embedding is None:
                logger.warning(f"Failed to extract embedding for face at {face.bbox}")
                continue
            
            face_metadata = metadata.copy()
            face_metadata.update({
                'detection_score': float(face.confidence) if face.confidence is not None else 0.0,
                'quality_score': float(face.quality_score) if face.quality_score is not None else 0.0,
                'bbox': face.bbox,
                'created_at': datetime.now().isoformat()
            })
            
            try:
                face_id = self.face_db.add_face(name, embedding, face_metadata)
                face_ids.append(face_id)
                logger.info(f"Registered face with ID {face_id} for {name}")
            except Exception as e:
                logger.error(f"Error registering face: {e}")
        
        return face_ids
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        model_info = {
            'model_type': 'ArcFace',
            'model_name': self.model_name,
            'backbone': 'mobilefacenet' if 'mobilefacenet' in self.model_name else 'resnet50',
            'detection_threshold': self.det_thresh,
            'detection_size': self.det_size,
            'similarity_threshold': self.similarity_threshold,
            'database_type': self.face_db.__class__.__name__ if self.face_db else 'None',
            'embedding_size': 512
        }
        
        return model_info
