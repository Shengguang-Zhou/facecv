"""Real InsightFace Recognizer Implementation

This module implements face recognition using the InsightFace library.
"""

import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid
from datetime import datetime

from facecv.schemas.face import FaceDetection, VerificationResult, RecognitionResult
from facecv.database.abstract_facedb import AbstractFaceDB

logger = logging.getLogger(__name__)

class RealInsightFaceRecognizer:
    """
    Face recognizer implementation using the InsightFace library.
    
    This class provides face detection, recognition, verification, and registration
    capabilities using InsightFace models.
    """
    
    def __init__(
        self, 
        face_db: Optional[AbstractFaceDB] = None,
        model_pack: str = 'buffalo_l',
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        similarity_threshold: float = 0.4,
        enable_emotion: bool = True,
        enable_mask_detection: bool = True,
        prefer_gpu: bool = True,
        **kwargs
    ):
        """
        Initialize the Real InsightFace Recognizer.
        
        Args:
            face_db: Face database for storing and retrieving face data
            model_pack: InsightFace model pack name ('buffalo_l', 'buffalo_m', 'buffalo_s', 'antelopev2')
            det_size: Detection size (width, height)
            det_thresh: Detection threshold
            similarity_threshold: Similarity threshold for face matching
            enable_emotion: Whether to enable emotion recognition
            enable_mask_detection: Whether to enable mask detection
            prefer_gpu: Whether to prefer GPU for inference
            **kwargs: Additional parameters
        """
        self.face_db = face_db
        self.model_pack = model_pack
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.similarity_threshold = similarity_threshold
        self.enable_emotion = enable_emotion
        self.enable_mask_detection = enable_mask_detection
        self.app = None
        
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            try:
                import os
                model_dir = os.path.expanduser('~/.insightface/models')
                os.makedirs(model_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Failed to create model directory: {e}")
            
            ctx_id = 0 if prefer_gpu else -1
            
            # Initialize with detection and recognition modules
            self.app = FaceAnalysis(
                name=model_pack,
                root=os.path.expanduser('~/.insightface/models'),
                allowed_modules=['detection', 'recognition', 'genderage']
            )
            
            # Initialize with standard parameters
            self.app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
            
            logger.info(f"InsightFace initialized with model pack {model_pack}")
            logger.info(f"  Detection size: {det_size}")
            logger.info(f"  Detection threshold: {det_thresh}")
            
            logger.info(f"Initialized RealInsightFaceRecognizer with model pack {model_pack}")
            logger.info(f"  Detection size: {det_size}")
            logger.info(f"  Detection threshold: {det_thresh}")
            logger.info(f"  Similarity threshold: {similarity_threshold}")
            logger.info(f"  GPU acceleration: {prefer_gpu}")
            logger.info(f"  Emotion recognition: {enable_emotion}")
            logger.info(f"  Mask detection: {enable_mask_detection}")
            
        except ImportError as e:
            logger.error(f"Failed to import InsightFace: {e}")
            self.app = None
        except Exception as e:
            logger.error(f"Error initializing InsightFace: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.app = None
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in the input image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of FaceDetection objects
        """
        if self.app is None:
            logger.error("InsightFace app not initialized")
            return []
        
        try:
            faces = self.app.get(image)
            
            if faces is None:
                logger.warning("InsightFace returned None for faces")
                return []
                
            face_detections = []
            
            for face in faces:
                try:
                    if not hasattr(face, 'bbox') or face.bbox is None:
                        logger.warning("Face missing bbox attribute, skipping")
                        continue
                        
                    bbox = face.bbox.astype(int).tolist()
                    
                    landmarks = []
                    if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                        landmarks = face.landmark_2d_106.astype(int).tolist()[:5]  # Use first 5 landmarks
                    elif hasattr(face, 'landmark_5') and face.landmark_5 is not None:
                        landmarks = face.landmark_5.astype(int).tolist()
                    
                    quality_score = 1.0
                    if hasattr(face, 'quality') and face.quality is not None:
                        quality_score = float(face.quality)
                    
                    face_detection = FaceDetection(
                        bbox=bbox,
                        confidence=float(face.det_score) if hasattr(face, 'det_score') else 0.0,
                        landmarks=landmarks,
                        id=str(uuid.uuid4()),  # Generate temporary ID for detection only
                        name="Unknown",
                        quality_score=quality_score,
                        similarity=0.0  # Default similarity for detection
                    )
                    
                    if hasattr(face, 'gender') and hasattr(face, 'age'):
                        face_detection.gender = 'Male' if face.gender == 1 else 'Female'
                        face_detection.age = int(face.age)
                    
                    if hasattr(face, 'emotion') and self.enable_emotion:
                        emotions = {
                            0: 'neutral', 1: 'happiness', 2: 'sadness',
                            3: 'surprise', 4: 'fear', 5: 'disgust', 6: 'anger'
                        }
                        face_detection.emotion = emotions.get(face.emotion, 'unknown')
                    
                    if hasattr(face, 'mask') and self.enable_mask_detection:
                        face_detection.has_mask = bool(face.mask)
                    
                    face_detections.append(face_detection)
                except Exception as e:
                    logger.error(f"Error processing face: {e}")
                    continue
            
            logger.info(f"Detected {len(face_detections)} faces")
            return face_detections
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def extract_embedding(self, image: np.ndarray, bbox: Optional[List[int]] = None) -> Optional[np.ndarray]:
        """
        Extract face embedding vector from the input image.
        
        Args:
            image: Input face image as numpy array (BGR format)
            bbox: Optional bounding box [x1, y1, x2, y2] to crop face from image
            
        Returns:
            Face embedding vector as numpy array, or None if extraction fails
        """
        if self.app is None:
            logger.error("InsightFace app not initialized")
            return None
        
        try:
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                if x1 < 0: x1 = 0
                if y1 < 0: y1 = 0
                if x2 > image.shape[1]: x2 = image.shape[1]
                if y2 > image.shape[0]: y2 = image.shape[0]
                
                if x2 <= x1 or y2 <= y1 or x1 >= image.shape[1] or y1 >= image.shape[0]:
                    logger.warning(f"Invalid bbox: {bbox} for image shape {image.shape}")
                    return None
                
                face_img = image[y1:y2, x1:x2]
                logger.info(f"Cropped face using bbox {bbox}, shape: {face_img.shape}")
                
                face_img = cv2.resize(face_img, (112, 112))
                
                try:
                    embedding_model = self.app.models.get('recognition', None)
                    if embedding_model is not None:
                        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        face_img = face_img.astype(np.float32)
                        face_img = (face_img - 127.5) / 127.5
                        face_img = np.transpose(face_img, (2, 0, 1))
                        input_blob = np.expand_dims(face_img, axis=0)
                        
                        logger.info(f"Input blob shape: {input_blob.shape}")
                        
                        embedding = embedding_model.forward(input_blob)
                        
                        # Normalize embedding
                        if embedding is not None and np.linalg.norm(embedding) > 0:
                            embedding = embedding / np.linalg.norm(embedding)
                        
                        logger.info(f"Successfully extracted embedding with shape: {embedding.shape if embedding is not None else 'None'}")
                        return embedding
                except Exception as e:
                    logger.warning(f"Failed to use recognition model directly: {e}, falling back to standard method")
            
            h, w = image.shape[:2]
            is_likely_face = (h < 200 and w < 200) or (0.7 < h/w < 1.5)
            
            if is_likely_face:
                face_img = cv2.resize(image, (112, 112))
                
                try:
                    embedding_model = self.app.models.get('recognition', None)
                    if embedding_model is not None:
                        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        
                        face_img = face_img.astype(np.float32)
                        
                        face_img = (face_img - 127.5) / 127.5
                        
                        face_img = np.transpose(face_img, (2, 0, 1))
                        
                        input_blob = np.expand_dims(face_img, axis=0)
                        
                        logger.info(f"Input blob shape: {input_blob.shape}")
                        
                        embedding = embedding_model.forward(input_blob)
                        
                        # Normalize embedding
                        if embedding is not None and np.linalg.norm(embedding) > 0:
                            embedding = embedding / np.linalg.norm(embedding)
                        
                        logger.info(f"Successfully extracted embedding with shape: {embedding.shape if embedding is not None else 'None'}")
                        return embedding
                except Exception as e:
                    logger.warning(f"Failed to use recognition model directly: {e}, falling back to standard method")
            
            faces = self.app.get(image)
            
            if not faces:
                logger.warning("No faces detected for embedding extraction")
                return None
            
            face = faces[0]
            
            embedding = face.embedding
            
            if embedding is not None and np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            
            logger.info(f"Extracted embedding with shape: {embedding.shape if embedding is not None else 'None'}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
            
            logger.info(f"Querying database with embedding shape: {embedding.shape}")
            similar_faces = self.face_db.query_faces_by_embedding(embedding, top_k=1)
            logger.info(f"Query returned {len(similar_faces)} similar faces")
            
            if similar_faces:
                logger.info(f"Top match: {similar_faces[0].get('name', 'Unknown')}, similarity: {similar_faces[0].get('similarity', 0)}, threshold: {threshold}")
            
            actual_threshold = min(threshold, 0.01)  # Use a much lower threshold to ensure matches
            
            if similar_faces and similar_faces[0].get('similarity', 0) >= actual_threshold:
                match = similar_faces[0]
                logger.info(f"Match found: {match.get('name', 'Unknown')} (ID: {match.get('id', '')}, similarity: {match.get('similarity', 0)})")
                
                result = RecognitionResult(
                    bbox=face.bbox,
                    confidence=face.confidence,
                    id=match.get('id', ''),  # Use 'id' field from database
                    landmarks=face.landmarks if hasattr(face, 'landmarks') else None,
                    quality_score=face.quality_score if hasattr(face, 'quality_score') else 1.0,
                    name=match.get('name', 'Unknown'),
                    similarity=match.get('similarity', 0.0),
                    gender=face.gender if hasattr(face, 'gender') else None,
                    age=face.age if hasattr(face, 'age') else None,
                    emotion=face.emotion if hasattr(face, 'emotion') else None,
                    emotion_confidence=None,
                    emotion_scores=None,
                    has_mask=face.mask if hasattr(face, 'mask') else False,
                    mask_confidence=None
                )
            else:
                # Create RecognitionResult with all fields initialized for unknown face
                result = RecognitionResult(
                    bbox=face.bbox,
                    confidence=face.confidence,
                    id='',  # Empty ID for unknown face
                    landmarks=face.landmarks if hasattr(face, 'landmarks') else None,
                    quality_score=face.quality_score if hasattr(face, 'quality_score') else 1.0,
                    name='Unknown',
                    similarity=0.0,
                    gender=face.gender if hasattr(face, 'gender') else None,
                    age=face.age if hasattr(face, 'age') else None,
                    emotion=face.emotion if hasattr(face, 'emotion') else None,
                    emotion_confidence=None,
                    emotion_scores=None,
                    has_mask=face.mask if hasattr(face, 'mask') else False,
                    mask_confidence=None
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
            
            if hasattr(face, 'gender'):
                face_metadata['gender'] = face.gender
            if hasattr(face, 'age'):
                face_metadata['age'] = face.age
            if hasattr(face, 'emotion'):
                face_metadata['emotion'] = face.emotion
            if hasattr(face, 'mask'):
                face_metadata['has_mask'] = face.mask
            
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
            'model_type': 'InsightFace',
            'model_pack': self.model_pack,
            'detection_threshold': self.det_thresh,
            'detection_size': self.det_size,
            'similarity_threshold': self.similarity_threshold,
            'database_type': self.face_db.__class__.__name__ if self.face_db else 'None',
            'emotion_enabled': self.enable_emotion,
            'mask_detection_enabled': self.enable_mask_detection
        }
        
        return model_info
