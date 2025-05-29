"""ONNX Face Recognizer Implementation

This module implements face recognition using ONNX Runtime for inference.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from facecv.database.abstract_facedb import AbstractFaceDB
from facecv.schemas.face import FaceDetection, RecognitionResult, VerificationResult

logger = logging.getLogger(__name__)

class ONNXFaceRecognizer:
    """
    Face recognizer implementation using ONNX Runtime for inference.
    
    This class provides face detection, recognition, verification, and registration
    capabilities using ONNX models.
    """
    
    def __init__(
        self, 
        face_db: Optional[AbstractFaceDB] = None,
        model_path: Optional[str] = None,
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        similarity_threshold: float = 0.4,
        enable_detection: bool = True,
        **kwargs
    ):
        """
        Initialize the ONNX Face Recognizer.
        
        Args:
            face_db: Face database for storing and retrieving face data
            model_path: Path to the ONNX model file
            det_size: Detection size (width, height)
            det_thresh: Detection threshold
            similarity_threshold: Similarity threshold for face matching
            enable_detection: Whether to enable face detection
            **kwargs: Additional parameters
        """
        self.face_db = face_db
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.similarity_threshold = similarity_threshold
        self.enable_detection = enable_detection
        
        if model_path:
            self.session = ort.InferenceSession(model_path)
            logger.info(f"Loaded ONNX model from {model_path}")
        else:
            self.session = None
            logger.warning("No ONNX model path provided, some functions may not work")
        
        logger.info(f"Initialized ONNXFaceRecognizer with detection threshold {det_thresh}")
    
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
        
        if self.session is None:
            logger.error("No ONNX model loaded for face detection")
            return []
        
        height, width = image.shape[:2]
        input_size = self.det_size
        
        if input_size != (height, width):
            img_resized = cv2.resize(image, input_size)
        else:
            img_resized = image.copy()
        
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_resized
        
        img_norm = img_rgb.astype(np.float32) / 255.0
        
        input_tensor = np.transpose(img_norm, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        try:
            outputs = self.session.run([output_name], {input_name: input_tensor})
            detections = outputs[0]
        except Exception as e:
            logger.error(f"Error during ONNX inference: {e}")
            return []
        
        face_detections = []
        
        for det in detections:
            if det[4] < self.det_thresh:
                continue
                
            scale_x = width / input_size[0]
            scale_y = height / input_size[1]
            
            x1, y1, x2, y2 = int(det[0] * scale_x), int(det[1] * scale_y), int(det[2] * scale_x), int(det[3] * scale_y)
            
            landmarks = []
            if len(det) > 5:
                for i in range(5):
                    lm_x = int(det[5 + i*2] * scale_x)
                    lm_y = int(det[5 + i*2 + 1] * scale_y)
                    landmarks.append([lm_x, lm_y])
            
            face_detection = FaceDetection(
                bbox=[x1, y1, x2, y2],
                confidence=float(det[4]),
                landmarks=landmarks,
                id=str(uuid.uuid4()),  # Generate temporary ID
                name="Unknown",
                quality_score=1.0  # Default quality score
            )
            
            face_detections.append(face_detection)
        
        logger.info(f"Detected {len(face_detections)} faces")
        return face_detections
    
    def extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding vector from the input image.
        
        Args:
            image: Input face image as numpy array (BGR format)
            
        Returns:
            Face embedding vector as numpy array, or None if extraction fails
        """
        if self.session is None:
            logger.error("No ONNX model loaded for embedding extraction")
            return None
        
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (112, 112))
            img_norm = img_resized.astype(np.float32) / 255.0
            
            input_tensor = np.transpose(img_norm, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            
            outputs = self.session.run([output_name], {input_name: input_tensor})
            embedding = outputs[0][0]
            
            embedding_norm = embedding / np.linalg.norm(embedding)
            
            return embedding_norm
            
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
            detection_score = 0.0
            if face.confidence is not None:
                detection_score = float(face.confidence)
                
            quality_score = 0.0
            if face.quality_score is not None:
                quality_score = float(face.quality_score)
                
            face_metadata['detection_score'] = detection_score
            face_metadata['quality_score'] = quality_score
            face_metadata['bbox'] = face.bbox
            face_metadata['created_at'] = datetime.now().isoformat()
            
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
            'model_type': 'ONNX',
            'detection_threshold': self.det_thresh,
            'detection_size': self.det_size,
            'similarity_threshold': self.similarity_threshold,
            'database_type': self.face_db.__class__.__name__ if self.face_db else 'None'
        }
        
        if self.session:
            try:
                metadata = self.session.get_modelmeta()
                if metadata:
                    model_info['model_name'] = metadata.name
                    model_info['model_version'] = metadata.version
                    model_info['model_description'] = metadata.description
            except Exception as e:
                logger.warning(f"Error getting ONNX model metadata: {e}")
        
        return model_info
