"""Face recognition service using InsightFace."""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import pickle
import json
from PIL import Image
import io
import base64

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

from app.config import get_config
from app.models import MetadataStore
from app.utils.resource_paths import get_face_models_path


class FaceService:
    """Service for face detection and recognition."""
    
    def __init__(self):
        """Initialize face service."""
        self.face_app = None
        self.face_embeddings: Dict[str, List[np.ndarray]] = {}  # face_id -> list of embeddings
        self.face_mapping_file = None
        
    def _ensure_face_app(self):
        """Ensure face analysis app is initialized."""
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("InsightFace library not available. Install with: pip install insightface")
        
        if self.face_app is None:
            config = get_config()
            
            # Determine model root directory
            if config.face_models_dir:
                # Use explicitly configured directory
                model_root = config.face_models_dir
            else:
                # Use PyInstaller-compatible model directory
                # This will be model/ in source or extracted to temp in PyInstaller bundle
                model_root = str(get_face_models_path())
            
            # Log model path for debugging
            print(f"[FaceService] Initializing with model root: {model_root}")
            model_path = Path(model_root)
            if model_path.exists():
                print(f"[FaceService] Model root exists: {model_path}")
                models_dir = model_path / "models"
                if models_dir.exists():
                    print(f"[FaceService] Models directory exists: {models_dir}")
                    buffalo_dir = models_dir / "buffalo_l"
                    if buffalo_dir.exists():
                        print(f"[FaceService] Buffalo_l directory exists: {buffalo_dir}")
                        model_files = list(buffalo_dir.glob("*.onnx"))
                        print(f"[FaceService] Found {len(model_files)} ONNX files: {[f.name for f in model_files]}")
                        
                        # Validate we have all required model files
                        required_models = ['det_10g.onnx', 'genderage.onnx', 'w600k_r50.onnx', '1k3d68.onnx', '2d106det.onnx']
                        found_models = [f.name for f in model_files]
                        missing_models = [m for m in required_models if m not in found_models]
                        if missing_models:
                            raise RuntimeError(f"Missing required model files: {missing_models}")
                    else:
                        raise RuntimeError(f"Buffalo_l directory not found at {buffalo_dir}. Make sure models are included in PyInstaller package.")
                else:
                    raise RuntimeError(f"Models directory not found at {models_dir}. Check PyInstaller configuration.")
            else:
                raise RuntimeError(f"Model root does not exist: {model_path}. Check PyInstaller configuration.")
            
            try:
                self.face_app = FaceAnalysis(
                    name='buffalo_l',
                    root=model_root,
                    providers=['CPUExecutionProvider'],  # Use CPU, can be changed to GPU
                    allowed_modules=['detection', 'recognition']  # Only use detection and recognition, skip landmarks
                )
                print(f"[FaceService] FaceAnalysis object created")
                self.face_app.prepare(ctx_id=0, det_size=(640, 640))
                print(f"[FaceService] Face analysis initialized successfully")
            except Exception as e:
                print(f"[FaceService] ERROR initializing face analysis: {e}")
                raise
    
    def _get_face_mapping_path(self, metadata_store: MetadataStore) -> Path:
        """Get path to face embeddings mapping file."""
        config = get_config()
        rag_dir = config.get_rag_directory()
        if not rag_dir:
            # Fallback to metadata store location
            storage_path = metadata_store.storage_path
            if storage_path:
                rag_dir = storage_path.parent / "rag"
            else:
                raise ValueError("Cannot determine RAG directory location")
        
        rag_dir.mkdir(exist_ok=True)
        return rag_dir / "face_embeddings.pkl"
    
    def load_face_embeddings(self, metadata_store: MetadataStore) -> bool:
        """
        Load face embeddings from file.
        
        Args:
            metadata_store: Metadata store to determine file location
            
        Returns:
            True if loaded successfully, False if file doesn't exist
        """
        self.face_mapping_file = self._get_face_mapping_path(metadata_store)
        
        if not self.face_mapping_file.exists():
            self.face_embeddings = {}
            return False
        
        try:
            with open(self.face_mapping_file, 'rb') as f:
                loaded_embeddings = pickle.load(f)
            
            # Migrate old format (single embedding) to new format (list of embeddings)
            self.face_embeddings = {}
            for face_id, embedding in loaded_embeddings.items():
                if isinstance(embedding, list):
                    # Already in new format
                    self.face_embeddings[face_id] = embedding
                else:
                    # Old format - convert single embedding to list
                    self.face_embeddings[face_id] = [embedding]
                    
            return True
        except Exception as e:
            print(f"Error loading face embeddings: {e}")
            self.face_embeddings = {}
            return False
    
    def save_face_embeddings(self, metadata_store: MetadataStore) -> bool:
        """
        Save face embeddings to file.
        
        Args:
            metadata_store: Metadata store to determine file location
            
        Returns:
            True if saved successfully
        """
        if self.face_mapping_file is None:
            self.face_mapping_file = self._get_face_mapping_path(metadata_store)
        
        try:
            with open(self.face_mapping_file, 'wb') as f:
                pickle.dump(self.face_embeddings, f)
            return True
        except Exception as e:
            print(f"Error saving face embeddings: {e}")
            return False
    
    def rename_face_id(
        self,
        old_face_id: str,
        new_face_id: str,
        metadata_store: MetadataStore
    ) -> bool:
        """
        Rename a face ID in the embeddings mapping.
        
        Note: If new_face_id already exists, the embeddings from old_face_id
        will be merged into new_face_id's embedding list. This allows multiple
        embeddings per person for better recognition accuracy.
        
        Args:
            old_face_id: Current face ID to rename
            new_face_id: New face ID name
            metadata_store: Metadata store to determine file location
            
        Returns:
            True if renamed successfully, False if old_face_id doesn't exist
        """
        # Load existing embeddings
        self.load_face_embeddings(metadata_store)
        
        # Check if old face ID exists
        if old_face_id not in self.face_embeddings:
            return False
        
        # If new face ID already exists, merge the embeddings
        if new_face_id in self.face_embeddings:
            print(f"[FaceService] Merging embeddings from '{old_face_id}' into '{new_face_id}' (now has {len(self.face_embeddings[new_face_id]) + len(self.face_embeddings[old_face_id])} embeddings).")
            self.face_embeddings[new_face_id].extend(self.face_embeddings[old_face_id])
        else:
            # Move embeddings to new ID
            self.face_embeddings[new_face_id] = self.face_embeddings[old_face_id]
        
        # Remove old face ID
        del self.face_embeddings[old_face_id]
        
        # Save updated embeddings
        return self.save_face_embeddings(metadata_store)
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Cosine similarity ranges from -1 to 1:
        - 1.0: Identical faces
        - 0.7-0.9: Very likely same person
        - 0.5-0.7: Possibly same person
        - < 0.5: Different people
        
        InsightFace embeddings are typically already L2-normalized,
        but we normalize again to be safe.
        """
        # Normalize embeddings
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        
        # Compute cosine similarity
        similarity = float(np.dot(emb1_norm, emb2_norm))
        
        return similarity
    
    def _find_matching_face_id(
        self, 
        embedding: np.ndarray, 
        threshold: float = 0.5
    ) -> Optional[Tuple[str, float]]:
        """
        Find matching face ID for an embedding.
        
        Compares against all embeddings stored for each face ID and returns
        the best match. Having multiple embeddings per person improves accuracy.
        
        Uses cosine similarity for comparison. Recommended thresholds:
        - 0.4: Very loose matching (may incorrectly match different people)
        - 0.5: Balanced matching (recommended default)
        - 0.6: Stricter matching (may create duplicates for same person)
        - 0.7+: Very strict (will likely create many duplicates)
        
        Args:
            embedding: Face embedding to match (512-dim from InsightFace)
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            Tuple of (face_id, similarity) for best match, or None if no match above threshold
        """
        if not self.face_embeddings:
            return None
        
        best_match_id = None
        best_similarity = threshold  # Start with threshold as minimum
        
        # Compare with all stored face embeddings
        for face_id, stored_embeddings in self.face_embeddings.items():
            # Compare against all embeddings for this face ID
            for stored_embedding in stored_embeddings:
                similarity = self._compute_similarity(embedding, stored_embedding)
                
                # Keep track of best match
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = face_id
        
        if best_match_id is not None:
            return (best_match_id, best_similarity)
        
        return None
    
    def _generate_new_face_id(self) -> str:
        """Generate a new unique face ID."""
        if not self.face_embeddings:
            return "face_001"
        
        # Get highest existing number
        existing_numbers = []
        for face_id in self.face_embeddings.keys():
            if face_id.startswith("face_"):
                try:
                    num = int(face_id.split("_")[1])
                    existing_numbers.append(num)
                except (ValueError, IndexError):
                    pass
        
        if existing_numbers:
            next_num = max(existing_numbers) + 1
        else:
            next_num = 1
        
        return f"face_{next_num:03d}"
    
    async def detect_and_identify_faces(
        self,
        image_path: Path,
        metadata_store: MetadataStore,
        similarity_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Detect faces in an image and identify them.
        
        Uses cosine similarity to match faces against stored embeddings.
        A face is considered the same person if similarity > threshold.
        
        Recommended thresholds:
        - 0.4: Loose matching (may group different people)
        - 0.5: Balanced (recommended default)
        - 0.6: Strict (may create duplicates for same person)
        
        Args:
            image_path: Path to image file
            metadata_store: Metadata store for face embeddings
            similarity_threshold: Minimum similarity to match faces (0-1)
            
        Returns:
            List of face information dicts with keys:
            - face_id: Identifier for the face
            - bbox: Bounding box [x, y, width, height]
            - confidence: Detection confidence
            - is_new: Whether this is a newly registered face
            - similarity: Similarity to matched face (only if is_new=False)
        """
        self._ensure_face_app()
        
        # Load existing face embeddings from file
        self.load_face_embeddings(metadata_store)
        
        # Read image
        print(f"[FaceService] Loading image from: {image_path}")
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            pil_img = Image.open(image_path).convert('RGB')
            if pil_img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            img = np.array(pil_img)
            if img is None or img.size == 0:
                raise ValueError(f"Invalid image array from: {image_path}")
            
            print(f"[FaceService] Image loaded: shape={img.shape}, dtype={img.dtype}")
        except Exception as e:
            print(f"[FaceService] ERROR loading image: {e}")
            raise
        
        # Detect faces using InsightFace
        print(f"[FaceService] Detecting faces...")
        print(f"[FaceService] face_app type: {type(self.face_app)}")
        print(f"[FaceService] face_app is None: {self.face_app is None}")
        print(f"[FaceService] img shape: {img.shape}, dtype: {img.dtype}")
        print(f"[FaceService] img min: {img.min()}, max: {img.max()}")
        
        try:
            print(f"[FaceService] Calling face_app.get()...")
            faces = self.face_app.get(img)
            print(f"[FaceService] get() returned successfully")
            print(f"[FaceService] Detected {len(faces)} face(s)")
        except Exception as e:
            print(f"[FaceService] ERROR during face detection: {type(e).__name__}: {e}")
            import traceback
            print(f"[FaceService] Traceback:")
            traceback.print_exc()
            raise
        
        results = []
        new_faces_added = False
        
        for face in faces:
            # Get 512-dimensional normalized embedding from InsightFace
            embedding = face.embedding
            
            # Try to match with existing faces in the face map
            match_result = self._find_matching_face_id(embedding, similarity_threshold)
            
            if match_result is not None:
                # Found a match - use existing face ID and add this embedding to the list
                face_id, similarity = match_result
                is_new = False
                
                # Add this embedding to the existing face ID's embedding list
                # This helps improve recognition with multiple samples per person
                self.face_embeddings[face_id].append(embedding)
                new_faces_added = True  # Mark as updated to save
            else:
                # No match found - create new face ID with this embedding
                face_id = self._generate_new_face_id()
                # Store the embedding as a list (can grow with more detections)
                self.face_embeddings[face_id] = [embedding]
                is_new = True
                new_faces_added = True
                similarity = None
            
            # Get bounding box
            bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
            # Convert to [x, y, width, height]
            bbox_formatted = [
                bbox[0],
                bbox[1],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1]
            ]
            
            result_dict = {
                "face_id": face_id,
                "bbox": bbox_formatted,
                "confidence": float(face.det_score),
                "is_new": is_new
            }
            
            # Include similarity score if this was a match
            if similarity is not None:
                result_dict["similarity"] = float(similarity)
            
            results.append(result_dict)
        
        # Save face embeddings to file if new faces were added
        if new_faces_added:
            self.save_face_embeddings(metadata_store)
        
        return results
    
    async def get_face_crop(
        self,
        image_path: Path,
        face_id: str,
        metadata_store: MetadataStore,
        padding: int = 20,
        min_similarity: float = 0.4
    ) -> Optional[bytes]:
        """
        Get cropped image of a specific face.
        
        Re-detects faces in the image and finds the one matching the stored
        face_id embedding using cosine similarity.
        
        Args:
            image_path: Path to image file
            face_id: Face ID to crop
            metadata_store: Metadata store for face embeddings
            padding: Padding around face bbox in pixels
            min_similarity: Minimum similarity to consider a match (default: 0.4)
            
        Returns:
            JPEG bytes of cropped face or None if not found
        """
        self._ensure_face_app()
        
        # Load existing face embeddings from file
        self.load_face_embeddings(metadata_store)
        
        if face_id not in self.face_embeddings:
            return None
        
        # Read image
        img_pil = Image.open(image_path).convert('RGB')
        img = np.array(img_pil)
        
        # Detect all faces in the image
        faces = self.face_app.get(img)
        
        # Find the face that best matches any of the stored embeddings for this face ID
        target_embeddings = self.face_embeddings[face_id]
        best_match = None
        best_similarity = min_similarity  # Must be at least this similar
        
        for face in faces:
            # Compare against all stored embeddings for this face ID
            for target_embedding in target_embeddings:
                similarity = self._compute_similarity(face.embedding, target_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = face
        
        # No match found above threshold
        if best_match is None:
            return None
        
        # Get bounding box with padding
        bbox = best_match.bbox.astype(int)
        x1 = max(0, bbox[0] - padding)
        y1 = max(0, bbox[1] - padding)
        x2 = min(img_pil.width, bbox[2] + padding)
        y2 = min(img_pil.height, bbox[3] + padding)
        
        # Crop face region
        face_crop = img_pil.crop((x1, y1, x2, y2))
        
        # Convert to JPEG bytes
        buffer = io.BytesIO()
        face_crop.save(buffer, format='JPEG', quality=95)
        return buffer.getvalue()
    
    async def get_face_crop_base64(
        self,
        image_path: Path,
        face_id: str,
        metadata_store: MetadataStore,
        padding: int = 20
    ) -> Optional[str]:
        """
        Get base64-encoded cropped image of a specific face.
        
        Args:
            image_path: Path to image file
            face_id: Face ID to crop
            metadata_store: Metadata store for face embeddings
            padding: Padding around face bbox in pixels
            
        Returns:
            Base64-encoded JPEG string or None if not found
        """
        face_bytes = await self.get_face_crop(image_path, face_id, metadata_store, padding)
        if face_bytes is None:
            return None
        
        return base64.b64encode(face_bytes).decode('utf-8')


# Global face service instance
_face_service = FaceService()


def get_face_service() -> FaceService:
    """Get the global face service instance."""
    return _face_service
