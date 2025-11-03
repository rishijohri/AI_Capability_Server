"""Image processing utilities."""

from PIL import Image
from pathlib import Path
from typing import Optional, Tuple
import io
import base64


class ImageProcessor:
    """Handle image processing and quality reduction."""
    
    @staticmethod
    def resize_image_with_quality(
        image_path: Path,
        scale: float = 1.0
    ) -> bytes:
        """
        Resize image by scale factor while preserving aspect ratio.
        
        Args:
            image_path: Path to the image file
            scale: Scale multiplier (1.0 = original, <1.0 = reduce dimensions)
            
        Returns:
            Resized image as bytes in JPEG format
        """
        img = Image.open(image_path)
        
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if scale is less than 1.0
        if scale < 1.0:
            width, height = img.size
            
            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Ensure minimum dimensions
            new_width = max(new_width, 1)
            new_height = max(new_height, 1)
            
            # Resize using LANCZOS interpolation
            if (new_width, new_height) != (width, height):
                img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return buffer.getvalue()
    
    @staticmethod
    def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
        """Get image dimensions without loading full image."""
        with Image.open(image_path) as img:
            return img.size
    
    @staticmethod
    def extract_video_frame(video_path: Path, time_seconds: float = 1.0) -> Optional[bytes]:
        """
        Extract a frame from video at specified time.
        
        Args:
            video_path: Path to video file
            time_seconds: Time in seconds to extract frame
            
        Returns:
            Frame as JPEG bytes, or None if extraction fails
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            # Set frame position
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(fps * time_seconds)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            img = Image.fromarray(frame_rgb)
            
            # Save to bytes
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            return buffer.getvalue()
            
        except Exception as e:
            print(f"Error extracting video frame: {e}")
            return None
    
    @staticmethod
    async def load_image_as_base64(
        image_name: str,
        metadata_store,
        scale: float = 1.0
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Load an image from metadata store and return as base64 encoded string.
        
        Args:
            image_name: Name of the image file
            metadata_store: MetadataStore instance to get file information
            scale: Scale factor for resizing (from config.image_quality)
            
        Returns:
            Tuple of (base64_string, error_message). 
            If successful, returns (base64_string, None).
            If failed, returns (None, error_message).
        """
        # Check if image exists in metadata
        image_metadata = metadata_store.get_metadata_by_filename(image_name)
        
        if not image_metadata:
            return None, f"Image not in metadata: {image_name}"
        
        # Get full path to image
        image_path = metadata_store.get_file_path(image_name)
        
        if not image_path.exists():
            return None, f"Image file not found: {image_name}"
        
        try:
            # Resize image according to scale
            image_bytes = ImageProcessor.resize_image_with_quality(image_path, scale)
            
            # Convert to base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            return image_base64, None
            
        except Exception as e:
            return None, f"Could not load image {image_name}: {str(e)}"
