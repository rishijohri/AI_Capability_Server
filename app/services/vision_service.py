"""Vision service for image/video tagging and description."""

from typing import List, Optional, Dict, Any
from pathlib import Path
import asyncio
import json
import base64

from app.config import get_config
from app.utils import ImageProcessor, get_process_manager
from app.models import FileMetadata
from app.services.llm_service import get_llm_service


class VisionService:
    """Service for vision-based tasks (tagging, description)."""
    
    def __init__(self):
        """Initialize vision service."""
        self.process_manager = get_process_manager()
        self.image_processor = ImageProcessor()
    
    def _extract_xml_tags(self, response: str) -> tuple[str, str]:
        """Extract thinking and conclusion from XML tags.
        
        Args:
            response: Model response with XML tags
            
        Returns:
            Tuple of (thinking, conclusion)
        """
        import re
        
        # Extract think tag
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
        thinking = think_match.group(1).strip() if think_match else ""
        
        # Extract conclusion tag
        conclusion_match = re.search(r'<conclusion>(.*?)</conclusion>', response, re.DOTALL | re.IGNORECASE)
        conclusion = conclusion_match.group(1).strip() if conclusion_match else response.strip()
        
        return thinking, conclusion
    
    async def generate_tags(
        self,
        file_path: Path,
        file_type: str,
        vision_model: str,
        mmproj_file: Optional[str] = None,
        startup_callback = None,
        dimension_callback = None,
        keep_loaded: bool = False
    ) -> tuple[str, List[str]]:
        """
        Generate tags for an image or video file.
        
        Args:
            file_path: Path to the file
            file_type: "image" or "video"
            vision_model: Name of vision model to use
            mmproj_file: Optional mmproj file for vision model
            startup_callback: Optional callback for startup command
            dimension_callback: Optional callback for image dimensions
            
        Returns:
            Tuple of (thinking, list of tags)
        """
        # Get original dimensions
        original_dims = self._get_original_dimensions(file_path, file_type)
        
        # Prepare image
        image_bytes = await self._prepare_image(file_path, file_type)
        
        # Get processed dimensions
        processed_dims = self._get_dimensions_from_bytes(image_bytes)
        
        # Send dimension info via callback before LLM call
        if dimension_callback:
            await dimension_callback(original_dims, processed_dims, len(image_bytes))
        
        # Get configurable prompt for tagging
        from app.config import get_config
        config = get_config()
        prompt = config.tag_prompt
        
        # Generate tags using vision model
        response, startup_cmd = await self._call_vision_model(
            image_bytes,
            prompt,
            vision_model,
            mmproj_file,
            keep_loaded
        )
        
        # Send startup command immediately via callback
        if startup_callback and startup_cmd:
            await startup_callback(startup_cmd)
        
        # Extract thinking and conclusion from XML tags
        thinking, conclusion = self._extract_xml_tags(response)
        
        # Parse tags from conclusion
        tags = self._parse_tags(conclusion)
        return thinking, tags
    
    async def generate_description(
        self,
        file_path: Path,
        file_type: str,
        vision_model: str,
        mmproj_file: Optional[str] = None,
        startup_callback = None,
        dimension_callback = None,
        keep_loaded: bool = False
    ) -> tuple[str, str]:
        """
        Generate description for an image or video file.
        
        Args:
            file_path: Path to the file
            file_type: "image" or "video"
            vision_model: Name of vision model to use
            mmproj_file: Optional mmproj file for vision model
            startup_callback: Optional callback for startup command
            dimension_callback: Optional callback for image dimensions
            
        Returns:
            Tuple of (thinking, description)
        """
        # Get original dimensions
        original_dims = self._get_original_dimensions(file_path, file_type)
        
        # Prepare image
        image_bytes = await self._prepare_image(file_path, file_type)
        
        # Get processed dimensions
        processed_dims = self._get_dimensions_from_bytes(image_bytes)
        
        # Send dimension info via callback before LLM call
        if dimension_callback:
            await dimension_callback(original_dims, processed_dims, len(image_bytes))
        
        # Get configurable prompt for description
        from app.config import get_config
        config = get_config()
        prompt = config.describe_prompt
        
        # Generate description using vision model
        response, startup_cmd = await self._call_vision_model(
            image_bytes,
            prompt,
            vision_model,
            mmproj_file,
            keep_loaded
        )
        
        # Send startup command immediately via callback
        if startup_callback and startup_cmd:
            await startup_callback(startup_cmd)
        
        # Extract thinking and conclusion from XML tags
        thinking, conclusion = self._extract_xml_tags(response)
        
        return thinking, conclusion.strip()
    
    async def _prepare_image(self, file_path: Path, file_type: str) -> bytes:
        """
        Prepare image for vision model processing.
        
        Args:
            file_path: Path to file
            file_type: "image" or "video"
            
        Returns:
            Image bytes in JPEG format
        """
        config = get_config()
        scale = config.get_image_scale()
        
        if file_type == "video":
            # Extract frame from video
            image_bytes = self.image_processor.extract_video_frame(file_path)
            if image_bytes is None:
                raise ValueError(f"Failed to extract frame from video: {file_path}")
            
            # If we need to resize, save and reload
            if scale < 1.0:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    tmp.write(image_bytes)
                    tmp_path = Path(tmp.name)
                
                try:
                    image_bytes = self.image_processor.resize_image_with_quality(
                        tmp_path,
                        scale
                    )
                finally:
                    tmp_path.unlink()
            
            return image_bytes
        else:
            # Process image directly
            return self.image_processor.resize_image_with_quality(
                file_path,
                scale
            )
    
    async def _call_vision_model(
        self,
        image_bytes: bytes,
        prompt: str,
        vision_model: str,
        mmproj_file: Optional[str] = None,
        keep_loaded: bool = False
    ) -> tuple[str, Optional[str]]:
        """
        Call vision model with image and prompt using LLMService.
        
        Args:
            image_bytes: Image data
            prompt: Text prompt
            vision_model: Name of vision model
            mmproj_file: Optional mmproj file
            keep_loaded: If True, don't unload model after use (for batch processing)
            
        Returns:
            Tuple of (model_response, startup_command)
        """
        llm_service = get_llm_service()
        
        # Load vision model with mmproj if provided
        load_kwargs = {}
        if mmproj_file:
            load_kwargs["mmproj"] = mmproj_file
        
        await llm_service.load_model(vision_model, **load_kwargs)
        
        # Get startup command
        startup_cmd = llm_service.get_startup_command()
        
        try:
            # Generate vision response through LLMService
            # This will use llama-server (with base64) or vision binaries depending on backend
            response = await llm_service.generate_vision(
                image_bytes,
                prompt,
                mmproj_file
            )
            return response, startup_cmd
        finally:
            # Only unload if not keeping loaded for batch processing
            if not keep_loaded:
                await llm_service.unload_model()
    
    def _parse_tags(self, response: str) -> List[str]:
        """
        Parse tags from model response.
        
        Args:
            response: Model response text
            
        Returns:
            List of tags
        """
        # Extract tags from response
        # Look for comma-separated values
        lines = response.split('\n')
        tag_line = ""
        
        for line in lines:
            line = line.strip()
            if line and ',' in line:
                # Likely contains tags
                tag_line = line
                break
        
        if not tag_line:
            # Use full response if no clear tag line
            tag_line = response
        
        # Split by comma and clean
        tags = [tag.strip().lower() for tag in tag_line.split(',')]
        tags = [tag for tag in tags if tag and len(tag) > 1]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        
        return unique_tags[:20]  # Limit to 20 tags
    
    def _get_original_dimensions(self, file_path: Path, file_type: str) -> tuple[int, int]:
        """
        Get original dimensions of image or video frame.
        
        Args:
            file_path: Path to file
            file_type: "image" or "video"
            
        Returns:
            Tuple of (width, height)
        """
        if file_type == "video":
            # For video, get dimensions of extracted frame
            try:
                import cv2
                cap = cv2.VideoCapture(str(file_path))
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    return (width, height)
            except:
                pass
            return (0, 0)
        else:
            # For image, get dimensions directly
            return self.image_processor.get_image_dimensions(file_path)
    
    def _get_dimensions_from_bytes(self, image_bytes: bytes) -> tuple[int, int]:
        """
        Get dimensions from image bytes.
        
        Args:
            image_bytes: Image data
            
        Returns:
            Tuple of (width, height)
        """
        from PIL import Image
        import io
        
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return img.size
        except:
            return (0, 0)
    
    async def unload_model(self):
        """Explicitly unload the currently loaded vision model."""
        llm_service = get_llm_service()
        await llm_service.unload_model()


# Global vision service instance
_vision_service = VisionService()


def get_vision_service() -> VisionService:
    """Get the global vision service instance."""
    return _vision_service
