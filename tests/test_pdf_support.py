"""Tests for PDF document processing support."""

import sys
import io
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pypdfium2 as pdfium
from PIL import Image

# Mock heavy transitive dependencies to avoid needing spacy models etc.
for mod in [
    "app.services", "app.services.llm_service", "app.services.rag_service",
    "app.services.conversation_compaction_service", "app.models", "app.models.metadata",
    "app.models.requests", "app.models.responses",
    "app.utils.objectivity_detector", "app.utils.chat_helpers",
    "app.utils.process_manager", "app.utils.logging_config",
    "app.utils.bookmark_resolver", "app.config",
    "app.utils.resource_paths", "app.utils.system_detector",
    "spacy", "textblob",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from app.utils.image_processor import ImageProcessor


def _create_sample_pdf(num_pages: int = 3, text: str = "Sample PDF content") -> Path:
    """Create a temporary sample PDF for testing."""
    pdf = pdfium.PdfDocument.new()
    for i in range(num_pages):
        pdf.new_page(612, 792)  # Letter size
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf.save(tmp)
    pdf.close()
    tmp.close()
    return Path(tmp.name)


class TestRenderPdfPage:
    """Tests for ImageProcessor.render_pdf_page()."""

    def test_renders_first_page_as_jpeg(self):
        pdf_path = _create_sample_pdf()
        try:
            result = ImageProcessor.render_pdf_page(pdf_path)
            assert result is not None
            # Verify it's valid JPEG
            img = Image.open(io.BytesIO(result))
            assert img.format == "JPEG"
            assert img.width > 0
            assert img.height > 0
        finally:
            pdf_path.unlink()

    def test_renders_specific_page(self):
        pdf_path = _create_sample_pdf(num_pages=5)
        try:
            result = ImageProcessor.render_pdf_page(pdf_path, page_num=2)
            assert result is not None
            img = Image.open(io.BytesIO(result))
            assert img.format == "JPEG"
        finally:
            pdf_path.unlink()

    def test_clamps_page_num_to_valid_range(self):
        pdf_path = _create_sample_pdf(num_pages=2)
        try:
            # page_num beyond range should clamp to last page
            result = ImageProcessor.render_pdf_page(pdf_path, page_num=100)
            assert result is not None
            # negative should clamp to 0
            result2 = ImageProcessor.render_pdf_page(pdf_path, page_num=-5)
            assert result2 is not None
        finally:
            pdf_path.unlink()

    def test_dpi_affects_dimensions(self):
        pdf_path = _create_sample_pdf(num_pages=1)
        try:
            low = ImageProcessor.render_pdf_page(pdf_path, dpi=72)
            high = ImageProcessor.render_pdf_page(pdf_path, dpi=300)
            low_img = Image.open(io.BytesIO(low))
            high_img = Image.open(io.BytesIO(high))
            assert high_img.width > low_img.width
            assert high_img.height > low_img.height
        finally:
            pdf_path.unlink()

    def test_scale_reduces_dimensions(self):
        pdf_path = _create_sample_pdf(num_pages=1)
        try:
            full = ImageProcessor.render_pdf_page(pdf_path, scale=1.0)
            half = ImageProcessor.render_pdf_page(pdf_path, scale=0.5)
            full_img = Image.open(io.BytesIO(full))
            half_img = Image.open(io.BytesIO(half))
            # Half scale should produce roughly half dimensions
            assert abs(half_img.width - full_img.width // 2) <= 1
            assert abs(half_img.height - full_img.height // 2) <= 1
        finally:
            pdf_path.unlink()

    def test_returns_none_for_invalid_path(self):
        result = ImageProcessor.render_pdf_page(Path("/nonexistent/file.pdf"))
        assert result is None


class TestGetPdfDimensions:
    """Tests for ImageProcessor.get_pdf_dimensions()."""

    def test_returns_letter_size_dimensions(self):
        pdf_path = _create_sample_pdf()
        try:
            width, height = ImageProcessor.get_pdf_dimensions(pdf_path)
            # Letter size is 612 x 792 points
            assert width == 612
            assert height == 792
        finally:
            pdf_path.unlink()

    def test_specific_page(self):
        pdf_path = _create_sample_pdf(num_pages=3)
        try:
            width, height = ImageProcessor.get_pdf_dimensions(pdf_path, page_num=1)
            assert width == 612
            assert height == 792
        finally:
            pdf_path.unlink()

    def test_returns_zero_for_invalid_path(self):
        width, height = ImageProcessor.get_pdf_dimensions(Path("/nonexistent.pdf"))
        assert width == 0
        assert height == 0


class TestGetPdfPageCount:
    """Tests for ImageProcessor.get_pdf_page_count()."""

    def test_returns_correct_count(self):
        pdf_path = _create_sample_pdf(num_pages=7)
        try:
            assert ImageProcessor.get_pdf_page_count(pdf_path) == 7
        finally:
            pdf_path.unlink()

    def test_single_page(self):
        pdf_path = _create_sample_pdf(num_pages=1)
        try:
            assert ImageProcessor.get_pdf_page_count(pdf_path) == 1
        finally:
            pdf_path.unlink()

    def test_returns_zero_for_invalid_path(self):
        assert ImageProcessor.get_pdf_page_count(Path("/nonexistent.pdf")) == 0


class TestLoadImageAsBase64WithPdf:
    """Tests for ImageProcessor.load_image_as_base64() with PDF files."""

    def test_loads_pdf_as_base64(self):
        pdf_path = _create_sample_pdf()
        try:
            # Create mock metadata store
            mock_metadata = MagicMock()
            mock_metadata.type = "pdf"
            mock_metadata.tags = ["document"]
            mock_metadata.description = "A sample PDF"

            mock_store = MagicMock()
            mock_store.get_metadata_by_filename.return_value = mock_metadata
            mock_store.get_file_path.return_value = pdf_path

            # Mock config for pdf_page and pdf_dpi (lazy import inside load_image_as_base64)
            mock_config = MagicMock()
            mock_config.pdf_page = 0
            mock_config.pdf_dpi = 150
            mock_get_config = MagicMock(return_value=mock_config)

            # Patch the app.config mock so `from app.config import get_config` works
            config_mod = sys.modules["app.config"]
            config_mod.get_config = mock_get_config

            result_b64, error = asyncio.get_event_loop().run_until_complete(
                ImageProcessor.load_image_as_base64("test.pdf", mock_store, scale=1.0)
            )

            assert error is None
            assert result_b64 is not None
            # Verify it decodes to valid JPEG
            import base64
            decoded = base64.b64decode(result_b64)
            img = Image.open(io.BytesIO(decoded))
            assert img.format == "JPEG"
        finally:
            pdf_path.unlink()

    def test_loads_video_as_base64(self):
        """Verify that video type triggers extract_video_frame path."""
        # Create a real temp file so Path.exists() returns True
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        video_path = Path(tmp.name)

        try:
            mock_metadata = MagicMock()
            mock_metadata.type = "video"

            mock_store = MagicMock()
            mock_store.get_metadata_by_filename.return_value = mock_metadata
            mock_store.get_file_path.return_value = video_path

            with patch.object(ImageProcessor, "extract_video_frame", return_value=None):
                result_b64, error = asyncio.get_event_loop().run_until_complete(
                    ImageProcessor.load_image_as_base64("video.mp4", mock_store, scale=1.0)
                )

            assert result_b64 is None
            assert "Failed to extract video frame" in error
        finally:
            video_path.unlink()

    def test_metadata_not_found(self):
        mock_store = MagicMock()
        mock_store.get_metadata_by_filename.return_value = None

        result_b64, error = asyncio.get_event_loop().run_until_complete(
            ImageProcessor.load_image_as_base64("missing.pdf", mock_store)
        )

        assert result_b64 is None
        assert "not in metadata" in error
