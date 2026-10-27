"""Image intake: cheap checks first, full decode last (review correction 5).

Order per image part:
1. magic bytes (JPEG or PNG), no Pillow involved;
2. Pillow header parse only (format, width, height), no pixel decode;
3. exact-size check against the trained 1600x900 source size;
4. full decode to RGB.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from bevserve.constants import SRC_HEIGHT, SRC_WIDTH

_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Pillow refuses headers above 2x this value during Image.open. The service accepts exactly
# 1600x900, so anything far above it is rejected while only the header has been read.
Image.MAX_IMAGE_PIXELS = 2 * SRC_WIDTH * SRC_HEIGHT


class ImageRejectedError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class ImageHeader:
    format: str  # "JPEG" or "PNG"
    width: int
    height: int


def sniff_format(data: bytes) -> str:
    if data.startswith(_JPEG_MAGIC):
        return "JPEG"
    if data.startswith(_PNG_MAGIC):
        return "PNG"
    raise ImageRejectedError("image_format", "expected JPEG or PNG data")


def inspect_header(data: bytes) -> ImageHeader:
    """Format and size from the header; pixel data is not decoded."""
    sniffed = sniff_format(data)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(data))
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageRejectedError(
            "image_size", f"declared size is far above {SRC_WIDTH}x{SRC_HEIGHT}"
        ) from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageRejectedError("image_corrupt", "image header could not be parsed") from exc
    if image.format != sniffed:
        raise ImageRejectedError("image_format", "content does not match its signature")
    width, height = image.size
    return ImageHeader(format=sniffed, width=width, height=height)


def check_size(header: ImageHeader) -> None:
    if (header.width, header.height) != (SRC_WIDTH, SRC_HEIGHT):
        raise ImageRejectedError(
            "image_size",
            f"image is {header.width}x{header.height}; expected {SRC_WIDTH}x{SRC_HEIGHT}",
        )


def decode_rgb(data: bytes) -> Image.Image:
    """Full decode. Non-RGB modes (L, RGBA, P, CMYK) are converted to RGB; EXIF orientation is
    ignored, as in BEVDet's own loader."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return image if image.mode == "RGB" else image.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageRejectedError("image_corrupt", "image data could not be decoded") from exc
