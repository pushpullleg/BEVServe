from __future__ import annotations

import struct
import zlib

import pytest
from PIL import Image

from bevserve import imaging
from bevserve.imaging import ImageRejectedError, check_size, decode_rgb, inspect_header
from tests.synthetic import encode, synthetic_image


def _png_with_declared_size(width: int, height: int) -> bytes:
    """A real 1x1 PNG whose IHDR claims another size (valid CRC), to test header-only checks."""
    data = encode(Image.new("RGB", (1, 1)), "PNG")
    ihdr = bytearray(data[8:33])  # length(4) type(4) payload(13) crc(4)
    ihdr[8:16] = struct.pack(">II", width, height)
    ihdr[21:25] = struct.pack(">I", zlib.crc32(bytes(ihdr[4:21])) & 0xFFFFFFFF)
    return data[:8] + bytes(ihdr) + data[33:]


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_header_reports_format_and_size(fmt: str) -> None:
    header = inspect_header(encode(synthetic_image(1), fmt))
    assert (header.format, header.width, header.height) == (fmt, 1600, 900)


def test_header_inspection_does_not_decode_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: object, **__: object) -> None:
        raise AssertionError("pixel data was decoded")

    data = encode(synthetic_image(2))
    monkeypatch.setattr(Image.Image, "load", fail)
    header = inspect_header(data)
    check_size(header)


@pytest.mark.parametrize(
    "data",
    [b"GIF89a....", b"not an image", b"", b"BM" + b"\x00" * 64],
    ids=["gif", "text", "empty", "bmp"],
)
def test_non_jpeg_png_is_rejected_by_signature(data: bytes) -> None:
    with pytest.raises(ImageRejectedError) as err:
        inspect_header(data)
    assert err.value.code == "image_format"


def test_truncated_header_is_corrupt() -> None:
    with pytest.raises(ImageRejectedError) as err:
        inspect_header(b"\x89PNG\r\n\x1a\n\x00\x00")
    assert err.value.code == "image_corrupt"


def test_wrong_size_is_rejected_before_decode() -> None:
    header = inspect_header(encode(synthetic_image(3, 1280, 720)))
    with pytest.raises(ImageRejectedError) as err:
        check_size(header)
    assert err.value.code == "image_size"
    assert "1280x720" in err.value.detail


def test_decompression_bomb_header_is_rejected_without_decoding() -> None:
    assert imaging.Image.MAX_IMAGE_PIXELS == 2 * 1600 * 900
    with pytest.raises(ImageRejectedError) as err:
        inspect_header(_png_with_declared_size(60000, 60000))
    assert err.value.code == "image_size"


def test_moderately_oversized_header_is_rejected_by_size_check() -> None:
    header = inspect_header(_png_with_declared_size(2000, 1000))
    with pytest.raises(ImageRejectedError):
        check_size(header)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_decode_converts_to_rgb(mode: str) -> None:
    image = synthetic_image(4).convert(mode)
    assert decode_rgb(encode(image, "PNG")).mode == "RGB"


def test_decode_reports_corrupt_body() -> None:
    data = encode(synthetic_image(5))
    with pytest.raises(ImageRejectedError) as err:
        decode_rgb(data[: len(data) // 3])
    assert err.value.code == "image_corrupt"
