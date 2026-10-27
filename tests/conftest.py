from __future__ import annotations

import pytest

from bevserve.constants import CAMERA_ORDER
from bevserve.preprocess import ModelInputs, build_inputs
from tests.synthetic import synthetic_image, synthetic_rig


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def model_inputs() -> ModelInputs:
    images = {name: synthetic_image(i) for i, name in enumerate(CAMERA_ORDER)}
    return build_inputs(images, synthetic_rig())
