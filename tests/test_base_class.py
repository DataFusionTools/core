from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pytest

from core.base_class import BaseClass


@dataclass
class _Sample(BaseClass):
    x: Union[List, None, np.ndarray] = None
    y: Optional[List[int]] = None
    z: "int | None" = None


class TestBaseClassValidation:
    @pytest.mark.unittest
    def test_union_accepts_valid_values(self):
        _Sample()
        _Sample(x=np.array([1]), y=[1], z=2)

    @pytest.mark.unittest
    def test_union_rejects_invalid_value(self):
        with pytest.raises(ValueError):
            _Sample(y="a")
