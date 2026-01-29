import pandas as pd
import pytest

from immunofoundation.rotterdam import _align_samples


def test_align_samples_requires_all_meta_samples():
    expr = pd.DataFrame(
        {
            "probe_id": ["p1", "p2"],
            "S1": [1.0, 2.0],
            "S2": [3.0, 4.0],
        }
    )
    meta = pd.DataFrame({"sample_id": ["S1", "S3"], "age": [50, 60]})
    with pytest.raises(ValueError):
        _align_samples(expr, meta, "sample_id")
