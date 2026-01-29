import pandas as pd
import pytest

from immunofoundation.io_utils import validate_mapping_success, validate_no_duplicate_samples


def test_no_duplicate_samples():
    meta = pd.DataFrame({"sample_id": ["S1", "S1", "S2"]})
    with pytest.raises(ValueError):
        validate_no_duplicate_samples(meta, "sample_id")


def test_mapping_success_rate():
    mapping = pd.DataFrame(
        {
            "gene_id": ["g1", "g2", "g3", "g4"],
            "gene": ["A", None, None, "D"],
        }
    )
    with pytest.raises(ValueError):
        validate_mapping_success(mapping, "gene_id", "gene", 0.75)
