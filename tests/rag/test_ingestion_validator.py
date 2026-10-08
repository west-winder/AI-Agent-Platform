
import numpy as np
import pytest

from backend.rag.contracts import Chunk
from backend.rag.ingestion.ingestion_validator import (
    validate_ingestion_data,
)


def build_chunks():
    return [
        Chunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="测试内容",
        )
    ]


# Case 1: 合法数据
def test_valid_ingestion_data():
    result = validate_ingestion_data(
        chunks=build_chunks(),
        chunk_vectors={"chunk_1": np.array([1.0, 0.0])},
        embedding_model="test-model",
        embedding_dimension=2,
    )

    assert result is None


# Case 2: Chunk ID 不匹配
def test_chunk_vector_id_mismatch():
    with pytest.raises(ValueError, match="ID mismatch"):
        validate_ingestion_data(
            chunks=build_chunks(),
            chunk_vectors={"chunk_2": np.array([1.0, 0.0])},
            embedding_model="test-model",
            embedding_dimension=2,
        )


# Case 3～5: 非法向量
@pytest.mark.parametrize(
    "vector, expected_error",
    [
        ([1.0, 0.0, 0.0], "dimension mismatch"),
        ([float("nan"), 1.0], "contains NaN or Inf"),
        ([0.0, 0.0], "must not be zero"),
    ],
)
def test_invalid_vectors(vector, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        validate_ingestion_data(
            chunks=build_chunks(),
            chunk_vectors={"chunk_1": np.array(vector)},
            embedding_model="test-model",
            embedding_dimension=2,
        )
