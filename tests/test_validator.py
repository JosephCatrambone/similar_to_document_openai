import pytest
from guardrails import Guard
from validator import SimilarToDocument


# Create the guard object
guard = Guard().use(
    SimilarToDocument(
        document="""
        Large language models (LLM) are very large deep learning models that are pre-trained on vast amounts of data. 
        The underlying transformer is a set of neural networks that consist of an encoder and a decoder with self-attention capabilities. 
        """,
        threshold=0.7,
        on_fail="exception",
    )
)


# Test happy path
def test_happy_path():
    """Test happy path."""
    response = guard.parse("Large language models are deep learning models that are pre-trained on huge amounts of data.")
    print("Happy path response", response)
    assert response.validation_passed is True


# Test fail path
def test_fail_path():
    """Test fail path."""
    with pytest.raises(Exception):
        response = guard.parse("Salmon is the opposite of salmoff.")
        print("Fail path response", response)
