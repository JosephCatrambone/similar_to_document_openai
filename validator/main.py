from typing import Any, Callable, Dict, Optional

from guardrails.logger import logger
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

from openai import OpenAI


@register_validator(name="guardrails/similar_to_document_openai", data_type="string")
class SimilarToDocument(Validator):
    """Validates that a value is similar to the document.

    This validator checks if the value is similar to the document by checking
    the cosine similarity between the value and the document, using an
    embedding.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `guardrails/similar_to_document`  |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        document (str): The document string to use for similarity check.
        threshold (float): The minimum cosine similarity to be considered similar.  Defaults to 0.7.
        embedding_function (Callable): A function that takes a string and returns a list of floats. If unspecified, uses the OpenAI client.
        embedding_model_name (str): The name of the OpenAI embedding model to use. Defaults to "text-embedding-ada-002". embedding_function takes precedence.
    """  # noqa

    def __init__(
        self,
        document: str,
        threshold: float = 0.7,
        on_fail: Optional[Callable] = None,
        embedding_function: Optional[Callable] = None,
        embedding_model_name: Optional[str] = "text-embedding-ada-002"
    ):
        super().__init__(
            on_fail=on_fail, document=document, threshold=threshold
        )

        self._document = document
        self._threshold = float(threshold)

        # Either use the embedding function or, if it's undefined, use an OpenAI client.
        if embedding_function is not None:
            self._embed_function = embedding_function
        else:
            client = OpenAI()
            def embed(txt):
                return client.embeddings.create(model=embedding_model_name, input=txt)
            self._embed_function = embed

        # Compute the document embedding
        try:
            self._document_embedding = self._embed_function(document)
        except Exception as e:
            raise RuntimeError(
                f"Failed to encode the document {document} using the provided model."
            ) from e

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        """Validation method for the SimilarToDocument validator."""

        logger.debug(f"Validating {value} is similar to the given document...")
        # Compute the value embedding
        try:
            value_embedding = self._embed_function(value)
        except Exception as e:
            raise RuntimeError(
                f"Failed to encode the value {value} using the model {self._model}."
            ) from e

        # Compute the cosine similarity between the document and the value
        similarity = cosine_similarity(self._document_embedding, value_embedding)
        print(f"Similarity: {round(similarity, 3)}, Type: {type(similarity)}")

        # Compare the similarity with the threshold
        if similarity < self._threshold:
            return FailResult(
                error_message=f"Value {value} is not similar enough "
                f"to document {self._document}.",
            )

        return PassResult()


def dot_product(a: list[float], b: list[float]) -> float:
    accumulator = 0.0
    for i, j in zip(a, b):
        accumulator += i*j
    return accumulator


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_magnitude = dot_product(a, a)
    b_magnitude = dot_product(b, b)
    # Prevent divide-by-zero:
    if a_magnitude == 0 or b_magnitude == 0:
        a_magnitude = 1
        b_magnitude = 1
    return dot_product(a, b) / (a_magnitude * b_magnitude)**0.5
