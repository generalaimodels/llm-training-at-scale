# OpenAI Embeddings API Documentation

## API Reference

<!-- ```python
__all__ = ["Embeddings", "AsyncEmbeddings"]
``` -->

## `Embeddings` Class

A synchronous API resource for creating text embeddings.

### Properties

#### `with_raw_response`
```python
@cached_property
def with_raw_response(self) -> EmbeddingsWithRawResponse:
    # Returns a wrapped version of the API that provides access to raw HTTP response data
```

#### `with_streaming_response`
```python
@cached_property
def with_streaming_response(self) -> EmbeddingsWithStreamingResponse:
    # Returns a wrapped version of the API that doesn't eagerly read the response body
```

### Methods

#### `create`
```python
def create(
    self,
    *,
    input: Union[str, List[str], Iterable[int], Iterable[Iterable[int]]],
    model: Union[str, EmbeddingModel],
    dimensions: int | NotGiven = NOT_GIVEN,
    encoding_format: Literal["float", "base64"] | NotGiven = NOT_GIVEN,
    user: str | NotGiven = NOT_GIVEN,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> CreateEmbeddingResponse:
    # Creates an embedding vector representing the input text
```

**Parameters:**

- `input`: Input text to embed, as a string, array of strings, or token arrays. Limited to model's max token count (8192 for `text-embedding-ada-002`). Cannot be empty, and arrays must be 2048 dimensions or less.
- `model`: Model ID to use for embedding generation
- `dimensions`: Number of dimensions for output embeddings (supported in `text-embedding-3` and later models)
- `encoding_format`: Format for embeddings - either `float` or `base64`
- `user`: Unique identifier for end-user to help with monitoring/abuse detection
- `extra_headers`: Additional HTTP headers for the request
- `extra_query`: Additional query parameters
- `extra_body`: Additional JSON properties for the request body
- `timeout`: Request timeout override (in seconds)

**Returns:** `CreateEmbeddingResponse` containing the embedding vectors

## `AsyncEmbeddings` Class

Asynchronous version of the Embeddings API resource.

### Properties

#### `with_raw_response`
```python
@cached_property
def with_raw_response(self) -> AsyncEmbeddingsWithRawResponse:
    # Returns a wrapped version of the API that provides access to raw HTTP response data
```

#### `with_streaming_response`
```python
@cached_property
def with_streaming_response(self) -> AsyncEmbeddingsWithStreamingResponse:
    # Returns a wrapped version of the API that doesn't eagerly read the response body
```

### Methods

#### `create`
```python
async def create(
    self,
    *,
    input: Union[str, List[str], Iterable[int], Iterable[Iterable[int]]],
    model: Union[str, EmbeddingModel],
    dimensions: int | NotGiven = NOT_GIVEN,
    encoding_format: Literal["float", "base64"] | NotGiven = NOT_GIVEN,
    user: str | NotGiven = NOT_GIVEN,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> CreateEmbeddingResponse:
    # Creates an embedding vector representing the input text asynchronously
```

**Parameters:** Same as the synchronous version

## Helper Classes

### `EmbeddingsWithRawResponse`
Wrapper class providing access to raw HTTP response data for synchronous embeddings requests.

### `AsyncEmbeddingsWithRawResponse`
Wrapper class providing access to raw HTTP response data for asynchronous embeddings requests.

### `EmbeddingsWithStreamingResponse`
Wrapper class for streamed response handling with synchronous embeddings requests.

### `AsyncEmbeddingsWithStreamingResponse`
Wrapper class for streamed response handling with asynchronous embeddings requests.

## Implementation Details

- The API automatically uses base64 encoding when numpy is available for performance optimization
- Response parsing handles base64-encoded embeddings, converting them to floating-point arrays
- Requests are made to the `/embeddings` endpoint