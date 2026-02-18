# Embeddings and AsyncEmbeddings: Technical Documentation

## Overview

The `Embeddings` and `AsyncEmbeddings` classes are fundamental components of the OpenAI Python client library, providing synchronous and asynchronous implementations for generating vector representations of text (embeddings). These classes expose methods to transform text into high-dimensional vectors that capture semantic meaning and relationships between words and concepts.

## Technical Pipeline

```
Input Text → API Request → Embedding Model Processing → Vector Representation → Optional Format Conversion → Response Object
```

## Embeddings Class

The `Embeddings` class provides synchronous operations for generating text embeddings through the OpenAI API.

### Attributes

#### `with_raw_response`
- **Type**: `EmbeddingsWithRawResponse`
- **Purpose**: Returns a wrapper object that preserves the raw HTTP response
- **Implementation**: Uses `@cached_property` decorator for efficient caching
- **Usage**: Access raw response data including headers, status codes, etc.

#### `with_streaming_response`
- **Type**: `EmbeddingsWithStreamingResponse`
- **Purpose**: Returns a wrapper that doesn't eagerly read response body
- **Implementation**: Uses `@cached_property` for performance
- **Usage**: Handles streaming responses for improved memory efficiency with large payloads

### Methods

#### `create()`
- **Purpose**: Generates embedding vectors from input text
- **Return Type**: `CreateEmbeddingResponse`
- **HTTP Method**: POST to "/embeddings" endpoint

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | `Union[str, List[str], Iterable[int], Iterable[Iterable[int]]]` | Yes | Text to embed |
| `model` | `Union[str, EmbeddingModel]` | Yes | Model identifier |
| `dimensions` | `int \| NotGiven` | No | Vector dimensions (text-embedding-3+ only) |
| `encoding_format` | `Literal["float", "base64"] \| NotGiven` | No | Output format |
| `user` | `str \| NotGiven` | No | End-user identifier |
| `extra_headers` | `Headers \| None` | No | Additional HTTP headers |
| `extra_query` | `Query \| None` | No | Additional query parameters |
| `extra_body` | `Body \| None` | No | Additional request body properties |
| `timeout` | `float \| httpx.Timeout \| None \| NotGiven` | No | Request timeout override |

**Implementation Details**:
- Default encoding is "base64" if not specified
- Response processing includes automatic conversion from base64 strings to float arrays
- Optimizes performance by using array.array when NumPy is unavailable
- Uses NumPy for efficient vector handling when available
- Makes API request using internal `_post` method

## AsyncEmbeddings Class

The `AsyncEmbeddings` class provides the same functionality as `Embeddings` but implemented for asynchronous operations.

### Attributes

#### `with_raw_response`
- **Type**: `AsyncEmbeddingsWithRawResponse`
- **Purpose**: Asynchronous version of the raw response accessor
- **Implementation**: Uses `@cached_property` for efficient caching

#### `with_streaming_response`
- **Type**: `AsyncEmbeddingsWithStreamingResponse`
- **Purpose**: Asynchronous streaming response accessor
- **Implementation**: Uses `@cached_property` for performance

### Methods

#### `async create()`
- **Purpose**: Asynchronously generates embedding vectors
- **Return Type**: `CreateEmbeddingResponse`
- **Implementation**: Awaitable version of the synchronous create method

**Parameters**:
Identical to the synchronous `create()` method

**Implementation Details**:
- Uses `await self._post()` for asynchronous HTTP requests
- Maintains the same post-processing of base64-encoded embeddings
- Parameter handling and validation identical to synchronous version

## Technical Considerations

### Input Constraints
- Maximum token limit depends on the model (8192 tokens for text-embedding-ada-002)
- Input cannot be an empty string
- Arrays must be 2048 dimensions or less
- Some models may impose limits on total tokens across multiple inputs

### Optimization Strategies
- Base64 encoding for efficient data transfer
- Conditional NumPy integration for vectorized operations
- Response parsing optimized for different encoding formats
- Cached properties for wrapper objects

### Error Handling
- Type validation through type annotations
- Integration with HTTP client error handling (httpx)
- Response validation through casting to typed objects

## Usage Context

These classes enable:
- Semantic search
- Clustering and classification
- Recommendation systems
- Feature extraction for machine learning
- Natural language processing tasks