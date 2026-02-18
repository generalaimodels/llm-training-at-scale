**Embeddings API Documentation**
=====================================

**Overview**
------------

The Embeddings API is used to create embedding vectors representing input text. This API is available in both synchronous (`Embeddings`) and asynchronous (`AsyncEmbeddings`) versions.

**Main API Endpoints**
----------------------

*   **`Embeddings.create()`**: Creates an embedding vector representing the input text. **[HIGHLIGHTED PRIMARY ENDPOINT]**
*   **`AsyncEmbeddings.create()`**: Asynchronous version of `Embeddings.create()`. **[HIGHLIGHTED PRIMARY ENDPOINT]**

**Embeddings Class**
--------------------

### **Attributes**

*   **`with_raw_response`** (`cached_property`): Returns an instance of `EmbeddingsWithRawResponse`, allowing access to raw response data (e.g., headers).
    *   **Usage:** `embeddings.with_raw_response.create()`
    *   **Details:** [Accessing Raw Response Data](https://www.github.com/openai/openai-python#accessing-raw-response-data-eg-headers)
*   **`with_streaming_response`** (`cached_property`): Returns an instance of `EmbeddingsWithStreamingResponse`, enabling streaming response handling.
    *   **Usage:** `embeddings.with_streaming_response.create()`
    *   **Details:** [With Streaming Response](https://www.github.com/openai/openai-python#with_streaming_response)

### **Methods**

#### **`create()`** **[PRIMARY METHOD]**

Creates an embedding vector representing the input text.

**Signature:**
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
```

**Parameters:**

| Name | Type | Description | Required |
| --- | --- | --- | --- |
| `input` | `Union[str, List[str], Iterable[int], Iterable[Iterable[int]]]` | Input text to embed. Can be a string, list of strings, or token arrays. Max 8192 tokens for `text-embedding-ada-002`. | **Yes** |
| `model` | `Union[str, EmbeddingModel]` | ID of the model to use. Retrieve available models via [List Models API](https://platform.openai.com/docs/api-reference/models/list). | **Yes** |
| `dimensions` | `int \| NotGiven` | Number of dimensions for output embeddings. Supported in `text-embedding-3` and later. | No |
| `encoding_format` | `Literal["float", "base64"] \| NotGiven` | Format of returned embeddings: `float` or `base64`. Defaults to `base64` if NumPy is installed. | No |
| `user` | `str \| NotGiven` | Unique end-user identifier for abuse monitoring. [Learn More](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids). | No |
| `extra_headers` | `Headers \| None` | Additional headers for the request. | No |
| `extra_query` | `Query \| None` | Additional query parameters. | No |
| `extra_body` | `Body \| None` | Additional JSON properties. | No |
| `timeout` | `float \| httpx.Timeout \| None \| NotGiven` | Request timeout override (seconds). | No |

**Returns:** `CreateEmbeddingResponse` object containing the embedding vector(s).

**Example:**
```python
from openai import OpenAI

client = OpenAI()
embeddings = client.embeddings

response = embeddings.create(
    input="The quick brown fox jumps over the lazy dog",
    model="text-embedding-ada-002"
)

print(response.data[0].embedding)  # Access the embedding vector
```

**Notes:**

*   Input length limits: 8192 tokens for `text-embedding-ada-002`. Use [token counting example](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) to verify.
*   Array inputs must be 2048 dimensions or less.
*   Empty strings are not allowed.

**AsyncEmbeddings Class**
-------------------------

Asynchronous counterpart to `Embeddings`. All attributes and methods mirror their synchronous versions, with `await` required for method calls.

### **Attributes**

*   **`with_raw_response`** (`cached_property`): Async version of `Embeddings.with_raw_response`.
    *   **Usage:** `await async_embeddings.with_raw_response.create()`
*   **`with_streaming_response`** (`cached_property`): Async version of `Embeddings.with_streaming_response`.
    *   **Usage:** `await async_embeddings.with_streaming_response.create()`

### **Methods**

#### **`create()`** **[PRIMARY ASYNC METHOD]**

Async version of `Embeddings.create()`. Creates an embedding vector representing the input text.

**Signature:**
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
```

**Parameters & Returns:** Identical to `Embeddings.create()`.

**Example:**
```python
from openai import AsyncOpenAI

aclient = AsyncOpenAI()
async_embeddings = aclient.embeddings

response = await async_embeddings.create(
    input="The quick brown fox jumps over the lazy dog",
    model="text-embedding-ada-002"
)

print(response.data[0].embedding)  # Access the embedding vector
```

**Response Handling Classes**
-----------------------------

These classes modify the behavior of `create()` to handle raw or streaming responses.

### **1. EmbeddingsWithRawResponse**

*   **Purpose:** Access raw HTTP response data (headers, status code, etc.).
*   **Initialization:** Automatically created via `embeddings.with_raw_response`.
*   **Methods:**
    *   `create()`: Same parameters as `Embeddings.create()`, but returns `RawCreateEmbeddingResponse`.

**Example:**
```python
raw_response = embeddings.with_raw_response.create(
    input="test",
    model="text-embedding-ada-002"
)

print(raw_response.headers)  # Access response headers
print(raw_response.status_code)  # Access HTTP status code
```

### **2. AsyncEmbeddingsWithRawResponse**

Async version of `EmbeddingsWithRawResponse`. Usage:

```python
raw_response = await async_embeddings.with_raw_response.create(
    input="test",
    model="text-embedding-ada-002"
)
```

### **3. EmbeddingsWithStreamingResponse**

*   **Purpose:** Handle responses as streaming data (efficient for large outputs).
*   **Initialization:** Via `embeddings.with_streaming_response`.
*   **Methods:**
    *   `create()`: Same parameters, returns a streamed response iterator.

**Example:**
```python
streamed_response = embeddings.with_streaming_response.create(
    input="test",
    model="text-embedding-ada-002"
)

with streamed_response as response:
    for chunk in response.iter_bytes():  # Or iter_text(), iter_lines()
        print(chunk)
```

### **4. AsyncEmbeddingsWithStreamingResponse**

Async streaming counterpart:

```python
streamed_response = await async_embeddings.with_streaming_response.create(
    input="test",
    model="text-embedding-ada-002"
)

async with streamed_response as response:
    async for chunk in response.iter_bytes():  # Or iter_text(), iter_lines()
        print(chunk)
```

**Response Format Optimization**
-------------------------------

*   If `encoding_format` is not explicitly set **and** NumPy is installed, the API defaults to `base64` encoding for efficiency.
*   Response parser automatically decodes `base64`→`float32` array if needed:
    ```python
    embedding.embedding = np.frombuffer(
        base64.b64decode(data), dtype="float32"
    ).tolist()
    ```
*   If NumPy is missing or `encoding_format="float"`, embeddings are returned as plain lists.

**Error Handling**
------------------

*   Standard HTTP errors (4xx, 5xx) raise exceptions. Check [OpenAI API Error Codes](https://platform.openai.com/docs/guides/error-codes) for details.
*   Input validation errors (e.g., empty string, oversized input) raise descriptive exceptions.

**Type Definitions**
--------------------

*   **`CreateEmbeddingResponse`**: Contains `data` (list of `Embedding` objects), `model`, `usage`.
*   **`Embedding`**: Has `index`, `object`, `embedding` (the vector).
*   **`NotGiven`**: Sentinel type indicating parameter was not supplied.
*   **`EmbeddingModel`**: Type alias for valid model IDs (string enum).

**Migration Notes**
-------------------

*   For `text-embedding-3` and later, `dimensions` parameter is supported.
*   Always check available models via [List Models API](https://platform.openai.com/docs/api-reference/models/list) rather than hardcoding IDs.

By following this documentation, developers should be able to:

1.  Initialize `Embeddings` or `AsyncEmbeddings` clients.
2.  Call `create()` with appropriate inputs and optional parameters.
3.  Understand response formats (`float` vs. `base64`+NumPy optimization).
4.  Utilize raw/streaming response variants when needed.
5.  Handle errors and input constraints properly.

