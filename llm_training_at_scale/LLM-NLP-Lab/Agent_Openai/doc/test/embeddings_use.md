# Detailed Explanation of `Embeddings` and `AsyncEmbeddings` Classes

The `Embeddings` and `AsyncEmbeddings` classes are part of a client library (likely for the OpenAI API) designed to interact with an embeddings endpoint. These classes facilitate the creation of embedding vectors for text or token inputs, enabling applications such as natural language processing (NLP), semantic search, and text similarity analysis. The `Embeddings` class is synchronous, while `AsyncEmbeddings` is asynchronous, catering to different use cases based on performance and concurrency needs.

This explanation covers the following aspects in detail:
1. **Purpose and Context of Embeddings**
2. **Class Definitions and Inheritance**
3. **Attributes and Properties**
4. **Methods and Their Technical Pipeline**
5. **Key Differences Between `Embeddings` and `AsyncEmbeddings`**
6. **Technical Pipeline for Embedding Creation**
7. **Detailed Explanation of Attributes and Methods**

---

## 1. Purpose and Context of Embeddings

Embeddings are numerical representations of text or tokens in a high-dimensional vector space, where semantically similar inputs are mapped to nearby points. They are widely used in machine learning and NLP for tasks such as:
- Text classification
- Semantic similarity analysis
- Clustering
- Information retrieval

The `Embeddings` and `AsyncEmbeddings` classes provide an interface to an API (e.g., OpenAI's embeddings endpoint) that generates these vectors using pre-trained models. The primary difference between the two classes is their execution model:
- `Embeddings`: Synchronous, blocking operations suitable for simpler workflows.
- `AsyncEmbeddings`: Asynchronous, non-blocking operations ideal for high-concurrency or performance-sensitive applications.

---

## 2. Class Definitions and Inheritance

Both classes are defined in the provided code and inherit from base classes in the library, ensuring consistency with the API client architecture.

### 2.1 `Embeddings` Class
- **Definition**: `class Embeddings(SyncAPIResource)`
- **Inheritance**: Inherits from `SyncAPIResource`, a base class for synchronous API resource interactions.
- **Purpose**: Provides a synchronous interface to interact with the embeddings endpoint, blocking the execution thread until the API response is received.

### 2.2 `AsyncEmbeddings` Class
- **Definition**: `class AsyncEmbeddings(AsyncAPIResource)`
- **Inheritance**: Inherits from `AsyncAPIResource`, a base class for asynchronous API resource interactions.
- **Purpose**: Provides an asynchronous interface, allowing non-blocking operations using Python's `asyncio` framework, suitable for concurrent or I/O-bound tasks.

---

## 3. Attributes and Properties

Both classes share similar attributes and properties, differing only in their synchronous or asynchronous nature. Below is a detailed breakdown of the key attributes and properties.

### 3.1 Common Properties in Both Classes
The following properties are defined using the `@cached_property` decorator, which ensures the property is computed once and cached for subsequent access.

#### 3.1.1 `with_raw_response`
- **Purpose**: Provides access to the raw HTTP response object instead of the parsed content, useful for debugging or accessing metadata such as headers.
- **Type**:
  - For `Embeddings`: Returns `EmbeddingsWithRawResponse`
  - For `AsyncEmbeddings`: Returns `AsyncEmbeddingsWithRawResponse`
- **Usage**:
  - Allows developers to inspect raw response data (e.g., HTTP status code, headers, raw body).
  - Example: `embeddings.with_raw_response.create(...)` returns the raw response instead of a parsed `CreateEmbeddingResponse`.
- **Technical Details**:
  - The raw response is typically an `httpx.Response` object, which contains unparsed data.
  - This is useful for advanced use cases, such as logging, debugging, or custom parsing.

#### 3.1.2 `with_streaming_response`
- **Purpose**: Provides access to a streaming response, which does not eagerly read the response body, suitable for handling large responses or real-time data.
- **Type**:
  - For `Embeddings`: Returns `EmbeddingsWithStreamingResponse`
  - For `AsyncEmbeddings`: Returns `AsyncEmbeddingsWithStreamingResponse`
- **Usage**:
  - Allows developers to process the response body incrementally, reducing memory usage.
  - Example: `embeddings.with_streaming_response.create(...)` streams the response.
- **Technical Details**:
  - Streaming responses are implemented using HTTP chunked transfer encoding.
  - The response body is read incrementally, which is efficient for large embedding outputs or real-time applications.

### 3.2 Inherited Attributes
Both classes inherit attributes from their respective base classes (`SyncAPIResource` and `AsyncAPIResource`), such as:
- **Client Configuration**: Attributes like base URL, API key, timeout settings, and HTTP client configuration.
- **Request Utilities**: Methods and utilities for constructing and sending HTTP requests.

These inherited attributes are not explicitly defined in the code but are crucial for the classes' functionality.

---

## 4. Methods and Their Technical Pipeline

Both classes define a single primary method, `create`, which is responsible for generating embedding vectors. The method is implemented differently in `Embeddings` (synchronous) and `AsyncEmbeddings` (asynchronous), but the functionality and parameters are identical.

### 4.1 `create` Method Overview
- **Purpose**: Sends a request to the embeddings endpoint to generate embedding vectors for the provided input.
- **Return Type**: `CreateEmbeddingResponse`, a typed response object containing the embedding vectors and metadata.
- **HTTP Method**: POST
- **Endpoint**: `/embeddings`

### 4.2 `create` Method Signature
The method signature is identical for both classes, except for the asynchronous nature of `AsyncEmbeddings`. Below is the signature:

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
) -> CreateEmbeddingResponse:  # For Embeddings

async def create(...) -> CreateEmbeddingResponse:  # For AsyncEmbeddings
```

### 4.3 Parameters of the `create` Method
Below is a detailed explanation of each parameter, its type, constraints, and purpose.

#### 4.3.1 `input`
- **Type**: `Union[str, List[str], Iterable[int], Iterable[Iterable[int]]]`
- **Purpose**: Specifies the input data to be embedded.
- **Supported Formats**:
  - **Single String**: A single piece of text to embed (e.g., `"Hello, world!"`).
  - **List of Strings**: Multiple pieces of text to embed in a single request (e.g., `["Hello", "World"]`).
  - **Iterable of Integers**: A single sequence of token IDs (e.g., `[101, 202, 303]`).
  - **Iterable of Iterable of Integers**: Multiple sequences of token IDs (e.g., `[[101, 202], [303, 404]]`).
- **Constraints**:
  - Input must not exceed the model's maximum token limit (e.g., 8192 tokens for `text-embedding-ada-002`).
  - Input cannot be an empty string.
  - Arrays must have 2048 dimensions or less.
- **Technical Details**:
  - Text inputs are tokenized internally by the API using a tokenizer specific to the model.
  - Token IDs can be provided directly, bypassing tokenization, which is useful for performance optimization or custom preprocessing.
  - Use cases include embedding multiple documents or sentences in a single request for batch processing.

#### 4.3.2 `model`
- **Type**: `Union[str, EmbeddingModel]`
- **Purpose**: Specifies the embedding model to use for generating embeddings.
- **Supported Values**:
  - A string identifier of the model (e.g., `"text-embedding-ada-002"`).
  - An `EmbeddingModel` object, which is a typed representation of a model.
- **Technical Details**:
  - The model determines the embedding algorithm, dimensionality, and tokenization strategy.
  - Developers can retrieve available models using the API's `/models` endpoint or refer to the model's documentation.
  - Example models include `text-embedding-ada-002`, `text-embedding-3`, etc., each with different capabilities and performance characteristics.

#### 4.3.3 `dimensions`
- **Type**: `int | NotGiven`
- **Purpose**: Specifies the desired number of dimensions for the output embeddings.
- **Default Value**: `NOT_GIVEN`, meaning the model's default dimensionality is used.
- **Constraints**:
  - Only supported in models like `text-embedding-3` and later.
  - Must be a positive integer.
- **Technical Details**:
  - Reducing dimensionality can improve performance and reduce storage requirements but may lose some semantic information.
  - The API internally applies dimensionality reduction techniques (e.g., PCA) if a lower dimensionality is requested.

#### 4.3.4 `encoding_format`
- **Type**: `Literal["float", "base64"] | NotGiven`
- **Purpose**: Specifies the format in which the embedding vectors are returned.
- **Supported Values**:
  - `"float"`: Returns embeddings as a list of floating-point numbers.
  - `"base64"`: Returns embeddings as a base64-encoded string, which can be decoded into floats.
- **Default Value**: `"base64"` if not explicitly provided.
- **Technical Details**:
  - The `"float"` format is human-readable and directly usable but may increase response size.
  - The `"base64"` format is more compact and efficient for transmission but requires decoding on the client side.
  - Decoding logic is implemented in the response parser (see below).

#### 4.3.5 `user`
- **Type**: `str | NotGiven`
- **Purpose**: A unique identifier for the end-user, used for monitoring and abuse detection.
- **Default Value**: `NOT_GIVEN`, meaning no user ID is provided.
- **Technical Details**:
  - User IDs help API providers (e.g., OpenAI) track usage patterns and detect potential misuse.
  - Recommended for production applications to comply with safety best practices.

#### 4.3.6 `extra_headers`, `extra_query`, `extra_body`
- **Type**:
  - `extra_headers`: `Headers | None`
  - `extra_query`: `Query | None`
  - `extra_body`: `Body | None`
- **Purpose**: Allows developers to pass additional HTTP headers, query parameters, or body properties not covered by the standard parameters.
- **Default Value**: `None` for all.
- **Technical Details**:
  - Useful for advanced use cases, such as adding custom metadata, authentication tokens, or API-specific parameters.
  - Takes precedence over client-level configurations.

#### 4.3.7 `timeout`
- **Type**: `float | httpx.Timeout | None | NotGiven`
- **Purpose**: Overrides the default timeout for the API request.
- **Default Value**: `NOT_GIVEN`, meaning the client-level timeout is used.
- **Technical Details**:
  - Specified in seconds as a float or as an `httpx.Timeout` object for more granular control (e.g., connect timeout, read timeout).
  - Useful for handling slow networks or ensuring responsiveness in time-sensitive applications.

### 4.4 Response Parsing Logic
The `create` method includes a custom response parser to handle the embedding data based on the `encoding_format`. Below is the parser logic:

```python
def parser(obj: CreateEmbeddingResponse) -> CreateEmbeddingResponse:
    if is_given(encoding_format):
        # Don't modify the response object if a user explicitly asked for a format
        return obj

    for embedding in obj.data:
        data = cast(object, embedding.embedding)
        if not isinstance(data, str):
            continue
        if not has_numpy():
            # Use array for base64 optimization
            embedding.embedding = array.array("f", base64.b64decode(data)).tolist()
        else:
            embedding.embedding = np.frombuffer(
                base64.b64decode(data), dtype="float32"
            ).tolist()

    return obj
```

- **Purpose**: Decodes base64-encoded embeddings into a list of floats if the `encoding_format` is not explicitly specified.
- **Conditions**:
  - If `encoding_format` is explicitly provided (e.g., `"float"` or `"base64"`), the response is returned as-is.
  - If `encoding_format` is not provided, the response is assumed to be base64-encoded, and the embeddings are decoded.
- **Decoding Logic**:
  - **Without NumPy**: Uses Python's `array.array` to decode base64 data into a list of floats.
  - **With NumPy**: Uses `np.frombuffer` for efficient decoding into a NumPy array, then converts to a list.
- **Technical Details**:
  - Base64 decoding reduces the response size during transmission but requires client-side processing.
  - The parser ensures compatibility with different environments (with or without NumPy) while optimizing performance.

### 4.5 Key Differences in `create` Method Implementation
- **Synchronous (`Embeddings`)**:
  - Uses `self._post` to send a blocking HTTP POST request.
  - Returns a `CreateEmbeddingResponse` directly.
- **Asynchronous (`AsyncEmbeddings`)**:
  - Uses `await self._post` to send a non-blocking HTTP POST request.
  - Returns a `CreateEmbeddingResponse` wrapped in an `await` expression, requiring `asyncio` integration.

---

## 5. Key Differences Between `Embeddings` and `AsyncEmbeddings`

The primary differences between the two classes are summarized below:

| **Aspect**              | **Embeddings**                     | **AsyncEmbeddings**                 |
|--------------------------|------------------------------------|-------------------------------------|
| **Execution Model**      | Synchronous (blocking)            | Asynchronous (non-blocking)         |
| **Base Class**           | `SyncAPIResource`                 | `AsyncAPIResource`                 |
| **Method Signature**     | `def create(...)`                 | `async def create(...)`            |
| **HTTP Request**         | `self._post` (blocking)           | `await self._post` (non-blocking)  |
| **Use Case**             | Simple scripts, sequential tasks  | High-concurrency, I/O-bound tasks  |
| **Properties**           | `EmbeddingsWithRawResponse`, etc. | `AsyncEmbeddingsWithRawResponse`, etc. |

---

## 6. Technical Pipeline for Embedding Creation

The technical pipeline for creating embeddings using the `create` method involves several steps, executed by either the `Embeddings` or `AsyncEmbeddings` class. Below is a detailed breakdown of the pipeline.

### 6.1 Pipeline Steps

#### Step 1: Input Validation and Preparation
- **Description**: The client validates and prepares the input data for the API request.
- **Tasks**:
  - Validate the `input` format (string, list of strings, token IDs, etc.).
  - Ensure input adheres to model-specific constraints (e.g., token limits, array dimensions).
  - Convert parameters into a request body using `maybe_transform` and `embedding_create_params.EmbeddingCreateParams`.
- **Technical Details**:
  - The `maybe_transform` utility maps Python types to API-compatible JSON structures.
  - The `embedding_create_params.EmbeddingCreateParams` type ensures type safety and schema validation.

#### Step 2: Request Construction
- **Description**: The client constructs an HTTP POST request to the `/embeddings` endpoint.
- **Tasks**:
  - Set the request body with parameters (`input`, `model`, `dimensions`, etc.).
  - Apply default values (e.g., `encoding_format="base64"` if not provided).
  - Include additional headers, query parameters, or body properties if provided via `extra_headers`, `extra_query`, or `extra_body`.
  - Configure the request timeout using the `timeout` parameter.
- **Technical Details**:
  - The request is constructed using the `make_request_options` utility, which merges client-level and method-level configurations.
  - The HTTP client (likely `httpx`) is used to manage the request lifecycle.

#### Step 3: API Request Execution
- **Description**: The client sends the HTTP POST request to the API.
- **Tasks**:
  - **Synchronous (`Embeddings`)**: Executes a blocking request using `self._post`.
  - **Asynchronous (`AsyncEmbeddings`)**: Executes a non-blocking request using `await self._post`.
- **Technical Details**:
  - The request is sent to the `/embeddings` endpoint, which is processed by the API server.
  - The server tokenizes text inputs (if not already tokenized), applies the embedding model, and generates embedding vectors.

#### Step 4: Response Handling
- **Description**: The client receives and processes the API response.
- **Tasks**:
  - Parse the raw HTTP response into a `CreateEmbeddingResponse` object.
  - Apply the custom response parser to decode base64-encoded embeddings if necessary.
- **Technical Details**:
  - The response is a JSON object containing the embedding vectors, model metadata, and usage statistics.
  - The parser ensures the embeddings are in a usable format (e.g., list of floats) based on the environment (with or without NumPy).

#### Step 5: Return Result
- **Description**: The client returns the processed response to the caller.
- **Tasks**:
  - **Synchronous (`Embeddings`)**: Returns the `CreateEmbeddingResponse` directly.
  - **Asynchronous (`AsyncEmbeddings`)**: Returns the `CreateEmbeddingResponse` as an awaitable.
- **Technical Details**:
  - The `CreateEmbeddingResponse` object contains a `data` attribute, which is a list of embedding objects, each with an `embedding` vector and metadata (e.g., index, object type).

### 6.2 Pipeline Diagram
Below is a visual representation of the pipeline:

```
[Input Data] --> [Validation & Preparation] --> [Request Construction] --> [API Request] --> [Response Handling] --> [Return Result]
```

---

## 7. Detailed Explanation of Attributes and Methods

Below is a consolidated and detailed explanation of all attributes and methods in the `Embeddings` and `AsyncEmbeddings` classes, covering their purpose, implementation, and technical details.

### 7.1 Attributes

#### 7.1.1 `with_raw_response`
- **Purpose**: Provides access to the raw HTTP response for debugging or custom processing.
