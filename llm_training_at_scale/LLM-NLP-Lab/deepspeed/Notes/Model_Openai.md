# Models API Documentation

## Main Classes

### `Models` (Synchronous)

A synchronous API resource class for interacting with OpenAI models.

```python
class Models(SyncAPIResource):
    # Methods for model management
```

### `AsyncModels` (Asynchronous)

An asynchronous counterpart to the Models class for asynchronous operations.

```python
class AsyncModels(AsyncAPIResource):
    # Asynchronous methods for model management
```

## Response Handling Properties

### Synchronous Response Properties

#### `with_raw_response`
Returns a `ModelsWithRawResponse` instance that provides access to raw HTTP response data.

```python
models = client.models
raw_response = models.with_raw_response.retrieve("text-davinci-003")
# Access headers, status code, etc.
```

#### `with_streaming_response`
Returns a `ModelsWithStreamingResponse` instance that allows for streaming responses without eagerly reading the response body.

```python
models = client.models
streaming_response = models.with_streaming_response.retrieve("text-davinci-003")
# Use streaming response pattern
```

### Asynchronous Response Properties

#### `with_raw_response`
Returns an `AsyncModelsWithRawResponse` instance for asynchronous raw response handling.

#### `with_streaming_response`
Returns an `AsyncModelsWithStreamingResponse` instance for asynchronous streaming response handling.

## Core Methods

### Retrieve Model

Fetches detailed information about a specific model.

#### Synchronous
```python
def retrieve(
    self,
    model: str,
    *,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> Model:
    # Returns a Model object with information about the specified model
```

#### Asynchronous
```python
async def retrieve(
    self,
    model: str,
    *,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> Model:
    # Asynchronously returns a Model object
```

**Parameters:**
- `model`: Required string identifier of the model
- `extra_headers`: Optional additional HTTP headers
- `extra_query`: Optional additional query parameters
- `extra_body`: Optional additional JSON body properties
- `timeout`: Optional request timeout override

**Raises:**
- `ValueError`: If model parameter is empty

**Returns:**
- `Model` object containing model information

### List Models

Retrieves a list of available models with basic information.

#### Synchronous
```python
def list(
    self,
    *,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> SyncPage[Model]:
    # Returns a paginated list of Model objects
```

#### Asynchronous
```python
def list(
    self,
    *,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> AsyncPaginator[Model, AsyncPage[Model]]:
    # Returns an async paginator for models
```

**Parameters:**
- `extra_headers`: Optional additional HTTP headers
- `extra_query`: Optional additional query parameters
- `extra_body`: Optional additional JSON body properties
- `timeout`: Optional request timeout override

**Returns:**
- Paginated list of `Model` objects (synchronous or asynchronous depending on class)

### Delete Model

Deletes a fine-tuned model. Requires Owner role in the organization.

#### Synchronous
```python
def delete(
    self,
    model: str,
    *,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> ModelDeleted:
    # Returns confirmation of model deletion
```

#### Asynchronous
```python
async def delete(
    self,
    model: str,
    *,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> ModelDeleted:
    # Asynchronously returns confirmation of model deletion
```

**Parameters:**
- `model`: Required string identifier of the model to delete
- `extra_headers`: Optional additional HTTP headers
- `extra_query`: Optional additional query parameters
- `extra_body`: Optional additional JSON body properties
- `timeout`: Optional request timeout override

**Raises:**
- `ValueError`: If model parameter is empty

**Returns:**
- `ModelDeleted` object confirming the deletion

## Response Wrapper Classes

### `ModelsWithRawResponse`
Wrapper that enhances `Models` methods to return raw response objects.

### `AsyncModelsWithRawResponse`
Asynchronous wrapper that enhances `AsyncModels` methods to return raw response objects.

### `ModelsWithStreamingResponse`
Wrapper that enhances `Models` methods with streaming response capabilities.

### `AsyncModelsWithStreamingResponse` 
Asynchronous wrapper that enhances `AsyncModels` methods with streaming response capabilities.