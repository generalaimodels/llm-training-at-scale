**Models API Documentation**
=====================================

**Overview**
------------

The Models API provides programmatic access to manage and interact with machine learning models. This API allows developers to retrieve, list, and delete models, as well as access raw response data and streaming responses.

**Main API Classes**
--------------------

*   [**`Models`**](models-class) (Sync API Resource)
*   [**`AsyncModels`**](asyncmodels-class) (Async API Resource)

**`Models` Class (Sync API Resource)**
--------------------------------------

### Description

The `Models` class provides synchronous access to model management functionality. It allows you to retrieve, list, and delete models.

### Attributes

*   **`with_raw_response`** ([**`ModelsWithRawResponse`**](#modelswithrawresponse-class)): Returns the raw response object instead of parsed content for any HTTP method call.
*   **`with_streaming_response`** ([**`ModelsWithStreamingResponse`**](#modelswithstreamingresponse-class)): Returns a streaming response object for any HTTP method call.

### Methods

#### **`retrieve(model: str, **kwargs) -> Model`**

**Highlights:**

*   **Required Parameter:** `model` (str) - The ID of the model to retrieve.
*   **Returns:** `Model` object containing basic information about the model.

 Retrieves a model instance, providing basic information about the model such as the owner and permissioning.

**Parameters:**

| Name | Type | Description | Required |
| --- | --- | --- | --- |
| `model` | `str` | The ID of the model to retrieve. | Yes |
| `extra_headers` | `Headers \| None` | Send extra headers. | No |
| `extra_query` | `Query \| None` | Add additional query parameters to the request. | No |
| `extra_body` | `Body \| None` | Add additional JSON properties to the request. | No |
| `timeout` | `float \| httpx.Timeout \| None \| NotGiven` | Override the client-level default timeout for this request, in seconds. | No |

**Raises:**

*   `ValueError`: If `model` is empty.

**Example:**
```python
from openai import OpenAI

client = OpenAI()
models = client.models

model_id = "my_model"
model_info = models.retrieve(model_id)
print(model_info)
```

#### **`list(**kwargs) -> SyncPage[Model]`**

**Highlights:**

*   **Returns:** A page of `Model` objects containing basic information about each model.

Lists the currently available models, and provides basic information about each one such as the owner and availability.

**Parameters:**

| Name | Type | Description | Required |
| --- | --- | --- | --- |
| `extra_headers` | `Headers \| None` | Send extra headers. | No |
| `extra_query` | `Query \| None` | Add additional query parameters to the request. | No |
| `extra_body` | `Body \| None` | Add additional JSON properties to the request. | No |
| `timeout` | `float \| httpx.Timeout \| None \| NotGiven` | Override the client-level default timeout for this request, in seconds. | No |

**Example:**
```python
from openai import OpenAI

client = OpenAI()
models = client.models

available_models = models.list()
for model in available_models:
    print(model)
```

#### **`delete(model: str, **kwargs) -> ModelDeleted`**

**Highlights:**

*   **Required Parameter:** `model` (str) - The ID of the model to delete.
*   **Returns:** `ModelDeleted` object indicating the model deletion status.
*   **Requires Owner Role**: You must have the Owner role in your organization to delete a model.

Deletes a fine-tuned model.

**Parameters:**

| Name | Type | Description | Required |
| --- | --- | --- | --- |
| `model` | `str` | The ID of the model to delete. | Yes |
| `extra_headers` | `Headers \| None` | Send extra headers. | No |
| `extra_query` | `Query \| None` | Add additional query parameters to the request. | No |
| `extra_body` | `Body \| None` | Add additional JSON properties to the request. | No |
| `timeout` | `float \| httpx.Timeout \| None \| NotGiven` | Override the client-level default timeout for this request, in seconds. | No |

**Raises:**

*   `ValueError`: If `model` is empty.

**Example:**
```python
from openai import OpenAI

client = OpenAI()
models = client.models

model_id = "my_model"
deletion_status = models.delete(model_id)
print(deletion_status)
```

**`AsyncModels` Class (Async API Resource)**
--------------------------------------------

### Description

The `AsyncModels` class provides asynchronous access to model management functionality. It allows you to retrieve, list, and delete models asynchronously.

### Attributes

*   **`with_raw_response`** ([**`AsyncModelsWithRawResponse`**](#asyncmodelswithrawresponse-class)): Returns the raw response object instead of parsed content for any HTTP method call.
*   **`with_streaming_response`** ([**`AsyncModelsWithStreamingResponse`**](#asyncmodelswithstreamingresponse-class)): Returns a streaming response object for any HTTP method call.

### Methods

#### **`async retrieve(model: str, **kwargs) -> Model`**

**Highlights:**

*   **Required Parameter:** `model` (str) - The ID of the model to retrieve.
*   **Returns:** `Model` object containing basic information about the model.

Asynchronously retrieves a model instance, providing basic information about the model such as the owner and permissioning.

**Parameters:**

| Name | Type | Description | Required |
| --- | --- | --- | --- |
| `model` | `str` | The ID of the model to retrieve. | Yes |
| `extra_headers` | `Headers \| None` | Send extra headers. | No |
| `extra_query` | `Query \| None` | Add additional query parameters to the request. | No |
| `extra_body` | `Body \| None` | Add additional JSON properties to the request. | No |
| `timeout` | `float \| httpx.Timeout \| None \| NotGiven` | Override the client-level default timeout for this request, in seconds. | No |

**Raises:**

*   `ValueError`: If `model` is empty.

**Example:**
```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()
models = client.models

async def main():
    model_id = "my_model"
    model_info = await models.retrieve(model_id)
    print(model_info)

asyncio.run(main())
```

#### **`list(**kwargs) -> AsyncPaginator[Model, AsyncPage[Model]]`**

**Highlights:**

*   **Returns:** An asynchronous paginator of `Model` objects containing basic information about each model.

Asynchronously lists the currently available models, and provides basic information about each one such as the owner and availability.

**Parameters:**

| Name | Type | Description | Required |
| --- | --- | --- | --- |
| `extra_headers` | `Headers \| None` | Send extra headers. | No |
| `extra_query` | `Query \| None` | Add additional query parameters to the request. | No |
| `extra_body` | `Body \| None` | Add additional JSON properties to the request. | No |
| `timeout` | `float \| httpx.Timeout \| None \| NotGiven` | Override the client-level default timeout for this request, in seconds. | No |

**Example:**
```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()
models = client.models

async def main():
    async for model in models.list():
        print(model)

asyncio.run(main())
```

#### **`async delete(model: str, **kwargs) -> ModelDeleted`**

**Highlights:**

*   **Required Parameter:** `model` (str) - The ID of the model to delete.
*   **Returns:** `ModelDeleted` object indicating the model deletion status.
*   **Requires Owner Role**: You must have the Owner role in your organization to delete a model.

Asynchronously deletes a fine-tuned model.

**Parameters:**

| Name | Type | Description | Required |
| --- | --- | --- | --- |
| `model` | `str` | The ID of the model to delete. | Yes |
| `extra_headers` | `Headers \| None` | Send extra headers. | No |
| `extra_query` | `Query \| None` | Add additional query parameters to the request. | No |
| `extra_body` | `Body \| None` | Add additional JSON properties to the request. | No |
| `timeout` | `float \| httpx.Timeout \| None \| NotGiven` | Override the client-level default timeout for this request, in seconds. | No |

**Raises:**

*   `ValueError`: If `model` is empty.

**Example:**
```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()
models = client.models

async def main():
    model_id = "my_model"
    deletion_status = await models.delete(model_id)
    print(deletion_status)

asyncio.run(main())
```

**`ModelsWithRawResponse` Class**
---------------------------------

### Description

The `ModelsWithRawResponse` class wraps the `Models` class to provide raw response objects instead of parsed content for any HTTP method call.

### Attributes

*   **`retrieve`**: Wraps `Models.retrieve` to return the raw response object.
*   **`list`**: Wraps `Models.list` to return the raw response object.
*   **`delete`**: Wraps `Models.delete` to return the raw response object.

**Example:**
```python
from openai import OpenAI

client = OpenAI()
models = client.models.with_raw_response

model_id = "my_model"
raw_response = models.retrieve(model_id)
print(raw_response.headers)
print(raw_response.text)
```

**`AsyncModelsWithRawResponse` Class**
--------------------------------------

### Description

The `AsyncModelsWithRawResponse` class wraps the `AsyncModels` class to provide raw response objects instead of parsed content for any HTTP method call asynchronously.

### Attributes

*   **`retrieve`**: Wraps `AsyncModels.retrieve` to return the raw response object.
*   **`list`**: Wraps `AsyncModels.list` to return the raw response object.
*   **`delete`**: Wraps `AsyncModels.delete` to return the raw response object.

**Example:**
```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()
models = client.models.with_raw_response

async def main():
    model_id = "my_model"
    raw_response = await models.retrieve(model_id)
    print(raw_response.headers)
    print(raw_response.text)

asyncio.run(main())
```

**`ModelsWithStreamingResponse` Class**
---------------------------------------

### Description

The `ModelsWithStreamingResponse` class wraps the `Models` class to provide streaming response objects for any HTTP method call.

### Attributes

*   **`retrieve`**: Wraps `Models.retrieve` to return a streaming response object.
*   **`list`**: Wraps `Models.list` to return a streaming response object.
*   **`delete`**: Wraps `Models.delete` to return a streaming response object.

**Example:**
```python
from openai import OpenAI

client = OpenAI()
models = client.models.with_streaming_response

model_id = "my_model"
streaming_response = models.retrieve(model_id)
for chunk in streaming_response:
    print(chunk)
```

**`AsyncModelsWithStreamingResponse` Class**
--------------------------------------------

### Description

The `AsyncModelsWithStreamingResponse` class wraps the `AsyncModels` class to provide streaming response objects for any HTTP method call asynchronously.

### Attributes

*   **`retrieve`**: Wraps `AsyncModels.retrieve` to return a streaming response object.
*   **`list`**: Wraps `AsyncModels.list` to return a streaming response object.
*   **`delete`**: Wraps `AsyncModels.delete` to return a streaming response object.

**Example:**
```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()
models = client.models.with_streaming_response

async def main():
    model_id = "my_model"
    streaming_response = await models.retrieve(model_id)
    async for chunk in streaming_response:
        print(chunk)

asyncio.run(main())
```

**Error Handling**
------------------

*   **`ValueError`**: Raised when required parameters are missing or invalid.
*   **`httpx.HTTPError`**: Raised for HTTP errors (4xx/5xx status codes).

**Type Definitions**
--------------------

*   **`Headers`**: `Dict[str, str]`
*   **`Query`**: `Dict[str, str]`
*   **`Body`**: `Dict[str, Any]`
*   **`NotGiven`**: A sentinel value indicating a parameter was not provided.
*   **`Model`**: The model instance type returned by `retrieve` and `list` methods.
*   **`ModelDeleted`**: The model deletion status type returned by `delete` method.
*   **`SyncPage[T]`**: A synchronous page of items of type `T`.
*   **`AsyncPage[T]`**: An asynchronous page of items of type `T`.
*   **`AsyncPaginator[T, PageT]`**: An asynchronous paginator of items of type `T` with pages of type `PageT`. 

**Links**
---------

*   [Accessing Raw Response Data](https://www.github.com/openai/openai-python#accessing-raw-response-data-eg-headers)
*   [Streaming Responses](https://www.github.com/openai/openai-python#with_streaming_response) 

By following this documentation, developers should be able to easily understand and utilize the Models API for managing machine learning models. The highlighted main API classes (`Models` and `AsyncModels`) and their methods provide clear entry points for interacting with the API. Additionally, the provided examples demonstrate how to handle various scenarios, including error handling and working with raw and streaming responses.