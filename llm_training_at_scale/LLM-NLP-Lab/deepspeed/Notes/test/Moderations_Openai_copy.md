**Moderations API Documentation**
=====================================

**Overview**
------------

The Moderations API is used to classify text and/or image inputs as potentially harmful. This API provides a synchronous and asynchronous interface for creating moderation requests.

**Main API Classes**
--------------------

*   **[`Moderations`](#moderations-class)**: Synchronous API for moderation requests.
*   **[`AsyncModerations`](#asyncmoderations-class)**: Asynchronous API for moderation requests.

**Moderations Class**
---------------------

### `class Moderations(SyncAPIResource)`

#### **Properties**

*   **`with_raw_response`** (`ModerationsWithRawResponse`): **[HIGHLIGHTED]** Returns the raw response object instead of parsed content for any HTTP method call.
    *   Use this property as a prefix for any HTTP method call to access raw response data (e.g., headers).
    *   [Learn more about accessing raw response data](https://www.github.com/openai/openai-python#accessing-raw-response-data-eg-headers)
*   **`with_streaming_response`** (`ModerationsWithStreamingResponse`): **[HIGHLIGHTED]** Alternative to `with_raw_response` that doesn't eagerly read the response body.
    *   [Learn more about streaming responses](https://www.github.com/openai/openai-python#with_streaming_response)

#### **Methods**

##### **`create` Method**
```python
def create(
    self,
    *,
    input: Union[str, List[str], Iterable[ModerationMultiModalInputParam]],
    model: Union[str, ModerationModel] | NotGiven = NOT_GIVEN,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> ModerationCreateResponse:
```
**[HIGHLIGHTED MAIN API]** Classifies if text and/or image inputs are potentially harmful.

**Parameters:**

*   **`input`** (`Union[str, List[str], Iterable[ModerationMultiModalInputParam]]`): **Required** Input (or inputs) to classify. Can be:
    *   A single string
    *   An array of strings
    *   An array of multi-modal input objects (similar to other models)
*   **`model`** (`Union[str, ModerationModel] | NotGiven`): The content moderation model to use. Default is `NOT_GIVEN`.
    *   Learn more in the [moderation guide](https://platform.openai.com/docs/guides/moderation)
    *   Available models: [see models documentation](https://platform.openai.com/docs/models#moderation)
*   **`extra_headers`** (`Headers | None`): Send extra headers with the request.
*   **`extra_query`** (`Query | None`): Add additional query parameters to the request.
*   **`extra_body`** (`Body | None`): Add additional JSON properties to the request body.
*   **`timeout`** (`float | httpx.Timeout | None | NotGiven`): Override the client-level default timeout for this request (in seconds).

**Returns:** `ModerationCreateResponse`

**Example Usage:**
```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")
moderations = client.moderations

# Classify a single text input
response = moderations.create(input="Hello, world!")
print(response)

# Classify multiple text inputs
response = moderations.create(input=["Hello", "world!"])
print(response)

# Using a specific moderation model
response = moderations.create(input="Hello, world!", model="text-moderation-stable")
print(response)
```

**AsyncModerations Class**
-------------------------

### `class AsyncModerations(AsyncAPIResource)`

#### **Properties**

*   **`with_raw_response`** (`AsyncModerationsWithRawResponse`): **[HIGHLIGHTED]** Asynchronous version of `Moderations.with_raw_response`.
*   **`with_streaming_response`** (`AsyncModerationsWithStreamingResponse`): **[HIGHLIGHTED]** Asynchronous version of `Moderations.with_streaming_response`.

#### **Methods**

##### **`create` Method**
```python
async def create(
    self,
    *,
    input: Union[str, List[str], Iterable[ModerationMultiModalInputParam]],
    model: Union[str, ModerationModel] | NotGiven = NOT_GIVEN,
    extra_headers: Headers | None = None,
    extra_query: Query | None = None,
    extra_body: Body | None = None,
    timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> ModerationCreateResponse:
```
**[HIGHLIGHTED MAIN API]** Asynchronous version of `Moderations.create`. Classifies if text and/or image inputs are potentially harmful.

**Parameters:** (Same as `Moderations.create`)

**Returns:** `ModerationCreateResponse`

**Example Usage:**
```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="YOUR_API_KEY")
moderations = client.moderations

async def main():
    # Classify a single text input
    response = await moderations.create(input="Hello, world!")
    print(response)

    # Classify multiple text inputs
    response = await moderations.create(input=["Hello", "world!"])
    print(response)

    # Using a specific moderation model
    response = await moderations.create(input="Hello, world!", model="text-moderation-stable")
    print(response)

asyncio.run(main())
```

**Raw Response Classes**
-------------------------

These classes provide access to the raw HTTP response for moderation requests.

### `class ModerationsWithRawResponse`

*   **`create` Method**: Wraps `Moderations.create` to return the raw response object.

**Example Usage:**
```python
moderations_raw = moderations.with_raw_response
response = moderations_raw.create(input="Hello, world!")
print(response.headers)  # Access raw response headers
print(response.json())   # Parse JSON response body
```

### `class AsyncModerationsWithRawResponse`

*   **`create` Method**: Asynchronous version of `ModerationsWithRawResponse.create`.

**Example Usage:**
```python
moderations_raw = moderations.with_raw_response
response = await moderations_raw.create(input="Hello, world!")
print(response.headers)  # Access raw response headers
print(response.json())   # Parse JSON response body
```

**Streaming Response Classes**
------------------------------

These classes provide streamed responses for moderation requests, useful for large responses.

### `class ModerationsWithStreamingResponse`

*   **`create` Method**: Wraps `Moderations.create` to stream the response body.

**Example Usage:**
```python
moderations_stream = moderations.with_streaming_response
with moderations_stream.create(input="Hello, world!") as response:
    for chunk in response.iter_bytes():
        print(chunk.decode())  # Process streamed response chunks
```

### `class AsyncModerationsWithStreamingResponse`

*   **`create` Method**: Asynchronous version of `ModerationsWithStreamingResponse.create`.

**Example Usage:**
```python
moderations_stream = moderations.with_streaming_response
async with moderations_stream.create(input="Hello, world!") as response:
    async for chunk in response.aiter_bytes():
        print(chunk.decode())  # Process streamed response chunks
```

**Response Type**
------------------

### `class ModerationCreateResponse`

The response type for `Moderations.create` and `AsyncModerations.create` methods.

**Attributes:**

*   (Documentation for `ModerationCreateResponse` attributes is not provided in the given API code. Refer to the OpenAI API documentation for details on the response structure.)

**Error Handling**
------------------

*   Refer to the OpenAI API documentation for error types and handling strategies.

**Additional Resources**
-------------------------

*   [Moderation Guide](https://platform.openai.com/docs/guides/moderation)
*   [Available Moderation Models](https://platform.openai.com/docs/models#moderation)
*   [OpenAI Python Library Documentation](https://www.github.com/openai/openai-python)