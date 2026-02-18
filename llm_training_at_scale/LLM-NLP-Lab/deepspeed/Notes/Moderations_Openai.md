# Moderations API Documentation

## Main Classes

### `Moderations`

A synchronous API resource class for classifying potentially harmful content.

```python
from openai import SyncAPIResource

class Moderations(SyncAPIResource):
    # Implementation methods
```

### `AsyncModerations`

An asynchronous version of the Moderations API resource.

```python
from openai import AsyncAPIResource

class AsyncModerations(AsyncAPIResource):
    # Implementation methods
```

## Properties

### `with_raw_response`

Returns a wrapper that provides access to raw HTTP response data.

```python
# Synchronous usage
moderations = Moderations()
raw_response = moderations.with_raw_response.create(input="Content to check")
```

```python
# Asynchronous usage
async_moderations = AsyncModerations()
raw_response = await async_moderations.with_raw_response.create(input="Content to check")
```

### `with_streaming_response`

Returns a wrapper that provides streaming response functionality without eagerly reading the response body.

```python
# Synchronous usage
moderations = Moderations()
streaming_response = moderations.with_streaming_response.create(input="Content to check")
```

```python
# Asynchronous usage
async_moderations = AsyncModerations()
streaming_response = await async_moderations.with_streaming_response.create(input="Content to check")
```

## Methods

### `create`

Classifies if text and/or image inputs are potentially harmful.

#### Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `input` | `Union[str, List[str], Iterable[ModerationMultiModalInputParam]]` | Input to classify. Can be a single string, array of strings, or array of multi-modal input objects. | Yes |
| `model` | `Union[str, ModerationModel] \| NotGiven` | The content moderation model to use. | No |
| `extra_headers` | `Headers \| None` | Additional headers for the request. | No |
| `extra_query` | `Query \| None` | Additional query parameters. | No |
| `extra_body` | `Body \| None` | Additional JSON properties for the request body. | No |
| `timeout` | `float \| httpx.Timeout \| None \| NotGiven` | Override client-level default timeout. | No |

#### Return Value

Returns a `ModerationCreateResponse` object containing moderation results.

#### Synchronous Usage

```python
from openai import Moderations

moderations = Moderations()
response = moderations.create(
    input="I want to harm someone.",
    model="text-moderation-latest"
)
```

#### Asynchronous Usage

```python
from openai import AsyncModerations
import asyncio

async def check_content():
    async_moderations = AsyncModerations()
    response = await async_moderations.create(
        input="I want to harm someone.",
        model="text-moderation-latest"
    )
    return response

result = asyncio.run(check_content())
```

## Response Wrapper Classes

### `ModerationsWithRawResponse`

Wrapper class that returns raw HTTP response data instead of parsed content.

### `AsyncModerationsWithRawResponse`

Asynchronous version of the raw response wrapper.

### `ModerationsWithStreamingResponse`

Wrapper class that provides streaming functionality without eagerly reading the response body.

### `AsyncModerationsWithStreamingResponse`

Asynchronous version of the streaming response wrapper.

## Multi-Modal Input Example

```python
from openai import Moderations

moderations = Moderations()
response = moderations.create(
    input=[
        {"type": "text", "text": "Is this content safe?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ],
    model="text-moderation-latest"
)
```