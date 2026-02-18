# FineTuning API Documentation

## Core Classes

### `FineTuning`

Primary synchronous interface for fine-tuning operations.

```python
from openai import OpenAI

client = OpenAI()
fine_tuning = client.fine_tuning
```

#### Properties

| Property | Return Type | Description |
|----------|-------------|-------------|
| `jobs` | `Jobs` | Access fine-tuning jobs functionality |
| `with_raw_response` | `FineTuningWithRawResponse` | Returns raw HTTP response objects instead of parsed content |
| `with_streaming_response` | `FineTuningWithStreamingResponse` | Returns streaming response objects without eagerly reading the response body |

### `AsyncFineTuning`

Asynchronous variant of the FineTuning API for non-blocking operations.

```python
from openai import AsyncOpenAI

async_client = AsyncOpenAI()
async_fine_tuning = async_client.fine_tuning
```

#### Properties

| Property | Return Type | Description |
|----------|-------------|-------------|
| `jobs` | `AsyncJobs` | Access asynchronous fine-tuning jobs functionality |
| `with_raw_response` | `AsyncFineTuningWithRawResponse` | Returns raw HTTP response objects instead of parsed content |
| `with_streaming_response` | `AsyncFineTuningWithStreamingResponse` | Returns streaming response objects without eagerly reading the response body |

## Response Handling Classes

### `FineTuningWithRawResponse`

Provides access to raw HTTP response data in synchronous operations.

```python
raw_response = client.fine_tuning.with_raw_response.jobs.create(...)
status_code = raw_response.status_code
headers = raw_response.headers
```

#### Properties

| Property | Return Type | Description |
|----------|-------------|-------------|
| `jobs` | `JobsWithRawResponse` | Access fine-tuning jobs with raw response handling |

### `AsyncFineTuningWithRawResponse`

Provides access to raw HTTP response data in asynchronous operations.

```python
raw_response = await async_client.fine_tuning.with_raw_response.jobs.create(...)
status_code = raw_response.status_code
headers = raw_response.headers
```

#### Properties

| Property | Return Type | Description |
|----------|-------------|-------------|
| `jobs` | `AsyncJobsWithRawResponse` | Access asynchronous fine-tuning jobs with raw response handling |

### `FineTuningWithStreamingResponse`

Enables streaming responses for synchronous operations without eagerly reading response bodies.

```python
with client.fine_tuning.with_streaming_response.jobs.create(...) as response:
    # Process streaming response
    for chunk in response.iter_bytes():
        # Handle each chunk
```

#### Properties

| Property | Return Type | Description |
|----------|-------------|-------------|
| `jobs` | `JobsWithStreamingResponse` | Access fine-tuning jobs with streaming response handling |

### `AsyncFineTuningWithStreamingResponse`

Enables streaming responses for asynchronous operations without eagerly reading response bodies.

```python
async with async_client.fine_tuning.with_streaming_response.jobs.create(...) as response:
    # Process streaming response
    async for chunk in response.aiter_bytes():
        # Handle each chunk
```

#### Properties

| Property | Return Type | Description |
|----------|-------------|-------------|
| `jobs` | `AsyncJobsWithStreamingResponse` | Access asynchronous fine-tuning jobs with streaming response handling |

## Usage Examples

### Basic Fine-tuning Job Creation

```python
from openai import OpenAI

client = OpenAI()

# Create a fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file="file-abc123",
    model="gpt-3.5-turbo"
)
```

### Asynchronous Fine-tuning with Raw Response

```python
import asyncio
from openai import AsyncOpenAI

async def create_job():
    client = AsyncOpenAI()
    
    response = await client.fine_tuning.with_raw_response.jobs.create(
        training_file="file-abc123",
        model="gpt-3.5-turbo"
    )
    
    print(f"Status: {response.status_code}")
    print(f"Headers: {response.headers}")
    print(f"Job ID: {response.parsed.id}")
    
asyncio.run(create_job())
```

### Streaming Response Handling

```python
from openai import OpenAI

client = OpenAI()

with client.fine_tuning.with_streaming_response.jobs.create(
    training_file="file-abc123",
    model="gpt-3.5-turbo"
) as response:
    # Process response without eagerly loading entire response body
    job = response.parse()
    print(f"Job ID: {job.id}")
```