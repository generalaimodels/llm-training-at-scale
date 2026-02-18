# `Completions` Class API Documentation

## Class Definition
```python
class Completions(SyncAPIResource):
```

The `Completions` class provides synchronous access to OpenAI's text completion API for generating text based on provided prompts.

## Properties

### `with_raw_response`
```python
@cached_property
def with_raw_response(self) -> CompletionsWithRawResponse:
```
Returns a wrapper that provides access to the raw HTTP response object instead of the parsed content.

### `with_streaming_response`
```python
@cached_property
def with_streaming_response(self) -> CompletionsWithStreamingResponse:
```
Returns a wrapper that doesn't eagerly read the response body, suitable for large responses.

## Methods

### `create`
```python
def create(
    self,
    *,
    model: Union[str, Literal["gpt-3.5-turbo-instruct", "davinci-002", "babbage-002"]],
    prompt: Union[str, List[str], Iterable[int], Iterable[Iterable[int]], None],
    # Additional parameters...
) -> Completion | Stream[Completion]:
```

Creates a completion for the provided prompt with specified parameters.

#### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `Union[str, Literal["gpt-3.5-turbo-instruct", "davinci-002", "babbage-002"]]` | ID of the model to use |
| `prompt` | `Union[str, List[str], Iterable[int], Iterable[Iterable[int]], None]` | The prompt(s) to generate completions for |

#### Optional Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `best_of` | `Optional[int]` | Number of completions generated server-side, returning only the "best" one |
| `echo` | `Optional[bool]` | Whether to echo back the prompt in addition to the completion |
| `frequency_penalty` | `Optional[float]` | Value between -2.0 and 2.0 to penalize frequent tokens |
| `logit_bias` | `Optional[Dict[str, int]]` | Modifies likelihood of specified tokens appearing in completion |
| `logprobs` | `Optional[int]` | Number of most likely tokens to include probabilities for (max 5) |
| `max_tokens` | `Optional[int]` | Maximum number of tokens to generate |
| `n` | `Optional[int]` | Number of completions to generate for each prompt |
| `presence_penalty` | `Optional[float]` | Value between -2.0 and 2.0 to penalize tokens based on presence |
| `seed` | `Optional[int]` | For deterministic sampling when possible |
| `stop` | `Union[Optional[str], List[str], None]` | Up to 4 sequences where generation will stop |
| `stream` | `Optional[bool]` | Whether to stream back partial progress |
| `stream_options` | `Optional[ChatCompletionStreamOptionsParam]` | Options for streaming response |
| `suffix` | `Optional[str]` | Text to append after completion (only for gpt-3.5-turbo-instruct) |
| `temperature` | `Optional[float]` | Sampling temperature between 0 and 2 |
| `top_p` | `Optional[float]` | Alternative sampling method using nucleus sampling |
| `user` | `str` | Unique identifier for end-user for monitoring |

#### Request Customization Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `extra_headers` | `Headers \| None` | Additional headers to send with the request |
| `extra_query` | `Query \| None` | Additional query parameters |
| `extra_body` | `Body \| None` | Additional JSON properties for the request body |
| `timeout` | `float \| httpx.Timeout \| None` | Override client-level timeout |

#### Return Types

- Standard mode: Returns a `Completion` object
- Streaming mode: Returns a `Stream[Completion]` when `stream=True`

## Usage Examples

### Basic Completion
```python
from openai import OpenAI

client = OpenAI()
response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Write a tagline for an ice cream shop.",
    max_tokens=50
)
print(response.choices[0].text)
```

### Streaming Completion
```python
from openai import OpenAI

client = OpenAI()
stream = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Write a short story about AI.",
    max_tokens=500,
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].text, end="")
```

### Controlling Output with Parameters
```python
from openai import OpenAI

client = OpenAI()
response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="List 5 ideas for mobile apps.",
    temperature=0.7,
    max_tokens=100,
    top_p=0.9,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    stop=["\n\n"]
)
print(response.choices[0].text)
```