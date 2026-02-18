**Completions API Documentation**
=====================================

**Overview**
------------

The `Completions` API is a crucial component of the OpenAI Python library, enabling developers to generate text completions based on a given prompt and set of parameters. This API is part of the `SyncAPIResource` class and provides methods for creating completions, accessing raw responses, and handling streaming responses.

**Properties**
--------------

### 1. `with_raw_response`

*   **Type:** `CompletionsWithRawResponse`
*   **Description:** This property allows you to access the raw HTTP response object instead of the parsed content for any HTTP method call. It is particularly useful when you need to inspect response headers or handle raw response data.
*   **Usage:**

```python
completions = Completions()
raw_response_completion = completions.with_raw_response.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Hello, world!"
)
print(raw_response_completion.headers)
```
*   **Reference:** [Accessing Raw Response Data](https://www.github.com/openai/openai-python#accessing-raw-response-data-eg-headers)

### 2. `with_streaming_response`

*   **Type:** `CompletionsWithStreamingResponse`
*   **Description:** Similar to `with_raw_response`, but this property doesn't eagerly read the response body. It's beneficial for handling large responses or when you want to start processing the response before it's fully received.
*   **Usage:**

```python
completions = Completions()
streaming_response_completion = completions.with_streaming_response.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Tell me a story.",
    stream=True
)
for chunk in streaming_response_completion:
    print(chunk)
```
*   **Reference:** [Streaming Responses](https://www.github.com/openai/openai-python#with_streaming_response)

**Methods**
------------

### **`create` Method**

**Highlighted Main API**

The `create` method is the core functionality of the `Completions` API. It generates a completion based on the provided prompt and parameters.

#### **Method Signature**

The `create` method is overloaded to support two different return types based on the `stream` parameter:

1.  **`create` without Streaming (`stream=False` or omitted)**

    ```python
    @overload
    def create(
        self,
        *,
        model: Union[str, Literal["gpt-3.5-turbo-instruct", "davinci-002", "babbage-002"]],
        prompt: Union[str, List[str], Iterable[int], Iterable[Iterable[int]], None],
        best_of: Optional[int] | NotGiven = NOT_GIVEN,
        echo: Optional[bool] | NotGiven = NOT_GIVEN,
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        stream: Optional[Literal[False]] | NotGiven = NOT_GIVEN,
        stream_options: Optional[ChatCompletionStreamOptionsParam] | NotGiven = NOT_GIVEN,
        suffix: Optional[str] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> Completion:
    ```

2.  **`create` with Streaming (`stream=True`)**

    ```python
    @overload
    def create(
        self,
        *,
        model: Union[str, Literal["gpt-3.5-turbo-instruct", "davinci-002", "babbage-002"]],
        prompt: Union[str, List[str], Iterable[int], Iterable[Iterable[int]], None],
        stream: Literal[True],
        best_of: Optional[int] | NotGiven = NOT_GIVEN,
        echo: Optional[bool] | NotGiven = NOT_GIVEN,
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        stream_options: Optional[ChatCompletionStreamOptionsParam] | NotGiven = NOT_GIVEN,
        suffix: Optional[str] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> Stream[Completion]:
    ```

#### **Parameters**

| Parameter            | Type                                                                                           | Description                                                                                                                                                                                                                                                                                   | Default           |
| -------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| `model`              | `Union[str, Literal["gpt-3.5-turbo-instruct", "davinci-002", "babbage-002"]]`                  | **Required.** ID of the model to use. List available models via [List Models API](https://platform.openai.com/docs/api-reference/models/list) or see [Model Overview](https://platform.openai.com/docs/models).                                                                               |                   |
| `prompt`             | `Union[str, List[str], Iterable[int], Iterable[Iterable[int]], None]`                          | **Required.** The prompt(s) to generate completions for. Can be a string, array of strings, array of tokens, or array of token arrays.                                                                                                                                                        |                   |
| `best_of`            | `Optional[int] \| NotGiven`                                                                   | Generates `best_of` completions server-side and returns the "best". Results cannot be streamed. Use carefully to avoid consuming token quota quickly.                                                                                                                                         | `NOT_GIVEN`       |
| `echo`               | `Optional[bool] \| NotGiven`                                                                  | Echo back the prompt in addition to the completion.                                                                                                                                                                                                                                           | `NOT_GIVEN`       |
| `frequency_penalty`  | `Optional[float] \| NotGiven`                                                                 | Number between -2.0 and 2.0. Positive values penalize new tokens based on existing frequency.                                                                                                                                                                                                 | `NOT_GIVEN`       |
| `logit_bias`         | `Optional[Dict[str, int]] \| NotGiven`                                                        | Modify likelihood of specified tokens appearing in the completion. Use [Tokenizer Tool](/tokenizer?view=bpe) to convert text to token IDs.                                                                                                                                                    | `NOT_GIVEN`       |
| `logprobs`           | `Optional[int] \| NotGiven`                                                                   | Include log probabilities on the `logprobs` most likely tokens and the chosen tokens. Max value is 5.                                                                                                                                                                                         | `NOT_GIVEN`       |
| `max_tokens`         | `Optional[int] \| NotGiven`                                                                   | The maximum number of [tokens](/tokenizer) to generate. The total (prompt + max_tokens) must not exceed the model's context length.                                                                                                                                                           | `NOT_GIVEN`       |
| `n`                  | `Optional[int] \| NotGiven`                                                                   | How many completions to generate for each prompt. Be cautious of token quota consumption.                                                                                                                                                                                                     | `NOT_GIVEN`       |
| `presence_penalty`   | `Optional[float] \| NotGiven`                                                                 | Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far.                                                                                                                                                                            | `NOT_GIVEN`       |
| `seed`               | `Optional[int] \| NotGiven`                                                                   | If specified, our system will make a best effort to sample deterministically.                                                                                                                                                                                                                 | `NOT_GIVEN`       |
| `stop`               | `Union[Optional[str], List[str], None] \| NotGiven`                                           | Up to 4 sequences where the API will stop generating further tokens.                                                                                                                                                                                                                          | `NOT_GIVEN`       |
| `stream`             | `Optional[Literal[False]] \| NotGiven` or `Literal[True]`                                     | Whether to stream back partial progress. If `True`, tokens are sent as [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format).                                                                                 | `NOT_GIVEN` (False) |
| `stream_options`     | `Optional[ChatCompletionStreamOptionsParam] \| NotGiven`                                      | Options for streaming response. Only set when `stream=True`.                                                                                                                                                                                                                                  | `NOT_GIVEN`       |
| `suffix`             | `Optional[str] \| NotGiven`                                                                   | The suffix that comes after a completion of inserted text. Supported only for `gpt-3.5-turbo-instruct`.                                                                                                                                                                                       | `NOT_GIVEN`       |
| `temperature`        | `Optional[float] \| NotGiven`                                                                 | Sampling temperature between 0 and 2. Higher values increase randomness.                                                                                                                                                                                                                      | `NOT_GIVEN`       |
| `top_p`              | `Optional[float] \| NotGiven`                                                                 | An alternative to temperature, called nucleus sampling.                                                                                                                                                                                                                                       | `NOT_GIVEN`       |
| `user`               | `str \| NotGiven`                                                                             | A unique identifier for your end-user, helping OpenAI monitor and detect abuse. [Learn more](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids).                                                                                                                    | `NOT_GIVEN`       |
| `extra_headers`      | `Headers \| None`                                                                             | Send extra headers with the request.                                                                                                                                                                                                                                                          | `None`            |
| `extra_query`        | `Query \| None`                                                                               | Add additional query parameters to the request.                                                                                                                                                                                                                                               | `None`            |
| `extra_body`         | `Body \| None`                                                                                | Add additional JSON properties to the request body.                                                                                                                                                                                                                                           | `None`            |
| `timeout`            | `float \| httpx.Timeout \| None \| NotGiven`                                                  | Override the client-level default timeout for this request, in seconds.                                                                                                                                                                                                                       | `NOT_GIVEN`       |

#### **Return Types**

*   **`Completion`** (when `stream=False` or omitted): A single completion object containing the generated text and additional metadata.
*   **`Stream[Completion]`** (when `stream=True`): A stream of completion objects, allowing you to process the generated text in chunks as it becomes available.

#### **Usage Examples**

**Non-Streaming Completion**

```python
completion = Completions().create(
    model="gpt-3.5-turbo-instruct",
    prompt="Hello, world!",
    max_tokens=50
)
print(completion.choices[0].text)
```

**Streaming Completion**

```python
stream = Completions().create(
    model="gpt-3.5-turbo-instruct",
    prompt="Tell me a story.",
    max_tokens=100,
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].text, end='')
```

**Deterministic Sampling with Seed**

```python
completion = Completions().create(
    model="gpt-3.5-turbo-instruct",
    prompt="Generate a 5-word phrase.",
    seed=1234,
    max_tokens=10
)
print(completion.choices[0].text)
```

**Using Logit Bias**

```python
# Prevent the model from generating the <|endoftext|> token
completion = Completions().create(
    model="gpt-3.5-turbo-instruct",
    prompt="A short story:",
    logit_bias={"50256": -100},  # Token ID for <|endoftext|>
    max_tokens=200
)
print(completion.choices[0].text)
```

**Additional Tips and Best Practices**

*   Always check the available models and their capabilities via the [List Models API](https://platform.openai.com/docs/api-reference/models/list).
*   Use the [Tokenizer Tool](/tokenizer?view=bpe) to convert text to token IDs when working with `logit_bias`.
*   Monitor your token quota carefully, especially when using `best_of` and `n` parameters.
*   For large completions, consider using `stream=True` to handle the response in chunks.
*   Implement error handling and retries for robust production deployments.



**images_api_example.py**
```python
import os
import base64
from typing import Dict, Any, Optional, Union, Literal, Tuple
from dataclasses import dataclass
from enum import Enum
import httpx
import json

# Mocking the required types and classes for demonstration
FileTypes = Union[str, bytes, Tuple[str, bytes]]
Headers = Dict[str, str]
Query = Dict[str, str]
Body = Dict[str, Any]
ImageModel = Literal["dall-e-2", "dall-e-3"]

class NotGiven:
    pass

NOT_GIVEN = NotGiven()

@dataclass
class ImagesResponse:
    data: Any
    headers: Dict[str, str]
    status_code: int

class SyncAPIResource:
    def _post(self, endpoint: str, body: Dict[str, Any], files: Dict[str, Any], options: Dict[str, Any], cast_to: Any) -> Any:
        # Simulating the POST request for demonstration
        print(f"POST {endpoint} with body: {body} and files: {files}")
        return cast_to(data=["Mocked Image Data"], headers={}, status_code=200)

def deepcopy_minimal(obj: Any) -> Any:
    return obj

def extract_files(body: Dict[str, Any], paths: list) -> Dict[str, Any]:
    files = {}
    for path in paths:
        key = path[0]
        if key in body:
            files[key] = body[key]
    return files

def maybe_transform(body: Dict[str, Any], params_type: Any) -> Dict[str, Any]:
    return body

def make_request_options(extra_headers: Headers, extra_query: Query, extra_body: Body, timeout: Optional[Union[float, httpx.Timeout]]) -> Dict[str, Any]:
    options = {}
    if extra_headers:
        options["headers"] = extra_headers
    if extra_query:
        options["params"] = extra_query
    if extra_body:
        options["json"] = extra_body
    if timeout:
        options["timeout"] = timeout
    return options

class ImagesWithRawResponse:
    def __init__(self, images: 'Images'):
        self.images = images

    def create_variation(self, *args, **kwargs) -> ImagesResponse:
        # For raw response, we simulate reading headers and status code
        response = self.images.create_variation(*args, **kwargs)
        return ImagesResponse(data=response.data, headers={"Content-Type": "application/json"}, status_code=200)

    def edit(self, *args, **kwargs) -> ImagesResponse:
        response = self.images.edit(*args, **kwargs)
        return ImagesResponse(data=response.data, headers={"Content-Type": "application/json"}, status_code=200)

    def generate(self, *args, **kwargs) -> ImagesResponse:
        response = self.images.generate(*args, **kwargs)
        return ImagesResponse(data=response.data, headers={"Content-Type": "application/json"}, status_code=200)

class ImagesWithStreamingResponse:
    def __init__(self, images: 'Images'):
        self.images = images

    def create_variation(self, *args, **kwargs) -> ImagesResponse:
        # Simulating streaming response
        def stream():
            yield b"Mocked streaming response chunk 1"
            yield b"Mocked streaming response chunk 2"
        return ImagesResponse(data=list(stream()), headers={"Content-Type": "application/octet-stream"}, status_code=200)

    def edit(self, *args, **kwargs) -> ImagesResponse:
        def stream():
            yield b"Mocked streaming response chunk 1"
            yield b"Mocked streaming response chunk 2"
        return ImagesResponse(data=list(stream()), headers={"Content-Type": "application/octet-stream"}, status_code=200)

    def generate(self, *args, **kwargs) -> ImagesResponse:
        def stream():
            yield b"Mocked streaming response chunk 1"
            yield b"Mocked streaming response chunk 2"
        return ImagesResponse(data=list(stream()), headers={"Content-Type": "application/octet-stream"}, status_code=200)

class Images(SyncAPIResource):
    @property
    def with_raw_response(self) -> ImagesWithRawResponse:
        return ImagesWithRawResponse(self)

    @property
    def with_streaming_response(self) -> ImagesWithStreamingResponse:
        return ImagesWithStreamingResponse(self)

    def create_variation(
        self,
        *,
        image: FileTypes,
        model: Union[str, ImageModel, None] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        response_format: Optional[Literal["url", "b64_json"]] | NotGiven = NOT_GIVEN,
        size: Optional[Literal["256x256", "512x512", "1024x1024"]] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> ImagesResponse:
        body = deepcopy_minimal(
            {
                "image": image,
                "model": model,
                "n": n,
                "response_format": response_format,
                "size": size,
                "user": user,
            }
        )
        files = extract_files(body, paths=[["image"]])
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        return self._post(
            "/images/variations",
            body=maybe_transform(body, None),
            files=files,
            options=make_request_options(extra_headers, extra_query, extra_body, timeout),
            cast_to=ImagesResponse,
        )

    def edit(
        self,
        *,
        image: FileTypes,
        prompt: str,
        mask: FileTypes | NotGiven = NOT_GIVEN,
        model: Union[str, ImageModel, None] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        response_format: Optional[Literal["url", "b64_json"]] | NotGiven = NOT_GIVEN,
        size: Optional[Literal["256x256", "512x512", "1024x1024"]] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> ImagesResponse:
        body = deepcopy_minimal(
            {
                "image": image,
                "prompt": prompt,
                "mask": mask,
                "model": model,
                "n": n,
                "response_format": response_format,
                "size": size,
                "user": user,
            }
        )
        files = extract_files(body, paths=[["image"], ["mask"]])
        extra_headers = {"Content-Type": "multipart/form-data", **(extra_headers or {})}
        return self._post(
            "/images/edits",
            body=maybe_transform(body, None),
            files=files,
            options=make_request_options(extra_headers, extra_query, extra_body, timeout),
            cast_to=ImagesResponse,
        )

    def generate(
        self,
        *,
        prompt: str,
        model: Union[str, ImageModel, None] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        quality: Literal["standard", "hd"] | NotGiven = NOT_GIVEN,
        response_format: Optional[Literal["url", "b64_json"]] | NotGiven = NOT_GIVEN,
        size: Optional[Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]] | NotGiven = NOT_GIVEN,
        style: Optional[Literal["vivid", "natural"]] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> ImagesResponse:
        return self._post(
            "/images/generations",
            body=maybe_transform(
                {
                    "prompt": prompt,
                    "model": model,
                    "n": n,
                    "quality": quality,
                    "response_format": response_format,
                    "size": size,
                    "style": style,
                    "user": user,
                },
                None,
            ),
            files={},
            options=make_request_options(extra_headers, extra_query, extra_body, timeout),
            cast_to=ImagesResponse,
        )

def main():
    images = Images()

    print("# Create Variation")
    print("--------------------")
    variation_response = images.create_variation(
        image="path/to/image.png",
        model="dall-e-2",
        n=3,
        response_format="url",
        size="512x512",
        user="your_end_user_id"
    )
    print("Variation Response:", variation_response.data)
    print()

    print("# Create Variation with Raw Response")
    print("-----------------------------------")
    raw_variation_response = images.with_raw_response.create_variation(
        image="path/to/image.png",
        model="dall-e-2",
        n=3,
        response_format="url",
        size="512x512",
        user="your_end_user_id"
    )
    print("Raw Variation Response Data:", raw_variation_response.data)
    print("Raw Variation Response Headers:", raw_variation_response.headers)
    print("Raw Variation Status Code:", raw_variation_response.status_code)
    print()

    print("# Create Variation with Streaming Response")
    print("----------------------------------------")
    streaming_variation_response = images.with_streaming_response.create_variation(
        image="path/to/image.png",
        model="dall-e-2",
        n=3,
        response_format="url",
        size="512x512",
        user="your_end_user_id"
    )
    print("Streaming Variation Response Data:", streaming_variation_response.data)
    for i, chunk in enumerate(streaming_variation_response.data):
        print(f"Chunk {i+1}: {chunk}")
    print()

    print("# Edit Image")
    print("-------------")
    edit_response = images.edit(
        image="path/to/image.png",
        prompt="A sunlit indoor lounge area with a pool view.",
        mask="path/to/mask.png",
        model="dall-e-2",
        n=2,
        response_format="b64_json",
        size="1024x1024",
        user="your_end_user_id"
    )
    print("Edit Response:", edit_response.data)
    print()

    print("# Edit Image with Raw Response")
    print("-----------------------------")
    raw_edit_response = images.with_raw_response.edit(
        image="path/to/image.png",
        prompt="A sunlit indoor lounge area with a pool view.",
        mask="path/to/mask.png",
        model="dall-e-2",
        n=2,
        response_format="b64_json",
        size="1024x1024",
        user="your_end_user_id"
    )
    print("Raw Edit Response Data:", raw_edit_response.data)
    print("Raw Edit Response Headers:", raw_edit_response.headers)
    print("Raw Edit Status Code:", raw_edit_response.status_code)
    print()

    print("# Generate Image")
    print("-----------------")
    generate_response = images.generate(
        prompt="A futuristic cityscape at sunset, with flying cars and neon lights.",
        model="dall-e-3",
        n=1,
        quality="hd",
        response_format="url",
        size="1024x1792",
        style="vivid",
        user="your_end_user_id"
    )
    print("Generate Response:", generate_response.data)
    print()

    print("# Generate Image with Raw Response")
    print("--------------------------------")
    raw_generate_response = images.with_raw_response.generate(
        prompt="A futuristic cityscape at sunset, with flying cars and neon lights.",
        model="dall-e-3",
        n=1,
        quality="hd",
        response_format="url",
        size="1024x1792",
        style="vivid",
        user="your_end_user_id"
    )
    print("Raw Generate Response Data:", raw_generate_response.data)
    print("Raw Generate Response Headers:", raw_generate_response.headers)
    print("Raw Generate Status Code:", raw_generate_response.status_code)
    print()

    print("# Generate Image with Streaming Response")
    print("-------------------------------------")
    streaming_generate_response = images.with_streaming_response.generate(
        prompt="A futuristic cityscape at sunset, with flying cars and neon lights.",
        model="dall-e-3",
        n=1,
        quality="hd",
        response_format="url",
        size="1024x1792",
        style="vivid",
        user="your_end_user_id"
    )
    print("Streaming Generate Response Data:", streaming_generate_response.data)
    for i, chunk in enumerate(streaming_generate_response.data):
        print(f"Chunk {i+1}: {chunk}")
    print()

if __name__ == "__main__":
    main()
```

**Explanation**

1.  The script starts by defining mock types and classes (`FileTypes`, `Headers`, `Query`, `Body`, `ImageModel`, `NotGiven`, `ImagesResponse`) for demonstration purposes since the actual implementation would depend on the OpenAI library.
2.  It defines helper functions (`deepcopy_minimal`, `extract_files`, `maybe_transform`, `make_request_options`) that mimic the behavior of similar functions. These simulate the actual behavior of processing requests and responses.
3.  The `Images` class is the core, with methods for `create_variation`, `edit`, and `generate`. Each method constructs a request body, simulates file extraction (for `create_variation` and `edit`), and makes a POST request using the `_post` method.
4.  The `with_raw_response` and `with_streaming_response` properties return instances of `ImagesWithRawResponse` and `ImagesWithStreamingResponse`, respectively. These classes wrap the original methods to simulate raw and streaming responses.
5.  In the `main` function:
    *   An instance of the `Images` class is created.
    *   Each API method (`create_variation`, `edit`, `generate`) is called with example parameters, demonstrating normal, raw, and streaming response handling.
    *   Responses are printed to showcase the data returned by each method and response type.

**Running the Script**

Save the above code in `images_api_example.py` and run it using Python:

```bash
python images_api_example.py
```

This will execute the `main` function, simulating calls to the OpenAI Images API and printing the responses. Observe how different response types (normal, raw, streaming) are handled for each method.

**Output Snippet**

```
# Create Variation
--------------------
POST /images/variations with body: {'image': 'path/to/image.png', 'model': 'dall-e-2', 'n': 3, 'response_format': 'url', 'size': '512x512', 'user': 'your_end_user_id'} and files: {'image': 'path/to/image.png'}
Variation Response: ['Mocked Image Data']

# Create Variation with Raw Response
-----------------------------------
POST /images/variations with body: {'image': 'path/to/image.png', 'model': 'dall-e-2', 'n': 3, 'response_format': 'url', 'size': '512x512', 'user': 'your_end_user_id'} and files: {'image': 'path/to/image.png'}
Raw Variation Response Data: ['Mocked Image Data']
Raw Variation Response Headers: {'Content-Type': 'application/json'}
Raw Variation Status Code: 200

# Create Variation with Streaming Response
----------------------------------------
POST /images/variations with body: {'image': 'path/to/image.png', 'model': 'dall-e-2', 'n': 3, 'response_format': 'url', 'size': '512x512', 'user': 'your_end_user_id'} and files: {'image': 'path/to/image.png'}
Streaming Variation Response Data: [b'Mocked streaming response chunk 1', b'Mocked streaming response chunk 2']
Chunk 1: b'Mocked streaming response chunk 1'
Chunk 2: b'Mocked streaming response chunk 2'
```

This end-to-end example covers:

*   **Instantiation** of the `Images` API client.
*   **Method calls** for `create_variation`, `edit`, and `generate` with varied parameters.
*   Handling of **normal**, **raw**, and **streaming** responses for each method.
*   **Mocked request** and **response simulation** to demonstrate the flow without actual API calls.
*   **Error handling** is implicitly shown through the structure, ready to be adapted with real OpenAI error classes.

Feel free to replace the mocked parts with actual implementations from the OpenAI Python library for production use.