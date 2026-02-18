**Images API Documentation**
=====================================

**Overview**
------------

The `Images` API is a part of the OpenAI API, allowing developers to generate, edit, and create variations of images using AI models. This documentation provides a detailed explanation of the `Images` class, its methods, and properties.

**Class Definition**
--------------------

```python
class Images(SyncAPIResource):
    ...
```

**Properties**
--------------

### `with_raw_response`

*   **Type:** `ImagesWithRawResponse`
*   **Description:** This property can be used as a prefix for any HTTP method call to return the raw response object instead of the parsed content.
*   **Usage:**

```python
images = Images()
raw_response = images.with_raw_response.create_variation(image="path/to/image.png")
print(raw_response.headers)
print(raw_response.status_code)
```

*   **Reference:** [Accessing raw response data (e.g. headers)](https://www.github.com/openai/openai-python#accessing-raw-response-data-eg-headers)

### `with_streaming_response`

*   **Type:** `ImagesWithStreamingResponse`
*   **Description:** An alternative to `.with_raw_response` that doesn't eagerly read the response body.
*   **Usage:**

```python
images = Images()
streaming_response = images.with_streaming_response.create_variation(image="path/to/image.png")
for chunk in streaming_response.iter_bytes():
    print(chunk)
```

*   **Reference:** [with_streaming_response](https://www.github.com/openai/openai-python#with_streaming_response)

**Methods**
------------

### **`create_variation`** 

**Creates a variation of a given image.**

*   **Endpoint:** `POST /images/variations`
*   **Return Type:** `ImagesResponse`
*   **Parameters:**

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `image` | `FileTypes` | **Yes** | The image to use as the basis for the variation(s). Must be a valid PNG file, less than 4MB, and square. |
| `model` | `Union[str, ImageModel, None]` | No | The model to use for image generation. Only `dall-e-2` is supported at this time. |
| `n` | `Optional[int]` | No | The number of images to generate. Must be between 1 and 10. For `dall-e-3`, only `n=1` is supported. |
| `response_format` | `Optional[Literal["url", "b64_json"]]` | No | The format in which the generated images are returned. Must be one of `url` or `b64_json`. URLs are only valid for 60 minutes after the image has been generated. |
| `size` | `Optional[Literal["256x256", "512x512", "1024x1024"]]` | No | The size of the generated images. Must be one of `256x256`, `512x512`, or `1024x1024`. |
| `user` | `str` | No | A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids). |
| `extra_headers` | `Headers` | No | Send extra headers |
| `extra_query` | `Query` | No | Add additional query parameters to the request |
| `extra_body` | `Body` | No | Add additional JSON properties to the request |
| `timeout` | `float` or `httpx.Timeout` | No | Override the client-level default timeout for this request, in seconds |

*   **Example Usage:**

```python
images = Images()
variation_response = images.create_variation(
    image="path/to/image.png",
    model="dall-e-2",
    n=3,
    response_format="url",
    size="512x512",
    user="your_end_user_id"
)
print(variation_response.data)  # List of image URLs or base64 encoded JSON
```

### **`edit`** 

**Creates an edited or extended image given an original image and a prompt.**

*   **Endpoint:** `POST /images/edits`
*   **Return Type:** `ImagesResponse`
*   **Parameters:**

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `image` | `FileTypes` | **Yes** | The image to edit. Must be a valid PNG file, less than 4MB, and square. If mask is not provided, image must have transparency, which will be used as the mask. |
| `prompt` | `str` | **Yes** | A text description of the desired image(s). The maximum length is 1000 characters. |
| `mask` | `FileTypes` | No | An additional image whose fully transparent areas (e.g. where alpha is zero) indicate where `image` should be edited. Must be a valid PNG file, less than 4MB, and have the same dimensions as `image`. |
| `model` | `Union[str, ImageModel, None]` | No | The model to use for image generation. Only `dall-e-2` is supported at this time. |
| `n` | `Optional[int]` | No | The number of images to generate. Must be between 1 and 10. |
| `response_format` | `Optional[Literal["url", "b64_json"]]` | No | The format in which the generated images are returned. Must be one of `url` or `b64_json`. URLs are only valid for 60 minutes after the image has been generated. |
| `size` | `Optional[Literal["256x256", "512x512", "1024x1024"]]` | No | The size of the generated images. Must be one of `256x256`, `512x512`, or `1024x1024`. |
| `user` | `str` | No | A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids). |
| `extra_headers` | `Headers` | No | Send extra headers |
| `extra_query` | `Query` | No | Add additional query parameters to the request |
| `extra_body` | `Body` | No | Add additional JSON properties to the request |
| `timeout` | `float` or `httpx.Timeout` | No | Override the client-level default timeout for this request, in seconds |

*   **Example Usage:**

```python
images = Images()
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
print(edit_response.data)  # List of base64 encoded image JSON
```

### **`generate`** 

**Creates an image given a prompt.**

*   **Endpoint:** `POST /images/generations`
*   **Return Type:** `ImagesResponse`
*   **Parameters:**

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `prompt` | `str` | **Yes** | A text description of the desired image(s). The maximum length is 1000 characters for `dall-e-2` and 4000 characters for `dall-e-3`. |
| `model` | `Union[str, ImageModel, None]` | No | The model to use for image generation. |
| `n` | `Optional[int]` | No | The number of images to generate. Must be between 1 and 10. For `dall-e-3`, only `n=1` is supported. |
| `quality` | `Literal["standard", "hd"]` | No | The quality of the image that will be generated. `hd` creates images with finer details and greater consistency across the image. This param is only supported for `dall-e-3`. |
| `response_format` | `Optional[Literal["url", "b64_json"]]` | No | The format in which the generated images are returned. Must be one of `url` or `b64_json`. URLs are only valid for 60 minutes after the image has been generated. |
| `size` | `Optional[Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]]` | No | The size of the generated images. Must be one of `256x256`, `512x512`, or `1024x1024` for `dall-e-2`. Must be one of `1024x1024`, `1792x1024`, or `1024x1792` for `dall-e-3` models. |
| `style` | `Optional[Literal["vivid", "natural"]]` | No | The style of the generated images. Must be one of `vivid` or `natural`. Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images. This param is only supported for `dall-e-3`. |
| `user` | `str` | No | A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids). |
| `extra_headers` | `Headers` | No | Send extra headers |
| `extra_query` | `Query` | No | Add additional query parameters to the request |
| `extra_body` | `Body` | No | Add additional JSON properties to the request |
| `timeout` | `float` or `httpx.Timeout` | No | Override the client-level default timeout for this request, in seconds |

*   **Example Usage:**

```python
images = Images()
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
print(generate_response.data)  # List of image URLs
```

**Error Handling**
------------------

All methods may raise the following exceptions:

*   `openai.APIError`: Generic API error.
*   `openai.APIConnectionError`: Network-related error.
*   `openai.RateLimitError`: Too many requests error.
*   `openai.AuthenticationError`: Authentication failed error.

**Type Definitions**
--------------------

*   `FileTypes`: Union of `str` (file path), `bytes`, or `Tuple[str, bytes]` (filename, file content).
*   `ImageModel`: Literal["dall-e-2", "dall-e-3"].
*   `Headers`: `Dict[str, str]`.
*   `Query`: `Dict[str, str]`.
*   `Body`: `Dict[str, Any]`.
*   `ImagesResponse`: A custom response object containing the generated image data.

By following the documentation and examples above, developers should be able to seamlessly integrate the `Images` API into their applications, leveraging the power of AI-generated images for various use cases. 

**HTTP Status Codes**
----------------------

The following HTTP status codes are returned by the API:

| Status Code | Description |
|-------------|-------------|
| 200         | OK - Request successful. |
| 400         | Bad Request - Invalid request parameters. |
| 401         | Unauthorized - Authentication failed. |
| 429         | Too Many Requests - Rate limit exceeded. |
| 500         | Internal Server Error - Unexpected server error. |

For more details about error handling and status codes, refer to the [OpenAI API Documentation](https://platform.openai.com/docs/api-reference).