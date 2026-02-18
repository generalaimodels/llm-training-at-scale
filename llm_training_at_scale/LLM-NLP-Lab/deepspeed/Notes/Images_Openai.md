# `Images` Class API Documentation

## Class Definition

```python
class Images(SyncAPIResource)
```

A synchronous API resource for image generation, variation, and editing operations.

## Properties

### `with_raw_response`

```python
@cached_property
def with_raw_response(self) -> ImagesWithRawResponse
```

Access the raw HTTP response data (headers, status codes) instead of just parsed content.

**Returns:** `ImagesWithRawResponse` - A wrapper that returns raw response objects.

**Example:**
```python
response = client.images.with_raw_response.generate(prompt="A sunset over mountains")
print(f"Status code: {response.status_code}")
print(f"Headers: {response.headers}")
```

### `with_streaming_response`

```python
@cached_property
def with_streaming_response(self) -> ImagesWithStreamingResponse
```

Alternative to `with_raw_response` that doesn't eagerly read the response body.

**Returns:** `ImagesWithStreamingResponse` - A wrapper for streaming responses.

**Example:**
```python
with client.images.with_streaming_response.generate(prompt="A sunset") as response:
    for chunk in response.iter_content(chunk_size=4096):
        # Process streaming content
```

## Methods

### `create_variation`

```python
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
) -> ImagesResponse
```

Creates variations of a given image.

**Parameters:**
- `image`: `FileTypes` - The image to use as the basis for variation(s). Must be a valid PNG file, less than 4MB, and square.
- `model`: `Union[str, ImageModel, None]` - The model to use for image generation. Only `dall-e-2` is supported for variations.
- `n`: `Optional[int]` - Number of images to generate (1-10). For `dall-e-3`, only `n=1` is supported.
- `response_format`: `Optional[Literal["url", "b64_json"]]` - Format of the generated images. URLs are valid for 60 minutes.
- `size`: `Optional[Literal["256x256", "512x512", "1024x1024"]]` - Size of the generated images.
- `user`: `str` - Unique identifier for your end-user for monitoring and abuse detection.
- `extra_headers`, `extra_query`, `extra_body`, `timeout` - Additional request parameters.

**Returns:** `ImagesResponse` - Contains the generated image variations.

**Example:**
```python
from pathlib import Path

# Create a variation of an existing image
response = client.images.create_variation(
    image=Path("original_image.png"),
    n=3,
    size="1024x1024",
    response_format="url"
)

# Access the image URLs
for image in response.data:
    print(image.url)
```

### `edit`

```python
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
) -> ImagesResponse
```

Creates an edited or extended image given an original image and a text prompt.

**Parameters:**
- `image`: `FileTypes` - The image to edit. Must be a valid PNG file, less than 4MB, and square. If no mask is provided, image must have transparency to use as the mask.
- `prompt`: `str` - Text description of the desired image(s). Maximum 1000 characters.
- `mask`: `FileTypes` - Optional image whose transparent areas indicate where `image` should be edited. Must match dimensions of `image`.
- `model`: `Union[str, ImageModel, None]` - Model for image generation. Only `dall-e-2` is supported for edits.
- `n`: `Optional[int]` - Number of images to generate (1-10).
- `response_format`: `Optional[Literal["url", "b64_json"]]` - Format of the generated images. URLs are valid for 60 minutes.
- `size`: `Optional[Literal["256x256", "512x512", "1024x1024"]]` - Size of the generated images.
- `user`: `str` - Unique identifier for your end-user for monitoring and abuse detection.
- `extra_headers`, `extra_query`, `extra_body`, `timeout` - Additional request parameters.

**Returns:** `ImagesResponse` - Contains the edited images.

**Example:**
```python
from pathlib import Path

# Edit an image with a mask
response = client.images.edit(
    image=Path("original.png"),
    mask=Path("mask.png"),
    prompt="A cozy living room with a fireplace",
    n=1,
    size="1024x1024"
)

# Get the edited image URL
edited_image_url = response.data[0].url
```

### `generate`

```python
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
) -> ImagesResponse
```

Creates an image given a text prompt.

**Parameters:**
- `prompt`: `str` - Text description of the desired image(s). Maximum 1000 characters for `dall-e-2` or 4000 for `dall-e-3`.
- `model`: `Union[str, ImageModel, None]` - Model for image generation.
- `n`: `Optional[int]` - Number of images to generate (1-10). For `dall-e-3`, only `n=1` is supported.
- `quality`: `Literal["standard", "hd"]` - Image quality. `hd` creates finer details and greater consistency. Only for `dall-e-3`.
- `response_format`: `Optional[Literal["url", "b64_json"]]` - Format of the generated images. URLs are valid for 60 minutes.
- `size`: `Optional[Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]]` - Size of the generated images. Size options vary by model.
- `style`: `Optional[Literal["vivid", "natural"]]` - Style of generated images. Only for `dall-e-3`.
  - `vivid`: Hyper-real and dramatic images
  - `natural`: More natural, less hyper-real images
- `user`: `str` - Unique identifier for your end-user for monitoring and abuse detection.
- `extra_headers`, `extra_query`, `extra_body`, `timeout` - Additional request parameters.

**Returns:** `ImagesResponse` - Contains the generated images.

**Example:**
```python
# Generate an image with dall-e-3
response = client.images.generate(
    prompt="An astronaut riding a horse on Mars, detailed digital art",
    model="dall-e-3",
    quality="hd",
    size="1024x1024",
    style="vivid"
)

# Get the generated image URL
image_url = response.data[0].url
```

## Response Types

All image methods return an `ImagesResponse` object containing the generated images data.

```python
"""
OpenAI Images API - Comprehensive Example

This file demonstrates all the functionality of the OpenAI Images API using the Python SDK,
including image generation, editing, variation creation, and working with different response types.
"""

import os
import base64
import time
from pathlib import Path
from typing import Dict, List, Optional

import openai
import httpx
from PIL import Image
from io import BytesIO

# Initialize the OpenAI client
# Set OPENAI_API_KEY environment variable or pass directly
api_key = os.environ.get("OPENAI_API_KEY", "your_api_key_here")
client = openai.OpenAI(api_key=api_key)

# Utility functions for working with images
def save_image_from_url(url: str, filename: str) -> None:
    """Download and save an image from a URL."""
    response = httpx.get(url)
    response.raise_for_status()
    
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"Image saved to {filename}")

def save_image_from_b64(b64_json: str, filename: str) -> None:
    """Decode and save a base64 encoded image."""
    image_data = base64.b64decode(b64_json)
    
    with open(filename, "wb") as f:
        f.write(image_data)
    print(f"Image saved to {filename}")

def create_sample_transparent_image(filename: str, size: tuple = (1024, 1024)) -> None:
    """Create a sample PNG with transparency for testing."""
    # Create a transparent image with a simple shape
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    
    # Draw a simple circle in the middle
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    center = (size[0] // 2, size[1] // 2)
    radius = min(size) // 3
    draw.ellipse(
        (center[0] - radius, center[1] - radius, 
         center[0] + radius, center[1] + radius),
        fill=(255, 0, 0, 128)
    )
    
    img.save(filename, 'PNG')
    print(f"Created sample image: {filename}")

def create_sample_mask(filename: str, size: tuple = (1024, 1024)) -> None:
    """Create a sample mask image for testing edits."""
    # Create a mask with transparent areas where edits should be applied
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    
    # Make a rectangular transparent area
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    width, height = size
    draw.rectangle(
        (width // 4, height // 4, 3 * width // 4, 3 * height // 4),
        fill=(0, 0, 0, 0)
    )
    # Fill the rest with black
    draw.rectangle(
        (0, 0, width, height),
        fill=(0, 0, 0, 255),
        outline=None,
        width=0
    )
    draw.rectangle(
        (width // 4, height // 4, 3 * width // 4, 3 * height // 4),
        fill=(0, 0, 0, 0),
        outline=None,
        width=0
    )
    
    img.save(filename, 'PNG')
    print(f"Created sample mask: {filename}")


# Example 1: Generate Images using different parameters
def example_image_generation():
    print("\n=== Image Generation Examples ===")
    
    # Basic image generation with default parameters
    basic_response = client.images.generate(
        prompt="A serene lake surrounded by mountains at sunset",
    )
    save_image_from_url(basic_response.data[0].url, "basic_generation.png")
    
    # Advanced generation with DALL-E 3
    dalle3_response = client.images.generate(
        prompt="A futuristic cityscape with flying vehicles and vertical gardens on skyscrapers",
        model="dall-e-3",
        quality="hd",
        size="1024x1024",
        style="vivid",
        response_format="url",
        user="user123"  # For tracking/monitoring purposes
    )
    save_image_from_url(dalle3_response.data[0].url, "dalle3_generation.png")
    
    # Generate multiple images with DALL-E 2
    multi_response = client.images.generate(
        prompt="A cartoon robot playing with a cat",
        model="dall-e-2",
        n=3,  # Generate 3 images
        size="512x512",
        response_format="b64_json"  # Get base64 encoded images
    )
    
    # Save multiple generated images
    for i, image_data in enumerate(multi_response.data):
        save_image_from_b64(image_data.b64_json, f"multi_generation_{i}.png")
    
    # Generate with different aspect ratios (DALL-E 3 only)
    wide_response = client.images.generate(
        prompt="A panoramic view of a mountain range",
        model="dall-e-3",
        size="1792x1024",  # Wide format
    )
    save_image_from_url(wide_response.data[0].url, "wide_generation.png")
    
    tall_response = client.images.generate(
        prompt="A tall redwood tree in a forest",
        model="dall-e-3",
        size="1024x1792",  # Tall format
    )
    save_image_from_url(tall_response.data[0].url, "tall_generation.png")


# Example 2: Edit Images
def example_image_editing():
    print("\n=== Image Editing Examples ===")
    
    # Create sample test images
    base_image_path = "sample_base.png"
    mask_image_path = "sample_mask.png"
    create_sample_transparent_image(base_image_path)
    create_sample_mask(mask_image_path)
    
    # Edit image with a mask
    edit_with_mask_response = client.images.edit(
        image=Path(base_image_path),
        mask=Path(mask_image_path),
        prompt="A beautiful garden with flowers and a small pond",
        size="1024x1024",
        n=1,
        model="dall-e-2"  # Only dall-e-2 supports edits
    )
    save_image_from_url(edit_with_mask_response.data[0].url, "edit_with_mask.png")
    
    # Edit using transparency in original image (no explicit mask)
    edit_transparent_response = client.images.edit(
        image=Path(base_image_path),
        prompt="A mystical glowing orb floating in darkness",
        response_format="b64_json"
    )
    save_image_from_b64(edit_transparent_response.data[0].b64_json, "edit_transparent.png")
    
    # Edit with multiple outputs
    edit_multi_response = client.images.edit(
        image=Path(base_image_path),
        prompt="A cosmic nebula with stars",
        n=2,
        size="512x512"
    )
    
    for i, image_data in enumerate(edit_multi_response.data):
        save_image_from_url(image_data.url, f"edit_multi_{i}.png")


# Example 3: Create Image Variations
def example_image_variations():
    print("\n=== Image Variation Examples ===")
    
    # Create a sample image if not already created
    source_image_path = "sample_base.png"
    if not os.path.exists(source_image_path):
        create_sample_transparent_image(source_image_path)
    
    # Basic variation
    basic_variation_response = client.images.create_variation(
        image=Path(source_image_path),
        n=1,
        size="1024x1024"
    )
    save_image_from_url(basic_variation_response.data[0].url, "basic_variation.png")
    
    # Multiple variations with specific model
    multi_variation_response = client.images.create_variation(
        image=Path(source_image_path),
        model="dall-e-2",  # Only dall-e-2 supports variations
        n=4,
        size="512x512",
        response_format="b64_json"
    )
    
    for i, image_data in enumerate(multi_variation_response.data):
        save_image_from_b64(image_data.b64_json, f"variation_{i}.png")
    
    # Variation with user identifier
    user_variation_response = client.images.create_variation(
        image=Path(source_image_path),
        user="variation_test_user",
        size="256x256"
    )
    save_image_from_url(user_variation_response.data[0].url, "user_variation.png")


# Example 4: Working with Raw Response
def example_raw_response():
    print("\n=== Raw Response Examples ===")
    
    # Get raw HTTP response with headers and status code
    raw_response = client.images.with_raw_response.generate(
        prompt="A watercolor painting of a coastal village",
        size="512x512"
    )
    
    # Access response metadata
    print(f"Status code: {raw_response.status_code}")
    print(f"Content type: {raw_response.headers.get('content-type')}")
    print(f"Request ID: {raw_response.headers.get('x-request-id')}")
    
    # Access the parsed response data
    result = raw_response.parsed
    save_image_from_url(result.data[0].url, "raw_response_image.png")
    
    # Raw response with variations
    raw_variation_response = client.images.with_raw_response.create_variation(
        image=Path("sample_base.png"),
        n=1
    )
    
    print(f"Variation response status: {raw_variation_response.status_code}")
    variation_result = raw_variation_response.parsed
    save_image_from_url(variation_result.data[0].url, "raw_variation_image.png")


# Example 5: Working with Streaming Response
def example_streaming_response():
    print("\n=== Streaming Response Examples ===")
    
    # Use streaming response for handling large responses efficiently
    with client.images.with_streaming_response.generate(
        prompt="An oil painting of a sailing ship in a storm",
    ) as response:
        # Access headers before processing the body
        print(f"Response headers received: {response.headers.get('content-type')}")
        
        # Get the parsed response
        result = response.parsed
        save_image_from_url(result.data[0].url, "streaming_response_image.png")
    
    # Streaming response with edit
    with client.images.with_streaming_response.edit(
        image=Path("sample_base.png"),
        prompt="Transform into a crystal ball showing the future"
    ) as edit_response:
        edit_result = edit_response.parsed
        save_image_from_url(edit_result.data[0].url, "streaming_edit_image.png")


# Example 6: Error Handling
def example_error_handling():
    print("\n=== Error Handling Examples ===")
    
    # Example 1: Empty prompt
    try:
        client.images.generate(prompt="")
    except Exception as e:
        print(f"Empty prompt error: {e}")
    
    # Example 2: Invalid model parameter
    try:
        client.images.generate(
            prompt="A beautiful sunset",
            model="nonexistent-model"
        )
    except Exception as e:
        print(f"Invalid model error: {e}")
    
    # Example 3: Using n>1 with DALL-E 3
    try:
        client.images.generate(
            prompt="A mountain landscape",
            model="dall-e-3",
            n=2  # DALL-E 3 only supports n=1
        )
    except Exception as e:
        print(f"Invalid n value for DALL-E 3 error: {e}")
    
    # Example 4: Invalid size for model
    try:
        client.images.generate(
            prompt="A cityscape",
            model="dall-e-3",
            size="256x256"  # Invalid size for DALL-E 3
        )
    except Exception as e:
        print(f"Invalid size error: {e}")
    
    # Example 5: Timeout handling
    try:
        client.images.with_raw_response.generate(
            prompt="A complex scene with many details",
            timeout=0.001  # Unreasonably short timeout
        )
    except Exception as e:
        print(f"Timeout error: {e}")


# Example 7: Complete Workflow
def example_complete_workflow():
    print("\n=== Complete Workflow Example ===")
    
    # Step 1: Generate initial image
    print("Step 1: Generating initial image...")
    initial_response = client.images.generate(
        prompt="A simple house on a hill with a tree",
        model="dall-e-2",
        size="1024x1024",
        response_format="b64_json"
    )
    
    # Save the initial image
    initial_image_path = "workflow_initial.png"
    save_image_from_b64(initial_response.data[0].b64_json, initial_image_path)
    
    # Convert to RGBA for editing
    with Image.open(initial_image_path) as img:
        rgba_img = img.convert("RGBA")
        # Create transparent area in center
        width, height = rgba_img.size
        for x in range(width//3, 2*width//3):
            for y in range(height//3, 2*height//3):
                rgba_img.putpixel((x, y), (0, 0, 0, 0))
        rgba_img.save("workflow_transparent.png")
    
    # Step 2: Edit the image
    print("Step 2: Editing the image...")
    edit_response = client.images.edit(
        image=Path("workflow_transparent.png"),
        prompt="Add a beautiful garden with flowers and a fountain",
        size="1024x1024"
    )
    save_image_from_url(edit_response.data[0].url, "workflow_edited.png")
    
    # Step 3: Create variations
    print("Step 3: Creating variations...")
    variation_response = client.images.create_variation(
        image=Path("workflow_edited.png"),
        n=3,
        size="512x512"
    )
    
    for i, image_data in enumerate(variation_response.data):
        save_image_from_url(image_data.url, f"workflow_variation_{i}.png")
    
    print("Complete workflow executed successfully!")


# Run all examples
if __name__ == "__main__":
    print("=== OpenAI Images API Examples ===")
    
    # Create directories for output if they don't exist
    os.makedirs("output", exist_ok=True)
    os.chdir("output")
    
    try:
        # Run examples
        example_image_generation()
        example_image_editing()
        example_image_variations()
        example_raw_response()
        example_streaming_response()
        example_error_handling()
        example_complete_workflow()
        
        print("\nAll examples completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nExamples finished. Generated images are in the 'output' directory.")
```