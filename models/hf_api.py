"""
HuggingFace Inference API client for Donut OCR.
Use this instead of local model loading for lightweight cloud deployment.
"""
import base64
import io
import json
import requests
from PIL import Image

HF_API_URL = "https://api-inference.huggingface.co/models"


def run_donut_hf_api(image, model_id="naver-clova-ix/donut-base-finetuned-cord-v2", api_token=None):
    """
    Run Donut OCR via HuggingFace Inference API.

    Args:
        image: PIL Image or path to image
        model_id: HuggingFace model ID (e.g., "your-username/your-model")
        api_token: HuggingFace API token (get from https://huggingface.co/settings/tokens)

    Returns:
        dict with "ocr_engine", "raw_sequence", "parsed"
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise ValueError("image must be PIL Image or file path")

    # Convert to bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # Call HF Inference API
    headers = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    api_endpoint = f"{HF_API_URL}/{model_id}"

    try:
        response = requests.post(
            api_endpoint,
            headers=headers,
            data=img_bytes,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        # HF API returns a list, get first element
        if isinstance(result, list):
            result = result[0]

        # Extract generated text
        generated_text = result.get("generated_text", "")

        return {
            "ocr_engine": f"HF API ({model_id})",
            "raw_sequence": generated_text,
            "parsed": None  # You'd need to parse the sequence same as local Donut
        }
    except requests.exceptions.HTTPError as e:
        if response.status_code == 503:
            return {"error": "Model is loading on HF. Wait a minute and retry."}
        elif response.status_code == 401:
            return {"error": "Invalid HF API token. Get one at huggingface.co/settings/tokens"}
        else:
            return {"error": f"HF API error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}
