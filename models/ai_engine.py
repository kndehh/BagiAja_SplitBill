import cv2
import numpy as np
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
import re
import pandas as pd
import json

# Try importing pytesseract (optional dependency)
try:
    import pytesseract
    import shutil
    if not shutil.which("tesseract"):
        import os
        default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_path):
            pytesseract.pytesseract.tesseract_cmd = default_path
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# ===============================
# MODEL LOADING
# ===============================
def load_donut_model(model_path="./donut-mega-finetuned-final-v6", hf_fallback="Kndeh/Finetuned_Donut_V6"):
    """Load Donut model from local path; fall back to HuggingFace Hub if not found."""
    import os
    # Determine which path to use
    print(f"[DEBUG] model_path: {model_path}, exists: {os.path.isdir(model_path) if model_path else False}")
    print(f"[DEBUG] hf_fallback: {hf_fallback}")
    if model_path and os.path.isdir(model_path):
        load_path = model_path
        print(f"[DEBUG] Using local path: {load_path}")
    elif hf_fallback:
        load_path = hf_fallback
        print(f"[DEBUG] Using HF Hub fallback: {load_path}")
    else:
        print("[DEBUG] No valid path found")
        return None, None, "cpu", False
    try:
        print(f"[DEBUG] Loading from: {load_path}")
        processor = DonutProcessor.from_pretrained(load_path)
        model = VisionEncoderDecoderModel.from_pretrained(load_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"[DEBUG] Successfully loaded on {device}")
        return processor, model, device, True
    except Exception as e:
        print(f"[DEBUG] Failed to load model from {load_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, "cpu", False

# ===============================
# PERSPECTIVE CORRECTION
# ===============================
def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def auto_crop_bright_region(img_cv, pad=20):
    """Fallback crop: finds the largest bright (paper-like) region via Otsu."""
    try:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        h, w = img_cv.shape[:2]
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        # Reject if region is too small or covers the whole frame
        if area < 0.05 * h * w or area > 0.95 * h * w:
            return None
        x, y, cw, ch = cv2.boundingRect(cnt)
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(w, x + cw + pad); y1 = min(h, y + ch + pad)
        return img_cv[y0:y1, x0:x1]
    except Exception:
        return None

def perspective_correction(img_cv):
    try:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours[:5]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                rect = order_points(pts)
                wA = np.linalg.norm(rect[2] - rect[3])
                wB = np.linalg.norm(rect[1] - rect[0])
                maxW = int(max(wA, wB))
                hA = np.linalg.norm(rect[1] - rect[2])
                hB = np.linalg.norm(rect[0] - rect[3])
                maxH = int(max(hA, hB))
                dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(rect, dst)
                return cv2.warpPerspective(img_cv, M, (maxW, maxH))
        return None
    except:
        return None

# ===============================
# IMAGE PREPROCESSING — Parameterized + Advanced
# ===============================
def preprocess_receipt(img_array, blur_type="Gaussian", blur_kernel=5, blur_sigma=0,
                       thresh_block=11, thresh_c=2, enable_bilateral=False,
                       enable_denoise=False, enable_morph=False, enable_sharpen=False,
                       enable_clahe=False, enable_perspective=False):
    if img_array is None:
        return [], None

    steps = []
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    steps.append((img_array, "1. Original"))

    if enable_perspective:
        corrected = perspective_correction(img_cv)
        if corrected is not None:
            img_cv = corrected
            steps.append((cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), "Perspective Corrected"))

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    steps.append((cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), "2. Grayscale"))

    if enable_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        steps.append((cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), "CLAHE"))

    if enable_denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        steps.append((cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), "Denoised"))

    if enable_bilateral:
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        steps.append((cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), "Bilateral Filter"))

    k = max(1, int(blur_kernel))
    if k % 2 == 0: k += 1
    if blur_type == "Median":
        blurred = cv2.medianBlur(gray, k)
    elif blur_type == "Box":
        blurred = cv2.blur(gray, (k, k))
    else:
        blurred = cv2.GaussianBlur(gray, (k, k), blur_sigma)
    steps.append((cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB), f"Blur ({blur_type} k={k})"))

    if enable_sharpen:
        kernel_s = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        blurred = cv2.filter2D(blurred, -1, kernel_s)
        steps.append((cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB), "Sharpened"))

    block = max(3, int(thresh_block))
    if block % 2 == 0: block += 1
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block, int(thresh_c))
    steps.append((cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB), f"Threshold (b={block}, C={int(thresh_c)})"))

    if enable_morph:
        km = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, km)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, km)
        steps.append((cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB), "Morphological Clean"))

    return steps, thresh

# ===============================
# PRICE UTILITIES
# ===============================
def clean_price(val):
    """Parse price string to float. Handles Indonesian format where dot = thousands separator."""
    if not val: return 0.0
    if isinstance(val, (int, float)): return float(val)
    val_str = str(val).strip()
    # Remove currency symbols and whitespace
    val_str = re.sub(r'[$€£¥₹Rp\s]', '', val_str)
    # Remove non-digit prefix
    val_str = re.sub(r'^[^\d]+', '', val_str)

    # Handle both dot and comma present (e.g., "1.234,56" or "1,234.56")
    if '.' in val_str and ',' in val_str:
        if val_str.rfind(',') > val_str.rfind('.'):
            # European/Indonesian: 1.234,56 -> 1234.56
            val_str = val_str.replace('.', '').replace(',', '.')
        else:
            # US format: 1,234.56 -> 1234.56
            val_str = val_str.replace(',', '')
    elif ',' in val_str:
        parts = val_str.split(',')
        if len(parts) == 2 and len(parts[1]) <= 2:
            # Comma is decimal separator: "25,00" -> 25.00
            val_str = val_str.replace(',', '.')
        else:
            # Comma is thousands separator: "25,000" -> 25000
            val_str = val_str.replace(',', '')
    elif '.' in val_str:
        parts = val_str.split('.')
        if len(parts) == 2 and len(parts[1]) <= 2:
            # Dot is decimal separator: "25.50" -> 25.50 (keep as is)
            pass
        else:
            # Dot is thousands separator: "25.000" -> 25000
            val_str = val_str.replace('.', '')

    match = re.search(r'[\d]+\.?[\d]*', val_str)
    if match:
        try: return float(match.group())
        except: return 0.0
    return 0.0

def is_valid_price_string(val):
    if val is None: return False
    val_str = str(val).strip()
    if not re.search(r'\d', val_str): return False
    digits = len(re.findall(r'\d', val_str))
    total_alnum = len(re.findall(r'[a-zA-Z\d]', val_str))
    if total_alnum > 0 and digits / total_alnum < 0.3: return False
    return True

# ===============================
# DONUT — SMART ITEM FILTER
# ===============================
def is_valid_item(nm, price_str, cnt_str=None):
    if not nm: return False
    nm_str = str(nm).strip()
    if len(nm_str) < 1: return False
    if not is_valid_price_string(price_str): return False
    price = clean_price(price_str)
    if price <= 0: return False
    skip_patterns = [
        r'(?i)^invoice', r'(?i)^date\s*(of|:)', r'(?i)^seller\s*:?', r'(?i)^client\s*:?',
        r'(?i)^buyer\s*:?', r'(?i)^customer\s*:?', r'(?i)^tax\s*id', r'(?i)^iban\s*:?',
        r'(?i)^dpo\s', r'(?i)^items?\s*$', r'(?i)^total\s*$', r'(?i)^sub\s*total',
        r'(?i)^summary', r'(?i)^vat\s', r'(?i)^no\.\s*$', r'(?i)^description\s*$',
        r'(?i)^worth\s*$', r'(?i)^ibay\s*:?', r'(?i)^qty\s*$', r'(?i)^quantity\s*$',
        r'(?i)^unit\s*price', r'(?i)^net\s*(worth|price)', r'(?i)^gross\s*(worth|price)',
        r'(?i)^amount\s*$', r'(?i)^payment', r'(?i)^change\s*$', r'(?i)^cash\s*$',
        r'(?i)^credit\s*card', r'(?i)^thank\s*you', r'(?i)^receipt',
        r'(?i)^bill\s*(no|number)', r'(?i)^order\s*(no|number|id)',
        r'(?i)^table\s*(no|number)', r'(?i)^server\s*:?', r'(?i)^cashier\s*:?',
    ]
    for p in skip_patterns:
        if re.search(p, nm_str): return False
    price_raw = str(price_str).strip()
    if len(re.findall(r'[a-zA-Z]', price_raw)) > 3: return False
    return True

# ===============================
# SMART FIELD MAPPING FIX
# ===============================
def smart_fix_summary(items_sum, subtotal, tax, service, discount, total):
    """Fix commonly misassigned CORD sub_total/total fields using heuristics."""
    vals = {"subtotal": subtotal, "tax": tax, "service": service, "total": total}
    non_zero = {k: v for k, v in vals.items() if v > 0}
    if not non_zero:
        return subtotal, tax, service, discount, total

    # Rule 1: largest value is most likely the total
    sorted_v = sorted(non_zero.items(), key=lambda x: x[1], reverse=True)
    top_key, top_val = sorted_v[0]
    if top_key != "total" and top_val > total:
        old_total = total
        total = top_val
        if top_key == "service": service = old_total
        elif top_key == "tax": tax = old_total
        elif top_key == "subtotal": subtotal = old_total

    # Rule 2: service should be small; if > items_sum it's wrong
    if service > 0 and items_sum > 0 and service > items_sum * 0.5:
        if abs(service - total) < 2:
            service = 0.0
        elif service > total:
            service, total = 0.0, service

    # Rule 3: if total == tax and there's a bigger value elsewhere, fix it
    if total > 0 and total == tax:
        candidates = [v for k, v in non_zero.items() if k not in ("tax", "total") and v > total]
        if candidates:
            total = max(candidates)
            service = 0.0

    # Rule 4: derive subtotal from items if missing
    if subtotal == 0 and items_sum > 0:
        subtotal = items_sum

    return subtotal, tax, service, discount, total

# ===============================
# DONUT PARSING
# ===============================
def parse_cord_to_schema(cord_json):
    items = []
    menu = cord_json.get("menu", [])
    if isinstance(menu, dict): menu = [menu]

    per_item_discount = 0.0
    for item in menu:
        entries = item if isinstance(item, list) else [item]
        for entry in entries:
            nm = entry.get("nm", "Unknown")
            price_raw = entry.get("price", "0")
            cnt_raw = entry.get("cnt", "1")
            if not is_valid_item(nm, price_raw, cnt_raw): continue
            price = clean_price(price_raw)
            cnt = clean_price(cnt_raw)
            if cnt == 0: cnt = 1
            items.append({"item_name": str(nm), "item_quantity": cnt, "item_price": price})
            per_item_discount += abs(clean_price(entry.get("discountprice", "0")))

    sub_total_node = cord_json.get("sub_total", {})
    if isinstance(sub_total_node, list):
        sub_total_node = sub_total_node[0] if sub_total_node else {}
    subtotal = clean_price(sub_total_node.get("subtotal_price", "0"))
    tax_amount = clean_price(sub_total_node.get("tax_price", "0"))
    service_charge = clean_price(sub_total_node.get("service_price", "0"))
    overall_discount = clean_price(sub_total_node.get("discount_price", "0"))
    discount_val = overall_discount if overall_discount > 0 else per_item_discount

    total_node = cord_json.get("total", {})
    if isinstance(total_node, list):
        total_node = total_node[0] if total_node else {}
    total_amount = clean_price(total_node.get("total_price", "0"))

    items_sum = sum(i["item_price"] for i in items)
    subtotal, tax_amount, service_charge, discount_val, total_amount = smart_fix_summary(
        items_sum, subtotal, tax_amount, service_charge, discount_val, total_amount
    )

    return {
        "items": items, "subtotal": subtotal, "tax_amount": tax_amount,
        "service_charge": service_charge,
        "discount_details": {"type": "fixed" if discount_val > 0 else "none", "value": discount_val},
        "total_amount": total_amount
    }

# ===============================
# DONUT OCR
# ===============================
def run_donut_ocr(img_array, processor, model, device, model_loaded,
                  blur_type="Gaussian", blur_kernel=5, blur_sigma=0,
                  thresh_block=11, thresh_c=2, use_preprocessed=False,
                  **filter_flags):
    gallery, thresh = preprocess_receipt(img_array, blur_type, blur_kernel, blur_sigma,
                                          thresh_block, thresh_c, **filter_flags)
    if not model_loaded:
        return gallery, {"error": "Donut model not loaded"}, None

    if use_preprocessed and thresh is not None:
        # Feed the thresholded (binary) preprocessing output as RGB
        rgb_for_donut = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB) if len(thresh.shape) == 2 else thresh
        print(f"[Donut] Using preprocessed image, shape={rgb_for_donut.shape}")
    else:
        # Auto-crop to receipt region (handles photos with lots of background)
        rgb_for_donut = img_array
        try:
            bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            cropped = perspective_correction(bgr)
            if cropped is None:
                cropped = auto_crop_bright_region(bgr)
            if cropped is not None:
                rgb_for_donut = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                print(f"[Donut] Cropped {bgr.shape[:2]} -> {cropped.shape[:2]}")
        except Exception as e:
            print(f"[Donut] crop failed: {e}")

    pil_img = Image.fromarray(rgb_for_donut).convert("RGB")
    # Let the processor handle resizing to the model's expected input size
    pixel_values = processor(pil_img, return_tensors="pt").pixel_values.to(device)
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(
        pixel_values, decoder_input_ids=decoder_input_ids,
        max_new_tokens=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True, bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    raw_seq = sequence
    sequence_stripped = re.sub(r"<.*?>", "", sequence, count=1).strip()
    raw_json = processor.token2json(sequence_stripped)
    # Fine-tuned model may output raw JSON instead of XML tokens — fall back if token2json returns empty
    if not raw_json or raw_json == {}:
        try:
            candidate = re.sub(r"^<[^>]+>", "", sequence.strip()).rstrip("</s>").strip()
            raw_json = json.loads(candidate)
            print("[Donut] Parsed output as raw JSON (fine-tuned model format)")
        except Exception:
            raw_json = {}
    parsed = parse_cord_to_schema(raw_json)
    # Append the cropped image to the gallery so the UI can show what Donut saw
    gallery = list(gallery) + [(rgb_for_donut, "→ Donut Input (cropped)")]
    return gallery, {
        "ocr_engine": "Donut",
        "raw_sequence": raw_seq,
        "raw_cord": raw_json,
        "parsed": parsed
    }, parsed

# ===============================
# TESSERACT OCR
# ===============================
def parse_tesseract_text(raw_text):
    lines = raw_text.strip().split('\n')
    items = []
    tax, service, discount, total, subtotal = 0.0, 0.0, 0.0, 0.0, 0.0

    summary_pats = {
        'tax': r'(?i)(?:tax|vat|pajak|ppn)\s*[:\s]*[\$€£¥₹Rp\s]*([0-9][0-9.,]*)',
        'service': r'(?i)(?:service\s*charge|servis|sc)\s*[:\s]*[\$€£¥₹Rp\s]*([0-9][0-9.,]*)',
        'discount': r'(?i)(?:discount|diskon|potongan)\s*[:\s]*[\$€£¥₹Rp\s]*([0-9][0-9.,]*)',
        'total': r'(?i)(?:(?:grand\s*)?total)\s*[:\s]*[\$€£¥₹Rp\s]*([0-9][0-9.,]*)',
        'subtotal': r'(?i)(?:sub\s*total)\s*[:\s]*[\$€£¥₹Rp\s]*([0-9][0-9.,]*)',
    }
    skip_pats = [
        r'(?i)^\s*invoice', r'(?i)^\s*date\s*(of|:)', r'(?i)^\s*seller', r'(?i)^\s*client',
        r'(?i)^\s*buyer', r'(?i)^\s*customer', r'(?i)^\s*tax\s*id', r'(?i)^\s*iban',
        r'(?i)^\s*phone', r'(?i)^\s*tel\s*:?', r'(?i)^\s*address', r'(?i)^\s*thank\s*you',
        r'(?i)^\s*receipt', r'(?i)^\s*bill\s*(no|num)', r'(?i)^\s*order\s*(no|num)',
        r'(?i)^\s*table\s*(no|num)', r'(?i)^\s*server', r'(?i)^\s*cashier',
        r'(?i)^\s*payment', r'(?i)^\s*change', r'(?i)^\s*cash\s*:?',
        r'(?i)^\s*credit\s*card', r'(?i)^\s*summary\s*$', r'(?i)^\s*items?\s*$',
        r'(?i)^\s*no\.\s+desc', r'(?i)^\s*qty\s+', r'^\s*[-=*]{3,}', r'^\s*$',
    ]

    for line in lines:
        line = line.strip()
        if not line: continue
        is_summary = False
        for key, pat in summary_pats.items():
            m = re.search(pat, line)
            if m:
                v = clean_price(m.group(1))
                if key == 'tax': tax = v
                elif key == 'service': service = v
                elif key == 'discount': discount = v
                elif key == 'total': total = v
                elif key == 'subtotal': subtotal = v
                is_summary = True; break
        if is_summary: continue
        skip = False
        for p in skip_pats:
            if re.search(p, line): skip = True; break
        if skip: continue

        numbers = re.findall(r'[\d][0-9.,]*[\d]|[\d]+', line)
        if not numbers: continue
        price = clean_price(numbers[-1])
        if price <= 0: continue
        first_num = re.search(r'\s+[\d]', line)
        item_name = line[:first_num.start()].strip() if first_num else line.strip()
        item_name = re.sub(r'^\d+[\.)\s]+', '', item_name).strip()
        if not item_name or len(item_name) < 2: continue

        qty = 1.0
        qm = re.search(r'(\d+)\s*[xX×]', line)
        if qm: qty = float(qm.group(1))
        elif len(numbers) >= 3:
            pq = clean_price(numbers[0])
            if 0 < pq <= 100: qty = pq
        items.append({"item_name": item_name, "item_quantity": qty, "item_price": price})

    return {
        "items": items, "subtotal": subtotal, "tax_amount": tax,
        "service_charge": service,
        "discount_details": {"type": "fixed" if discount > 0 else "none", "value": discount},
        "total_amount": total
    }

def run_tesseract_ocr(img_array, blur_type="Gaussian", blur_kernel=5, blur_sigma=0,
                      thresh_block=11, thresh_c=2, **filter_flags):
    gallery, thresh = preprocess_receipt(img_array, blur_type, blur_kernel, blur_sigma,
                                         thresh_block, thresh_c, **filter_flags)
    if not TESSERACT_AVAILABLE:
        return gallery, {"error": "Tesseract not installed"}, None
    if thresh is None:
        return gallery, {"error": "No image provided"}, None
    try:
        raw_text = pytesseract.image_to_string(Image.fromarray(thresh), lang='eng')
        parsed = parse_tesseract_text(raw_text)
        result_json = {"ocr_engine": "Tesseract", "raw_text": raw_text, "parsed": parsed}
        return gallery, result_json, parsed
    except Exception as e:
        return gallery, {"error": f"Tesseract failed: {e}"}, None

# ===============================
# HELPERS — Populate split bill from parsed data
# ===============================
def parsed_to_split_bill(parsed):
    if parsed is None:
        return pd.DataFrame(columns=["Item Name","Qty","Price","Assigned To"]), 0, 0, 0, 0
    rows = [{"Item Name": i["item_name"], "Qty": i["item_quantity"],
             "Price": i["item_price"], "Assigned To": ""} for i in parsed.get("items", [])]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Item Name","Qty","Price","Assigned To"])
    return (df, parsed.get("tax_amount", 0), parsed.get("service_charge", 0),
            parsed.get("discount_details", {}).get("value", 0), parsed.get("total_amount", 0))

def add_item(df):
    if df is None or df.empty:
        df = pd.DataFrame(columns=["Item Name","Qty","Price","Assigned To"])
    new = pd.DataFrame([{"Item Name": "", "Qty": 1, "Price": 0, "Assigned To": ""}])
    return pd.concat([df, new], ignore_index=True)

def remove_last_item(df):
    if df is None or len(df) == 0: return df
    return df.iloc[:-1].reset_index(drop=True)

def assign_all_unassigned(df, name):
    if df is None or df.empty or not name: return df
    df = df.copy()
    mask = df["Assigned To"].isna() | (df["Assigned To"].astype(str).str.strip() == "")
    df.loc[mask, "Assigned To"] = name.strip()
    return df

# ===============================
# OCR ACCURACY COMPARISON
# ===============================
def score_parsed_result(parsed):
    """Score a parsed OCR result 0-50. Higher = more likely accurate."""
    if not parsed:
        return 0, {}

    score = 0
    items = parsed.get("items", [])
    total = parsed.get("total_amount", 0)
    items_sum = sum(i["item_price"] for i in items)

    # Up to 20 pts: item count (2 per item, max 10 items)
    item_score = min(len(items) * 2, 20)
    score += item_score

    # 5 pts: total amount was detected
    if total > 0:
        score += 5

    # Up to 15 pts: how close is items_sum to total
    sum_ratio = 0.0
    if total > 0 and items_sum > 0:
        sum_ratio = min(items_sum, total) / max(items_sum, total)
        score += int(sum_ratio * 15)

    # Up to 10 pts: item name quality (length 3-60, no excessive special chars)
    name_quality = 0.0
    if items:
        ok = sum(
            1 for i in items
            if 3 <= len(str(i.get("item_name", ""))) <= 60
            and len(re.findall(r'[^a-zA-Z0-9\s\-\./,()&]', str(i.get("item_name", "")))) <= 2
        )
        name_quality = ok / len(items)
        score += int(name_quality * 10)

    return score, {
        "items_found": len(items),
        "items_sum": items_sum,
        "total_detected": total,
        "sum_accuracy_pct": round(sum_ratio * 100, 1),
        "name_quality_pct": round(name_quality * 100, 1),
    }


def compare_ocr_results(donut_parsed, tess_parsed):
    """Compare Donut and Tesseract results, return recommendation dict."""
    donut_score, donut_detail = score_parsed_result(donut_parsed)
    tess_score, tess_detail = score_parsed_result(tess_parsed)

    gap = abs(donut_score - tess_score)
    if donut_score > tess_score:
        winner = "donut"
    elif tess_score > donut_score:
        winner = "tesseract"
    else:
        winner = "tie"

    confidence = "high" if gap >= 10 else ("moderate" if gap >= 4 else "low")

    return {
        "winner": winner,
        "confidence": confidence,
        "donut_score": donut_score,
        "tess_score": tess_score,
        "donut_detail": donut_detail,
        "tess_detail": tess_detail,
    }


def calculate_split_bill(df, tax, service, discount, total, split_mode="After Tax"):
    if df is None or df.empty:
        return pd.DataFrame(columns=["Person","Items Total","Tax","Service","Discount","Total to Pay"]), "No items to split."
    # Guard against None from empty Streamlit number_input
    tax = float(tax or 0)
    service = float(service or 0)
    discount = float(discount or 0)
    total = float(total or 0)
    total_items_price = df["Price"].astype(float).sum()
    if total_items_price == 0:
        return pd.DataFrame(), "Total items price is 0."

    person_totals = {}
    for _, row in df.iterrows():
        price = float(row["Price"])
        assigned = str(row.get("Assigned To", "")).strip()
        if not assigned: continue
        persons = [p.strip() for p in assigned.split(",") if p.strip()]
        if not persons: continue
        split_price = price / len(persons)
        for p in persons:
            if p not in person_totals:
                person_totals[p] = {"items_cost": 0.0}
            person_totals[p]["items_cost"] += split_price

    if not person_totals:
        return pd.DataFrame(), "No one assigned to any items."

    num_people = len(person_totals)
    result_list = []
    for p, data in person_totals.items():
        proportion = data["items_cost"] / total_items_price
        if split_mode == "Before Tax (Equal Extras)":
            p_tax = tax / num_people
            p_svc = service / num_people
            p_disc = discount / num_people
        else:  # After Tax (Proportional)
            p_tax = tax * proportion
            p_svc = service * proportion
            p_disc = discount * proportion
        p_total = data["items_cost"] + p_tax + p_svc - p_disc
        result_list.append({
            "Person": p, "Items Total": round(data["items_cost"], 2),
            "Tax": round(p_tax, 2), "Service": round(p_svc, 2),
            "Discount": round(p_disc, 2), "Total to Pay": round(p_total, 2)
        })

    df_res = pd.DataFrame(result_list)
    calc_total = sum(r["Total to Pay"] for r in result_list)
    summary = f"**Calculated Total:** Rp {calc_total:,.0f} | **Receipt Total:** Rp {total:,.0f}"
    if total > 0 and abs(calc_total - total) > 1000:
        summary += "\n\n⚠️ *Warning: Calculated total differs significantly from receipt total.*"
    return df_res, summary
