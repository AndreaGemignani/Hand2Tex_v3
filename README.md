# Hand2TeX V2.9 DEBUG — TrustedRouter

Pipeline: **DETECT → PACK → DECODE → REBUILD**.

The selected models are unchanged:

- **Qwen-VL-OCR** for layout, text, formulas/matrices and tables.
- **Qwen3.8 Max** only as rescue for low-confidence decoding.
- Drawings/figures are preserved as original pixels.
- Final placement is deterministic LaTeX; no LLM is used to improvise the page layout.

Qwen calls go through the OpenAI-compatible **TrustedRouter** gateway. V2.9 adds layout diagnostics; the provider, models, request prompts, OCR parser and rendering architecture remain the same as V2.8.

## V2.9 DEBUG

Debug is enabled by default in the web form and API. Repeat the failing conversion and inspect these files at the root of the downloaded ZIP:

- `qwen_layout_raw.json`: complete JSON response for every Qwen layout request, grouped by zero-based page index. Every retry is retained with its HTTP status. Non-JSON responses keep the complete body in `response_text`; transport failures include the error.
- `request_payload_sanitized.json`: matching request attempts, including model, prompt, provider and `ocr_options`. Authorization headers are never recorded; API keys are redacted, and image base64 data is replaced with `[OMITTED]`. No outgoing request is changed.
- `parsed_layout.json`: actual normalized `words_info`, resulting blocks, usable box count, quality and fallback status for each page that uses Qwen layout.

Responses are saved **before parsing**, and all diagnostics are written before PDF compilation. Each conversion has its own collector. Diagnostics add no model calls or token costs. Qwen crop decoding and optional rescue responses are outside this layout trace.

Zero usable boxes now reports `quality: 0.0`, `status: "no_usable_boxes"` and `fallback: "whole_page_no_text"`. The original page is preserved, while the UI and manifest report a warning instead of claiming a successful transcription. A drawing-only page can also trigger this warning; no new classification heuristic is introduced.

If parsing, an OCR request or PDF compilation fails in debug mode, the API returns `hand2tex_debug.zip` containing the available diagnostics and `diagnostic_error.json`. Its HTTP status is 200 so the browser can download it; `X-Hand2TeX-Status: error` identifies the failed conversion. Successful conversions use `ok` or `warning`. This diagnostic ZIP may not contain a PDF.

Uncheck the debug option, send `include_debug=false`, or set `INCLUDE_DEBUG=false` to disable these files. With debug disabled, failed conversions retain the usual HTTP error response. Inspect the real response before adapting the parser.

## Required environment variable

Create a TrustedRouter key and set:

```text
TRUSTEDROUTER_API_KEY=sk-tr-...
```

Defaults already included in `render.yaml`:

```text
TRUSTEDROUTER_BASE_URL=https://api.trustedrouter.com/v1
QWEN_OCR_MODEL=qwen/qwen-vl-ocr-2025-11-20
QWEN_RESCUE_MODEL=qwen/qwen3.8-max
TRUSTEDROUTER_SORT=price
```

No Alibaba region, workspace ID or DashScope key is required.

## Render deploy

1. Upload the extracted repository contents to GitHub.
2. Render → New Blueprint → connect the repository.
3. Enter `TRUSTEDROUTER_API_KEY` when requested.
4. `MISTRAL_API_KEY` and `TYPESAFE_API_KEY` can remain empty.
5. Deploy.

## Output

Each conversion ZIP contains `main.pdf`, `main.tex`, `layout.json`, `decoded.json`, `manifest.json`, preserved image assets and `compile.log`.

`manifest.json > cost_estimate` contains the estimated Qwen spend for that conversion.

## Local test

```bash
docker build -t hand2tex-v29 .
docker run --rm -p 8000:10000 \
  -e TRUSTEDROUTER_API_KEY='YOUR_KEY' \
  hand2tex-v29
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```


## V2.8 layout compatibility fix

- Uses `qwen/qwen-vl-ocr-2025-11-20` for deterministic advanced-recognition support.
- Requests native `ocr_options.task=advanced_recognition` through TrustedRouter when supported.
- Falls back automatically to the documented prompt if the gateway rejects vendor-specific parameters.
- Parses both native `words_info` and OpenAI-compatible `pos_list`/`rotate_rect` output.
- Correctly converts rotated rectangles into axis-aligned page boxes.
- Position-only boxes are decoded afterward; math-like crops are automatically reprocessed as LaTeX math.
