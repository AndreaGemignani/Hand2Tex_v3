# Hand2TeX V2.8

# Hand2TeX V2.5 — TrustedRouter

Pipeline: **DETECT → PACK → DECODE → REBUILD**.

The selected models are unchanged:

- **Qwen-VL-OCR** for layout, text, formulas/matrices and tables.
- **Qwen3.8 Max** only as rescue for low-confidence decoding.
- Drawings/figures are preserved as original pixels.
- Final placement is deterministic LaTeX; no LLM is used to improvise the page layout.

The only change from V2.4 is the API provider: Qwen calls now go through the OpenAI-compatible **TrustedRouter** gateway instead of a direct Alibaba/DashScope account.

## Required environment variable

Create a TrustedRouter key and set:

```text
TRUSTEDROUTER_API_KEY=sk-tr-...
```

Defaults already included in `render.yaml`:

```text
TRUSTEDROUTER_BASE_URL=https://api.trustedrouter.com/v1
QWEN_OCR_MODEL=qwen/qwen-vl-ocr
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
docker build -t hand2tex-v25 .
docker run --rm -p 8000:10000 \
  -e TRUSTEDROUTER_API_KEY='YOUR_KEY' \
  hand2tex-v25
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
