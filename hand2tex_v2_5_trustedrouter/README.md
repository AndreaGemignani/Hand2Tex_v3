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
