# Hand2TeX V2.14 DEBUG — TrustedRouter

Convert handwritten notes into a normal, editable LaTeX document: paragraphs, formulas, tables and original drawings in reading order, with regular margins and typography.

Pipeline: **DETECT → PACK → DECODE → COMPOSE**.

The selected models are unchanged:

- **Qwen-VL-OCR** for layout, text, formulas/matrices and tables.
- **Qwen3.8 Max** only as rescue for low-confidence decoding.
- Drawings/figures are preserved as original pixels and inserted into the document as images.
- Detected geometry helps isolate and pack OCR crops, determine reading order and retain broad structural relationships. It does not fix the output text to scan coordinates.
- LaTeX lays out the decoded content naturally, with normal paragraphs, inline or displayed mathematics, tables and figures.

Qwen calls go through the OpenAI-compatible **TrustedRouter** gateway. Provider, models and request prompts are unchanged.

## V2.14: a standard flowing document

The output now reads like a document written on a computer, from beginning to end. Decoded text flows through normal LaTeX paragraphs; mathematics and tables use their native environments, and preserved drawings appear with the related content. Margins, wrapping and pagination are controlled by LaTeX instead of the original pixel coordinates. Source geometry guides content order and grouping, so the broad organization can remain familiar without reproducing each handwritten position.

The scan is no longer the background of a reconstructed page with replacement text boxes. Layout detection, straightened line crops and OCR packing still reduce the material sent for decoding. This rendering change adds no model call. If content cannot be transcribed or a source fragment needs review, the conversion reports a warning and attaches the original page in the ZIP's `sources/` directory, separately from the typeset document.

The notes below describe earlier versions; their positioned page rendering has been superseded by V2.14.

## V2.13: line crops and readable positioned text

The first successful PDF exposed overlapping OCR text, oversized fonts and literal math commands. Rotated layout corners are now preserved alongside the existing placement envelopes. Text/math crops exclude pixels outside their line polygons and are straightened before decoding. Overlapping polygons within a group are cropped as one union so shared handwriting is not copied twice; separate lines can still be packed into one OCR request.

Grouping now uses actual line geometry, keeps columns separate and limits text groups to six lines and one fifth of a page. Equal-score duplicate boxes retain one stable representative. The renderer measures text and formulas with TeX and shrinks them to their available width and height, including the space before the next overlapping block. Explicit math delimiters in text are rendered as formulas without another OCR request.

The residual layer uses line polygons and a neutral paper color instead of a color averaged from highlighted corners. Drawings outside the removed text regions retain their original pixels. Synthetic tests cover rotated crops, overlapping groups, drawing preservation, inline formulas and rendered PDF bounds.

Provider, models and prompts remain unchanged. Validation scores check structure and do not establish transcription accuracy; the next conversion is needed to assess OCR quality with the improved crops.

## V2.12: display math inside positioned boxes

The next captured run reached formula rendering but failed with `Bad math environment delimiter`. The mathematical OCR returned both an `equation` and an `align*` environment; each opens display math, so neither can be placed directly inside the renderer's existing math box.

The renderer now removes `equation`/`equation*` and `displaymath` wrappers and converts multiline display environments to their inner equivalents (`align` to `aligned`, `gather`/`multline` to `gathered`, and `alignat` to `alignedat`). Formula content, alignment markers, matrix environments and box positions are preserved. Synthetic regression cases reproduce both structures from the captured output and compile with real TeX Live in GitHub Actions.

`manifest.json` now records the PDF compilation outcome and changes its status to `error` if compilation fails, matching `diagnostic_error.json`. Saved decoding and cost details remain available. This correction changes no model, prompt or provider and performs no new OCR requests.

## V2.11: OCR paragraphs and math rendering

The captured V2.10 run correctly detected 43 boxes, but PDF compilation failed with `There's no line here to end`. OCR blank lines had generated consecutive LaTeX line-break commands. Text boxes now use explicit paragraphs, so CRLF input and empty OCR lines do not cause that error.

Math validation now returns an exact maximum score of `1.0`: floating-point rounding previously caused fully valid formula decoding to lose against a text score of `1.0`. Formulas are now eligible for the existing automatic math routing. Paragraphs with substantial prose remain in the text path; a few embedded symbols do not cause a whole paragraph to be replaced by formula-only OCR. The renderer removes external inline/display math delimiters before adding its own math wrapper.

`decoded.json`, `layout.json`, `manifest.json` and the cost estimate are saved before compilation, including in a failed diagnostic ZIP. This exposes the actual OCR/routing result without needing another paid OCR request just to retrieve it.

The GitHub test workflow installs the same TeX Live packages as production and runs the suite on Python 3.11, including real PDF compilation tests for OCR paragraphs and formulas. Models and provider are unchanged; these fixes add no API requests.

## V2.10: parse the observed layout response

The real TrustedRouter response contained **44 comma-separated rows**, each with `cx,cy,width,height,angle`, instead of a JSON `words_info` or `pos_list` object. The previous parser discarded this listing and preserved the whole page as an image.

The parser now accepts a complete numeric CSV listing, including an optional CSV/text code fence. In this output, the first four values use the model's normalized 0..1000 coordinate space. It computes the rotated rectangle's envelope in that space, then scales each axis to the original page dimensions and clips it to the page. Native JSON coordinates retain their existing interpretation. Invalid rows, non-finite coordinates and non-positive dimensions are rejected.

The captured 1663×2420 test page now yields **44 usable boxes**. Their positions were checked against the original page, including its final line near the bottom. The coordinate-only regression fixture contains no image, handwriting transcription, credentials or request IDs.

These boxes do not contain recognized text: the existing grouped crop decoding stage performs the transcription and math decoding afterward. RAW diagnostics remain enabled by default, and no additional model call is introduced by the parser.

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

Each successful conversion ZIP contains the typeset `main.pdf`, editable `main.tex`, `layout.json`, `decoded.json`, `manifest.json`, preserved drawing assets and `compile.log`. The LaTeX source contains the decoded content in document order; layout coordinates remain available as diagnostic metadata.

When a conversion has warnings, `manifest.json` lists `content_warnings` and `source_pages`. Original pages in `sources/` are review attachments and are not inserted as page backgrounds or substitutes for transcribed content.

`manifest.json > cost_estimate` contains the estimated Qwen spend for that conversion.

## Local test

```bash
docker build -t hand2tex-v214 .
docker run --rm -p 8000:10000 \
  -e TRUSTEDROUTER_API_KEY='YOUR_KEY' \
  hand2tex-v214
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
