#NOTA PER ME molto difficile avere un output pulito come se fosse stato scritto al computer, problemi di interpretazione testo e formule risultano in parole prive di senso, l'obiettivo dell'app era creare una versione di mathpix con modelli ai potenti e tagliare i costi sull'aggregazione prima della chiamata api, test fallito miseramente, congratulazioni a mathpix

# Hand2TeX V2.15 DEBUG — TrustedRouter

Convert handwritten notes into a normal, editable LaTeX document: paragraphs, formulas, tables and original drawings in reading order, with regular margins and typography.

Pipeline: **DETECT → PACK → DECODE → REVIEW → COMPOSE**.

The selected models are unchanged:

- **Qwen-VL-OCR** for layout, text, formulas/matrices and tables.
- **Qwen3.8 Max** for image-based content review of all transcribed regions, enabled by default, and as rescue for structurally invalid decoding.
- Drawings/figures are preserved as original pixels and inserted into the document as images.
- Detected geometry helps isolate and pack OCR crops, determine reading order and retain broad structural relationships. It does not fix the output text to scan coordinates.
- LaTeX lays out the decoded content naturally, with normal paragraphs, inline or displayed mathematics, tables and figures.

Qwen calls go through the OpenAI-compatible **TrustedRouter** gateway. The existing models are retained, with Qwen3.8 Max now also checking transcription against the source image before composition.

## V2.15: verify the content against the scan

The captured V2.14 result compiled successfully but contained incorrect words, changed mathematical coefficients and an invented matrix. Every region received a structural score of `1.0`; those scores check text or LaTeX syntax and are not confidence estimates of transcription accuracy.

Content review now checks every typed region against an image crop with nearby source context, in bounded batches. The reviewer can recover mixed prose and mathematics, preserve headings and report uncertain readings. Corrections must be supported by the visible handwriting: the task is transcription, not rewriting the author's ideas, fixing their mathematics or completing missing material from subject knowledge. Uncertain or failed reviews are reported and the original page is attached separately for inspection.

Uncertain regions carry the visible note “Trascrizione da verificare sull’originale.” beside their content in the PDF. Regions combined into a reviewed expression are printed once; their original OCR remains in the diagnostic records. Verified content continues through normal paragraphs without added notes.

`decoded.json` retains the original OCR and reviewed result. `quality_review.json` records review decisions and changes so the content can be checked without repeating the conversion. Review uses additional API calls; the cost estimate includes reported token usage from all review responses, including responses rejected as invalid, rather than only accepted corrections. An image-based review can still make mistakes; its accuracy needs to be assessed on the next real conversion.

The [documented native OCR tasks](https://www.alibabacloud.com/help/en/model-studio/qwen-vl-ocr-api-reference) are forwarded through TrustedRouter when supported, with a compatible prompt-based fallback if the gateway rejects them. The final document keeps the normal flowing LaTeX layout introduced in V2.14.

Content review settings:

```text
ENABLE_CONTENT_REVIEW=true
CONTENT_REVIEW_MODEL=qwen/qwen3.8-max
CONTENT_REVIEW_BATCH_SIZE=6
CONTENT_REVIEW_MAX_PIXELS=3000000
CONTENT_REVIEW_MAX_INPUT_CHARS=12000
CONTENT_REVIEW_MAX_OUTPUT_TOKENS=4096
```

When `CONTENT_REVIEW_MODEL` is unset, review uses `QWEN_RESCUE_MODEL`. Set `ENABLE_CONTENT_REVIEW=false` to disable this extra review stage. The limits bound each review request's image pixels, supplied text and generated response.

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

Each successful conversion ZIP contains the typeset `main.pdf`, editable `main.tex`, `layout.json`, `decoded.json`, `manifest.json`, `quality_review.json`, preserved drawing assets and `compile.log`. With content review enabled, `review_crops/` contains the original context crops used as evidence; debug output also includes sanitized `quality_review_raw.json`. If review is disabled, its report explicitly records that no content verification was performed. The LaTeX source contains the final content in document order; layout coordinates remain available as diagnostic metadata.

When a conversion has warnings, `manifest.json` lists `content_warnings` and `source_pages`. Original pages in `sources/` are review attachments and are not inserted as page backgrounds or substitutes for transcribed content.

`manifest.json > cost_estimate` contains the estimated Qwen spend for that conversion.

## Local test

```bash
docker build -t hand2tex-v215 .
docker run --rm -p 8000:10000 \
  -e TRUSTEDROUTER_API_KEY='YOUR_KEY' \
  hand2tex-v215
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
