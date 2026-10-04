import base64
import io
import json

import httpx
from PIL import Image
import pytest

from app.config import Settings
from app.models import BBox, ProcessingUnit
from app.services.content_review import ContentReviewClient, apply_review, plan_review_batches


def make_unit(tmp_path, identifier="a", *, draft="Original wording", category="text", page=0,
              box=(100, 100, 600, 140), size=(560, 100), members=None):
    path = tmp_path / f"{identifier}.png"
    Image.new("RGB", size, "white").save(path)
    item = ProcessingUnit(identifier, page, category, BBox(*box), members or [identifier], .95,
                          crop_path=str(tmp_path / "packed-ocr-crop-must-not-be-used.png"), decoded=draft)
    item.review_crop_path = str(path)
    item.review_source_bbox = BBox(box[0] - 30, box[1] - 30, box[2] + 30, box[3] + 30)
    item.review_target_bbox = BBox(30, 30, box[2] - box[0] + 30, box[3] - box[1] + 30)
    return item


def record(identifier="a", *, text="Original wording", category="text", status="verified", issues=None, sources=None):
    value = {"id": identifier, "text": text, "category": category, "status": status, "issues": issues or []}
    if sources is not None:
        value["source_ids"] = sources
    return value


def settings(monkeypatch):
    monkeypatch.setenv("TRUSTEDROUTER_API_KEY", "test-key")
    return Settings()


async def review(monkeypatch, units, records, *, response_status=200, content=None, seen=None, finish_reason="stop"):
    async def handler(request):
        if seen is not None:
            seen.update(json.loads(request.content))
        return httpx.Response(response_status, json={
            "choices": [{"message": {"content": content if content is not None else json.dumps({"units": records})},
                         "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 25},
        })
    client = ContentReviewClient(settings(monkeypatch), httpx.MockTransport(handler))
    return await client.review(units)


def test_batches_bound_count_pixels_and_pages_without_using_packed_crops(tmp_path):
    units = [make_unit(tmp_path, str(i), page=i // 4) for i in range(7)]
    batches = plan_review_batches(units, max_units=3)
    assert [[item.id for item in batch] for batch in batches] == [["0", "1", "2"], ["3"], ["4", "5", "6"]]
    pixel_batches = plan_review_batches(units[:4], max_pixels=160_000)
    assert all(len(batch) <= 2 for batch in pixel_batches)
    units[0].review_crop_path = ""
    assert all(units[0] not in batch for batch in plan_review_batches(units))


def test_full_page_fallback_missing_source_and_oversized_draft_are_not_scheduled(tmp_path):
    fullpage = make_unit(tmp_path, "full", members=["p0000_fullpage"])
    missing = make_unit(tmp_path, "missing")
    missing.review_crop_path = str(tmp_path / "no-file.png")
    oversized = make_unit(tmp_path, "oversized", draft="x" * 12_000)
    figure = make_unit(tmp_path, "figure", category="figure")
    good = make_unit(tmp_path, "good")
    assert plan_review_batches([fullpage, missing, oversized, figure, good]) == [[good]]


@pytest.mark.asyncio
async def test_review_sends_original_context_image_with_exact_target_metadata(monkeypatch, tmp_path):
    item = make_unit(tmp_path, draft="Nel risultato $x=3$.")
    seen = {}
    result = await review(monkeypatch, [item], [record(text=item.decoded)], seen=seen)
    assert list(result.records) == ["a"]
    assert seen["provider"]["sort"] == "price"
    assert seen["response_format"] == {"type": "json_object"}
    content = seen["messages"][0]["content"]
    assert "Do not solve" in content[0]["text"]
    metadata = json.loads(content[0]["text"].split("TARGET METADATA:\n")[1])
    assert metadata[0]["ocr_draft"] == item.decoded
    target = metadata[0]["target_region"]
    image_region = metadata[0]["image_region"]
    assert image_region[0] < target[0] < target[2] < image_region[2]
    assert image_region[1] < target[1] < target[3] < image_region[3]
    encoded = content[1]["image_url"]["url"].split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as sheet:
        assert sheet.width * sheet.height <= 3_000_000
    metadata = apply_review([item], result)
    assert item.decoded == item.raw_decoded == "Nel risultato $x=3$."
    assert metadata["a"]["status"] == "verified"


@pytest.mark.asyncio
async def test_corrected_reading_is_applied_and_original_is_retained(monkeypatch, tmp_path):
    item = make_unit(tmp_path, draft="c0ntenuto")
    result = await review(monkeypatch, [item], [record(text="contenuto", status="corrected", issues=["Source shows o, not zero"] )])
    apply_review([item], result)
    assert item.decoded == "contenuto"
    assert item.raw_decoded == "c0ntenuto"
    assert item.quality_review["issues"] == ["Source shows o, not zero"]


@pytest.mark.asyncio
async def test_uncertain_reading_never_overwrites_original(monkeypatch, tmp_path):
    item = make_unit(tmp_path, draft="x=?", category="math")
    result = await review(monkeypatch, [item], [record(text="x=3", category="math", status="uncertain", issues=["Last symbol could be 3 or 8"])])
    apply_review([item], result)
    assert item.decoded == item.raw_decoded == "x=?"
    assert item.quality_review["status"] == "uncertain"
    assert item.quality_review["proposed_text"] == "x=3"


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["not JSON", '{"units":[],"commentary":"guess"}', '{"units":"wrong"}'])
async def test_invalid_batch_output_preserves_paid_usage_and_draft(monkeypatch, tmp_path, content):
    item = make_unit(tmp_path)
    result = await review(monkeypatch, [item], [], content=content)
    assert result.usage == {"prompt_tokens": 200, "completion_tokens": 25}
    assert result.errors and not result.records
    apply_review([item], result)
    assert item.decoded == item.raw_decoded == "Original wording"


@pytest.mark.asyncio
async def test_unknown_duplicate_and_omitted_ids_cannot_override_units(monkeypatch, tmp_path):
    units = [make_unit(tmp_path, "a"), make_unit(tmp_path, "b"), make_unit(tmp_path, "c")]
    result = await review(monkeypatch, units, [record("a"), record("a"), record("unknown"), record("b")])
    assert set(result.records) == {"b"}
    metadata = apply_review(units, result)
    assert metadata["a"]["status"] == metadata["c"]["status"] == "uncertain"
    assert metadata["b"]["status"] == "verified"
    assert all(item.decoded == "Original wording" for item in units)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    record(text="changed while claiming verified"),
    record(text="", status="corrected", issues=["empty"]),
    record(status="corrected"),
    record(status="guessed"),
    dict(record(), extra="unexpected"),
    dict(record(), issues="wrong type"),
])
async def test_invalid_record_schema_or_status_is_rejected(monkeypatch, tmp_path, bad):
    item = make_unit(tmp_path)
    result = await review(monkeypatch, [item], [bad])
    assert not result.records and result.errors
    apply_review([item], result)
    assert item.decoded == "Original wording"


@pytest.mark.asyncio
async def test_mixed_prose_cannot_be_replaced_by_math_or_lose_inline_math(monkeypatch, tmp_path):
    item = make_unit(tmp_path, draft=r"Attenzione alla formula originale $\frac{1}{3}$ scritta qui.")
    for candidate in [record(text=r"\frac{1}{3}", category="math", status="corrected", issues=["Formula"]),
                      record(text="Attenzione alla formula originale scritta qui.", status="corrected", issues=["Formula"] )]:
        result = await review(monkeypatch, [item], [candidate])
        assert not result.records
        apply_review([item], result)
        assert item.decoded == item.raw_decoded


@pytest.mark.asyncio
async def test_adjacent_split_fraction_merges_without_losing_original_fragments(monkeypatch, tmp_path):
    numerator = make_unit(tmp_path, "a", draft="1", category="math", box=(400, 100, 460, 125))
    denominator = make_unit(tmp_path, "b", draft=r"\sqrt{3}", category="math", box=(395, 130, 465, 155))
    result = await review(monkeypatch, [numerator, denominator], [
        record("a", text=r"\frac{1}{\sqrt{3}}", category="math", status="corrected",
               issues=["Visible fraction bar joins both source targets"], sources=["a", "b"]),
    ])
    assert not result.errors
    metadata = apply_review([numerator, denominator], result)
    assert numerator.decoded == r"\frac{1}{\sqrt{3}}"
    assert denominator.decoded == ""
    assert numerator.raw_decoded == "1" and denominator.raw_decoded == r"\sqrt{3}"
    assert metadata["b"]["merged_into"] == "a"


@pytest.mark.asyncio
async def test_split_formula_with_attached_short_note_can_be_merged_as_mixed_prose(monkeypatch, tmp_path):
    numerator = make_unit(tmp_path, "a", draft=r"1 = \begin{matrix}2&3\end{matrix} Attenzione se si", category="math", box=(100, 100, 700, 140))
    denominator = make_unit(tmp_path, "b", draft=r"\sqrt{3}", category="math", box=(120, 145, 190, 170))
    fixed = r"Attenzione se si usa $\frac{1}{\sqrt{3}}$."
    result = await review(monkeypatch, [numerator, denominator], [
        record("a", text=fixed, status="corrected", issues=["Fraction and attached note share source line"], sources=["a", "b"]),
    ])
    assert not result.errors
    apply_review([numerator, denominator], result)
    assert numerator.category == "text" and numerator.decoded == fixed
    assert denominator.quality_review["merged_into"] == "a"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["nonadjacent", "different-page", "separate-note", "duplicate-record", "missing-prose"])
async def test_unsafe_merges_leave_every_source_intact(monkeypatch, tmp_path, mode):
    a = make_unit(tmp_path, "a", draft="1", category="math", box=(100, 100, 160, 125))
    b = make_unit(tmp_path, "b", draft="3", category="math", box=(100, 130, 160, 155))
    if mode == "nonadjacent":
        b.bbox = b.review_target_bbox = BBox(700, 900, 760, 925)
    elif mode == "different-page":
        b.page = 1
    elif mode == "separate-note":
        b.category, b.decoded = "text", "Una nota separata importante"
    elif mode == "missing-prose":
        a.decoded = "1 = 2 Attenzione se si"
    records = [record("a", text=r"\frac{1}{3}", category="math", status="corrected", issues=["Fraction"], sources=["a", "b"])]
    if mode == "duplicate-record":
        records.append(record("b", text=b.decoded, category=b.category))
    result = await review(monkeypatch, [a, b], records)
    assert not result.records
    originals = [a.decoded, b.decoded]
    apply_review([a, b], result)
    assert [a.decoded, b.decoded] == originals
    assert all("merged_into" not in item.quality_review for item in [a, b])


@pytest.mark.asyncio
async def test_unsupported_json_format_does_not_retry_or_lose_usage(monkeypatch, tmp_path):
    item = make_unit(tmp_path)
    calls = []
    async def handler(request):
        calls.append(request)
        return httpx.Response(400, json={"error": "unsupported response_format", "usage": {"prompt_tokens": 7}})
    client = ContentReviewClient(settings(monkeypatch), httpx.MockTransport(handler))
    result = await client.review([item])
    assert len(calls) == 1 and result.usage == {"prompt_tokens": 7}
    assert result.errors == ["TrustedRouter review HTTP 400"]
    apply_review([item], result)
    assert item.decoded == "Original wording"


@pytest.mark.asyncio
async def test_crop_relative_target_is_not_shifted_by_source_page_origin(monkeypatch, tmp_path):
    item = make_unit(tmp_path, box=(1200, 1600, 1700, 1640))
    seen = {}
    await review(monkeypatch, [item], [record()], seen=seen)
    metadata = json.loads(seen["messages"][0]["content"][0]["text"].split("TARGET METADATA:\n")[1])[0]
    assert metadata["source_bbox"] == [1170, 1570, 1730, 1670]
    assert metadata["target_bbox"] == [30, 30, 530, 70]
    image = metadata["image_region"]
    target = metadata["target_region"]
    assert target[0] == image[0] + 30
    assert target[1] == image[1] + 30
    assert image[0] < target[0] < target[2] < image[2]
    assert image[1] < target[1] < target[3] < image[3]


@pytest.mark.asyncio
async def test_token_limit_rejects_even_valid_json_and_retains_usage(monkeypatch, tmp_path):
    item = make_unit(tmp_path)
    result = await review(monkeypatch, [item], [record(text="truncated", status="corrected", issues=["Cut short"])], finish_reason="length")
    assert not result.records and "token limit" in result.errors[0]
    assert result.usage == {"prompt_tokens": 200, "completion_tokens": 25}
    apply_review([item], result)
    assert item.decoded == item.raw_decoded == "Original wording"


@pytest.mark.asyncio
@pytest.mark.parametrize("formula", [
    r"\frac{x}{y", r"\begin{matrix}1&2\end{aligned}", r"$x=3",
    r"\begin{matrix}1&2", r"\begin{document}x\end{document}",
])
async def test_invalid_math_syntax_cannot_overwrite_raw_formula(monkeypatch, tmp_path, formula):
    item = make_unit(tmp_path, draft="x=?", category="math")
    result = await review(monkeypatch, [item], [record(text=formula, category="math", status="corrected", issues=["Source formula"])])
    assert not result.records and result.errors
    apply_review([item], result)
    assert item.decoded == item.raw_decoded == "x=?"


@pytest.mark.asyncio
async def test_invalid_inline_math_preserves_the_mixed_prose_draft(monkeypatch, tmp_path):
    item = make_unit(tmp_path, draft=r"Il risultato $\frac{x}{y}$ è questo.")
    result = await review(monkeypatch, [item], [
        record(text=r"Il risultato $\frac{x}{y$ è questo.", status="corrected", issues=["Formula"]),
    ])
    assert not result.records and "braces" in result.errors[0]
    apply_review([item], result)
    assert item.decoded == item.raw_decoded


@pytest.mark.asyncio
async def test_valid_aligned_formula_remains_accepted(monkeypatch, tmp_path):
    item = make_unit(tmp_path, draft="x=?", category="math")
    fixed = r"\begin{aligned}a&=\frac{1}{\sqrt{3}}\\ b&=2\end{aligned}"
    result = await review(monkeypatch, [item], [record(text=fixed, category="math", status="corrected", issues=["Two visible equations"])])
    assert not result.errors
    apply_review([item], result)
    assert item.decoded == fixed


@pytest.mark.asyncio
async def test_invalid_duplicate_record_also_invalidates_a_valid_claim_for_same_id(monkeypatch, tmp_path):
    item = make_unit(tmp_path)
    result = await review(monkeypatch, [item], [record(), record(status="bad-status")])
    assert not result.records
    apply_review([item], result)
    assert item.decoded == item.raw_decoded


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [r"\alpha", r"\beta\approx 2", r"\sqrt{3}", r"Nota \alpha=\beta."])
async def test_naked_math_commands_are_not_claimed_as_renderable_text(monkeypatch, tmp_path, raw):
    item = make_unit(tmp_path, draft=raw)
    result = await review(monkeypatch, [item], [record(text=raw)])
    assert not result.records and "outside a math delimiter" in result.errors[0]
    apply_review([item], result)
    assert item.decoded == item.raw_decoded == raw


@pytest.mark.asyncio
async def test_wrapping_naked_math_as_inline_formula_is_an_accepted_correction(monkeypatch, tmp_path):
    item = make_unit(tmp_path, draft=r"Nota \alpha=\beta.")
    fixed = r"Nota $\alpha=\beta$."
    result = await review(monkeypatch, [item], [record(text=fixed, status="corrected", issues=["Greek symbols need inline math"])])
    assert not result.errors
    apply_review([item], result)
    assert item.decoded == fixed


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [
    r"Prima \begin{equation}\alpha=\beta\end{equation} dopo.",
    r"Usa C:\work\beta\notes.tex e poi continua.",
    r"File assets\alpha.tex e la cartella \\server\beta\notes.",
    r"Un normale \backslash nel testo.",
])
async def test_supported_math_environment_and_normal_paths_remain_valid_prose(monkeypatch, tmp_path, raw):
    item = make_unit(tmp_path, draft=raw)
    result = await review(monkeypatch, [item], [record(text=raw)])
    assert not result.errors
    apply_review([item], result)
    assert item.decoded == raw and item.quality_review["status"] == "verified"


def test_fraction_cluster_moves_whole_group_past_six_target_batch_boundary(tmp_path):
    units = [make_unit(tmp_path, str(i), box=(20, i * 150, 700, i * 150 + 40)) for i in range(5)]
    first = make_unit(tmp_path, "numerator", draft="Attenzione se si usa C_L = 0.58 NON", box=(930, 900, 1551, 976))
    second = make_unit(tmp_path, "denominator", draft=r"\sqrt{3}", category="math", box=(1012, 977, 1077, 1025))
    units.extend([first, second])
    batches = plan_review_batches(units, max_units=6)
    assert [[item.id for item in batch] for batch in batches] == [["0", "1", "2", "3", "4"], ["numerator", "denominator"]]
    assert all(len(batch) <= 6 for batch in batches)
    assert units[-2:] == [first, second]


def test_source_fraction_cluster_can_skip_independent_long_paragraph_in_request_order(tmp_path):
    units = [make_unit(tmp_path, str(i), box=(20, i * 100, 700, i * 100 + 40)) for i in range(5)]
    first = make_unit(tmp_path, "T5", draft=r"1=2 \begin{matrix}2&3\end{matrix} Attenzione se si", category="math", box=(930, 659, 1551, 735))
    paragraph = make_unit(tmp_path, "T6", draft="An independent paragraph in the left column. " * 5,
                          box=(44, 668, 932, 1074), members=[f"line{i}" for i in range(6)])
    denominator = make_unit(tmp_path, "T7", draft=r"\sqrt{3}", box=(1012, 736, 1077, 784))
    note = make_unit(tmp_path, "T8", draft="usa C_L = 0,58 NON", box=(1160, 739, 1554, 790))
    units.extend([first, paragraph, denominator, note])
    batches = plan_review_batches(units, max_units=6)
    assert [[item.id for item in batch] for batch in batches] == [["0", "1", "2", "3", "4"], ["T5", "T7", "T8", "T6"]]
    assert [item.id for item in units[-4:]] == ["T5", "T6", "T7", "T8"]


def test_atomic_expression_that_cannot_fit_the_budget_remains_unscheduled(tmp_path):
    a = make_unit(tmp_path, "a", draft="1", category="math", box=(400, 100, 460, 125), size=(560, 100))
    b = make_unit(tmp_path, "b", draft=r"\sqrt{3}", category="math", box=(395, 130, 465, 155), size=(560, 100))
    assert not plan_review_batches([a, b], max_units=1)
    assert not plan_review_batches([a, b], max_pixels=100_000)
    assert not plan_review_batches([a, b], max_input_chars=150)


@pytest.mark.asyncio
async def test_merged_leader_geometry_covers_all_sources_but_keeps_original_review_target(monkeypatch, tmp_path):
    a = make_unit(tmp_path, "a", draft="1", category="math", box=(400, 100, 460, 125))
    b = make_unit(tmp_path, "b", draft=r"\sqrt{3}", category="math", box=(395, 130, 465, 155))
    target = a.review_target_bbox
    result = await review(monkeypatch, [a, b], [record("a", text=r"\frac{1}{\sqrt{3}}", category="math", status="corrected",
                           issues=["Visible fraction"], sources=["a", "b"])])
    assert not result.errors
    apply_review([a, b], result)
    assert a.bbox == BBox(395, 100, 465, 155)
    assert len(a.polygons) == 2
    assert a.polygons[0] == [400, 100, 460, 100, 460, 125, 400, 125]
    assert a.polygons[1] == [395, 130, 465, 130, 465, 155, 395, 155]
    assert a.review_target_bbox == target
