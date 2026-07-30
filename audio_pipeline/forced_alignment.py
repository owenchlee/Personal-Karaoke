"""Word-level lyrics timing via CTC forced alignment.

Given lyrics text that's already known to be correct (e.g. from
lyrics_lookup.fetch_synced_lyrics, sourced from lrclib.net) but only
approximately timed -- lrclib gives line-level timestamps, and word timing
within a line is a linear character-weighted guess (see
lyrics_lookup._distribute_words) -- align that exact text against the real
vocal stem audio to get word-level timestamps grounded in this specific
recording. This replaces the previous approach of trying to correct
lrclib's own timestamps by fitting an intro-offset + proportional-drift
model (see NOTES.md's "Lyrics drifting out of sync" entry): that only ever
approximated a mismatch against a different release's timing, whereas
forced alignment measures this video's own audio directly.

Uses Meta's MMS wav2vec2 CTC model (multilingual, covers 1000+ languages
including English and Cantonese) via `transformers`, romanized through
`uroman` (the model's vocabulary is a small universal phonetic alphabet,
not native scripts), with the actual Viterbi alignment computed by
`torchaudio.functional.forced_align`.

This reimplements the relevant slice of the `ctc-forced-aligner` package
(https://github.com/MahmoudAshraf97/ctc-forced-aligner) rather than
depending on it directly: that package's C++ extension has no prebuilt
wheel for Windows and fails to build here without MSVC Build Tools.
`torchaudio.functional.forced_align` is the same underlying algorithm
(confirmed by cross-checking output against the original package's Python
API on real cached vocal stems) and already ships prebuilt with the
`torchaudio` install this project already has.
"""
import math
import re
import subprocess
import unicodedata
from pathlib import Path

import torch
import torchaudio
from transformers import AutoModelForCTC, AutoTokenizer
from uroman import Uroman

_MODEL_PATH = "MahmoudAshraf/mms-300m-1130-forced-aligner"
_SAMPLING_FREQ = 16000

# A long song is processed in windows the same way generate_emissions below
# already does (mirroring lyrics_extraction._CHUNK_SECONDS's own reasoning
# for chunking Whisper) -- 30s/2s are the upstream ctc-forced-aligner
# package's own defaults, unchanged here since there's no known failure mode
# specific to this project's audio that would call for different values.
_WINDOW_SECONDS = 30
_CONTEXT_SECONDS = 2
_BATCH_SIZE = 4

# This project's own internal language codes ("en"/"yue", see
# lyrics_extraction._SUPPORTED_LANGUAGES) vs. the ISO 639-3 codes uroman and
# the MMS model expect. English differs ("en" -> "eng"); Cantonese's real
# ISO 639-3 code already happens to be "yue".
_LANGUAGE_MAP = {"en": "eng", "yue": "yue"}

_uroman = Uroman()


def _load_audio(path: str, dtype: torch.dtype, device: str) -> torch.Tensor:
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(_SAMPLING_FREQ), "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    # frombuffer's result shares memory with `out` (read-only) -- clone before
    # any in-place op could touch it; none currently do, but relying on that
    # implicitly would be fragile.
    waveform = torch.frombuffer(bytearray(out), dtype=torch.int16).flatten()
    return waveform.to(dtype).to(device) / 32768.0


def _generate_emissions(model, audio_waveform: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Run ``model`` over ``audio_waveform`` in overlapping windows (a full
    song is far longer than the model's own effective receptive field) and
    return per-frame log-probabilities plus the stride (ms of audio each
    frame represents), used later to convert frame indices back to seconds.
    """
    ratio = model.config.inputs_to_logits_ratio
    window = _WINDOW_SECONDS * _SAMPLING_FREQ
    context = _CONTEXT_SECONDS * _SAMPLING_FREQ
    context_frames = context // ratio
    window_frames = window // ratio

    if audio_waveform.size(0) < window:
        extension = 0
        context = 0
        input_tensor = audio_waveform.unsqueeze(0)
    else:
        extension = math.ceil(audio_waveform.size(0) / window) * window - audio_waveform.size(0)
        padded = torch.nn.functional.pad(audio_waveform, (context, context + extension))
        input_tensor = padded.unfold(0, window + 2 * context, window)

    emissions_chunks = []
    with torch.inference_mode():
        for i in range(0, input_tensor.size(0), _BATCH_SIZE):
            batch = input_tensor[i : i + _BATCH_SIZE]
            emissions_chunks.append(model(batch).logits)

    emissions = torch.cat(emissions_chunks, dim=0)
    if context > 0:
        emissions = emissions[:, context_frames : context_frames + window_frames]
    emissions = emissions.flatten(0, 1)
    if extension > 0:
        emissions = emissions[: -(extension // ratio)]

    emissions = torch.log_softmax(emissions, dim=-1)
    # A <star> column, matching preprocess's <star> tokens -- required by the
    # alignment vocabulary even though this project never actually uses
    # star_frequency="segment" wildcards between every word.
    emissions = torch.cat([emissions, torch.zeros(emissions.size(0), 1, device=emissions.device)], dim=1)
    stride_ms = ratio * 1000 / _SAMPLING_FREQ
    return emissions, stride_ms


def _normalize_uroman(text: str) -> str:
    text = text.lower()
    text = re.sub(r"([^a-z' ])", " ", text)
    return re.sub(r" +", " ", text).strip()


def _romanize(tokens: list[str], iso_language: str) -> list[str]:
    romanized = []
    for token in tokens:
        normalized = unicodedata.normalize("NFKC", token.strip())
        out = _uroman.romanize_string(normalized, lcode=iso_language)
        out = re.sub(r"\s+", " ", " ".join(out.strip())).strip()
        romanized.append(_normalize_uroman(out))
    return romanized


class _Segment:
    __slots__ = ("label", "start", "end")

    def __init__(self, label: str, start: int, end: int):
        self.label, self.start, self.end = label, start, end


def _merge_repeats(path: list[int], idx_to_token: dict[int, str]) -> list[_Segment]:
    segments = []
    i1 = 0
    while i1 < len(path):
        i2 = i1
        while i2 < len(path) and path[i1] == path[i2]:
            i2 += 1
        segments.append(_Segment(idx_to_token[path[i1]], i1, i2 - 1))
        i1 = i2
    return segments


def _align(emissions: torch.Tensor, romanized_tokens: list[str], tokenizer) -> tuple[list[_Segment], str]:
    dictionary = {k.lower(): v for k, v in tokenizer.get_vocab().items()}
    dictionary["<star>"] = len(dictionary)
    token_indices = [
        dictionary[char] for char in " ".join(romanized_tokens).split(" ") if char in dictionary
    ]
    blank_label = dictionary.get("<blank>", tokenizer.pad_token_id)

    targets = torch.tensor([token_indices], dtype=torch.int32)
    path, _scores = torchaudio.functional.forced_align(
        emissions.unsqueeze(0).float(), targets, blank=blank_label
    )

    idx_to_token = {v: k for k, v in dictionary.items()}
    segments = _merge_repeats(path.squeeze(0).tolist(), idx_to_token)
    return segments, idx_to_token[blank_label]


def _spans_per_token(romanized_tokens: list[str], segments: list[_Segment], blank_label: str) -> list[list[_Segment]]:
    """Group ``segments`` (one per CTC frame, after merging consecutive
    repeats) back into one span per entry of ``romanized_tokens``, absorbing
    each token's own share of the blank/silence run on either side of it --
    ports ``ctc-forced-aligner``'s ``get_spans`` unchanged (see module
    docstring for why this project reimplements around it instead of
    depending on it directly).
    """
    char_idx = 0
    token_idx = 0
    intervals = []
    start = 0
    for seg_idx, seg in enumerate(segments):
        if token_idx == len(romanized_tokens):
            continue
        chars = romanized_tokens[token_idx].split(" ")
        if seg.label == blank_label:
            continue
        if char_idx == 0:
            start = seg_idx
        if char_idx == len(chars) - 1:
            char_idx = 0
            token_idx += 1
            intervals.append((start, seg_idx))
            while token_idx < len(romanized_tokens) and len(romanized_tokens[token_idx]) == 0:
                intervals.append((seg_idx, seg_idx))
                token_idx += 1
        else:
            char_idx += 1

    spans = []
    for idx, (start, end) in enumerate(intervals):
        span = segments[start : end + 1]
        if start > 0 and segments[start - 1].label == blank_label:
            prev = segments[start - 1]
            pad_start = prev.start if idx == 0 else (prev.start + prev.end) // 2
            span = [_Segment(blank_label, pad_start, span[0].start)] + span
        if end + 1 < len(segments) and segments[end + 1].label == blank_label:
            nxt = segments[end + 1]
            pad_end = nxt.end if idx == len(intervals) - 1 else math.floor((nxt.start + nxt.end) / 2)
            span = span + [_Segment(blank_label, span[-1].end, pad_end)]
        spans.append(span)
    return spans


# Loaded fresh on every call rather than cached at module scope, matching this project's existing
# convention for every other model-backed stage (WhisperModel in lyrics_extraction.py, Demucs's
# Separator in separation.py): the job server processes different songs concurrently in their own
# threads (see scripts/server.py's per-slug locking, which only serializes the *same* song), and a
# shared PyTorch model isn't guaranteed safe for concurrent forward passes from multiple threads.
# Reloading from the local Hugging Face cache (not re-downloading) is fast -- well under a second
# in practice, negligible next to the alignment inference itself.
def _load_model(device: str):
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCTC.from_pretrained(_MODEL_PATH, dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH, word_delimiter_token=None)
    return model, tokenizer


def align_tokens(tokens: list[str], audio_path: str | Path, language: str) -> list[dict]:
    """Force-align ``tokens`` (already tokenized, in the exact order they're
    sung) against ``audio_path``, returning ``[{"start", "end"}, ...]`` --
    one entry per input token, same order and length, grounded in this
    audio's own timing rather than any externally-reported timestamp.

    ``language`` is this project's internal code ("en" or "yue" -- see
    lyrics_extraction._SUPPORTED_LANGUAGES). Cantonese ``tokens`` are
    expected pre-split into individual characters (matching
    lyrics_lookup._tokenize_line's own CJK handling), since Cantonese has no
    whitespace word boundaries for this to align against either.
    """
    if not tokens:
        return []
    if language not in _LANGUAGE_MAP:
        raise ValueError(f"Unsupported language {language!r}; expected one of {sorted(_LANGUAGE_MAP)}")

    iso_language = _LANGUAGE_MAP[language]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = _load_model(device)

    audio_waveform = _load_audio(str(audio_path), model.dtype, device)
    emissions, stride_ms = _generate_emissions(model, audio_waveform)

    # A leading/trailing <star> "wildcard" token is required, not cosmetic: it gives the CTC
    # decoder a symbol that can absorb the instrumental intro/outro (and any other stretch that
    # isn't literally one of these words) without forcing the very first/last real word to
    # stretch to cover it -- dropping it (tried first) produced badly wrong spans, e.g. the first
    # word absorbing 30+ seconds of lead-in silence, confirmed directly against a real cached
    # vocal stem.
    romanized_starred = ["<star>"] + _romanize(tokens, iso_language) + ["<star>"]
    segments, blank_label = _align(emissions, romanized_starred, tokenizer)
    spans = _spans_per_token(romanized_starred, segments, blank_label)[1:-1]

    if len(spans) != len(tokens):
        raise RuntimeError(
            f"Forced alignment produced {len(spans)} spans for {len(tokens)} input tokens -- "
            "a tokenization mismatch between this project's tokens and the alignment model's "
            "own romanization, not a normal per-song failure."
        )

    results = []
    for span in spans:
        start_sec = span[0].start * stride_ms / 1000
        end_sec = (span[-1].end + 1) * stride_ms / 1000
        results.append({"start": round(start_sec, 4), "end": round(end_sec, 4)})
    return results
