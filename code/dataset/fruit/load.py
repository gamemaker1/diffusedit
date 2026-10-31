"""Loader for the FRUIT-Wiki subset (Logan et al., NAACL 2022).

Pure Python, no TensorFlow. Reads the processed EdiT5-format TFRecord files
from gs://gresearch/FRUIT/dataset and parses their marker grammar:

  inputs:  "[0] sent0 [1] sent1 ... [CONTEXT] (0) evidence0 (1) evidence1 ..."
  targets: sequence of
             "[i]"            -> copy source sentence i unchanged
             "(j) (k) text"   -> generated (new or rewritten) sentence,
                                 supported by evidence j and k
           source sentences whose marker is absent were deleted or rewritten.

The reconstruction logic mirrors rendering_utils.normalize() in
github.com/google-research/language/language/fruit (delimiters are matched
anywhere in the string, as the original code does — a bracketed number
occurring naturally in prose is misparsed identically by both).

Public API:
  read_tfrecords(path)   -> yields {"inputs": str, "targets": str, ...}
  parse_instance(feats)  -> normalized dict (see parse_instance docstring)
  load_jsonl(path)       -> yields dicts from a .jsonl file
"""

import difflib
import json
import re
import struct

CONTEXT_MARKER = "[CONTEXT]"
_SENT_RE = re.compile(r"\[(\d+)\]")
_EVID_RE = re.compile(r"\((\d+)\)")

# ---------------------------------------------------------------------------
# TFRecord framing: [uint64 length][uint32 masked crc of length]
#                   [data]         [uint32 masked crc of data]
# ---------------------------------------------------------------------------

_CRC_TABLE = []


def _crc32c_table():
    if not _CRC_TABLE:
        poly = 0x82F63B78  # Castagnoli, reflected
        for i in range(256):
            crc = i
            for _ in range(8):
                crc = (crc >> 1) ^ poly if crc & 1 else crc >> 1
            _CRC_TABLE.append(crc)
    return _CRC_TABLE


def _crc32c(data):
    table = _crc32c_table()
    crc = 0xFFFFFFFF
    for byte in data:
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _masked_crc(data):
    crc = _crc32c(data)
    return ((crc >> 15) | (crc << 17)) + 0xA282EAD8 & 0xFFFFFFFF


def read_tfrecord_bytes(path, check_crc=False):
    """Yields raw record payloads from a TFRecord file."""
    with open(path, "rb") as f:
        while True:
            header = f.read(12)
            if not header:
                return
            if len(header) < 12:
                raise IOError(f"truncated TFRecord header in {path}")
            (length,), (length_crc,) = (
                struct.unpack("<Q", header[:8]),
                struct.unpack("<I", header[8:]),
            )
            data = f.read(length)
            (data_crc,) = struct.unpack("<I", f.read(4))
            if len(data) < length:
                raise IOError(f"truncated TFRecord payload in {path}")
            if check_crc:
                if _masked_crc(header[:8]) != length_crc:
                    raise IOError("length CRC mismatch")
                if _masked_crc(data) != data_crc:
                    raise IOError("data CRC mismatch")
            yield data


# ---------------------------------------------------------------------------
# Minimal protobuf wire-format parsing for tf.train.Example:
#   Example{1: Features}; Features{1: repeated MapEntry{1: key, 2: Feature}}
#   Feature{1: BytesList, 2: FloatList, 3: Int64List}; BytesList{1: repeated}
# ---------------------------------------------------------------------------


def _read_varint(buf, pos):
    result = shift = 0
    while pos < len(buf):
        b = buf[pos]
        result |= (b & 0x7F) << shift
        shift += 7
        pos += 1
        if not b & 0x80:
            return result, pos
    raise ValueError("unterminated varint")


def _fields(buf):
    """Yields (field_number, wire_type, value) for a message payload."""
    pos = 0
    while pos < len(buf):
        tag, pos = _read_varint(buf, pos)
        field, wire = tag >> 3, tag & 7
        if wire == 0:  # varint
            value, pos = _read_varint(buf, pos)
        elif wire == 2:  # length-delimited
            length, pos = _read_varint(buf, pos)
            value = buf[pos:pos + length]
            pos += length
        elif wire == 5:  # 32-bit
            value = buf[pos:pos + 4]
            pos += 4
        elif wire == 1:  # 64-bit
            value = buf[pos:pos + 8]
            pos += 8
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield field, wire, value


def parse_example(record):
    """Parses a serialized tf.train.Example into {name: [bytes, ...]}."""
    features = {}
    for field, _, value in _fields(record):  # Example
        if field != 1:
            continue
        for f2, _, entry in _fields(value):  # Features.feature map
            if f2 != 1:
                continue
            key, feat = None, b""
            for f3, _, v3 in _fields(entry):  # MapEntry
                if f3 == 1:
                    key = v3.decode("utf-8")
                elif f3 == 2:
                    feat = v3
            values = []
            for f4, _, v4 in _fields(feat):  # Feature
                if f4 == 1:  # BytesList
                    values.extend(
                        v5 for f5, _, v5 in _fields(v4) if f5 == 1)
                elif f4 == 3:  # Int64List (repeated varint, maybe packed)
                    for f5, w5, v5 in _fields(v4):
                        if f5 != 1:
                            continue
                        if w5 == 0:
                            values.append(v5)
                        else:  # packed
                            pos = 0
                            while pos < len(v5):
                                n, pos = _read_varint(v5, pos)
                                values.append(n)
                else:
                    raise ValueError(
                        f"feature '{key}' has unsupported type {f4}")
            features[key] = values
    return features


def read_tfrecords(path, check_crc=False):
    """Yields feature dicts {name: str} from a FRUIT TFRecord file."""
    def as_python(v):
        decoded = [x.decode("utf-8") if isinstance(x, bytes) else x for x in v]
        return decoded[0] if len(decoded) == 1 else decoded

    for record in read_tfrecord_bytes(path, check_crc=check_crc):
        yield {k: as_python(v) for k, v in parse_example(record).items()}


# ---------------------------------------------------------------------------
# Marker-grammar parsing
# ---------------------------------------------------------------------------


def _sequential_split(text, fmt):
    """Splits text on markers fmt.format(0), fmt.format(1), ... in order.

    Returns the list of segments following each marker. Only the first
    occurrence of each expected next marker (searching left to right) is
    treated as a delimiter, so e.g. a stray "(2)" occurring before marker
    "(1)" is left inside the preceding segment.
    """
    start = text.find(fmt.format(0))
    if start < 0:
        return []
    segments = []
    uid = 0
    pos = start + len(fmt.format(0))
    while True:
        nxt = text.find(fmt.format(uid + 1), pos)
        if nxt < 0:
            segments.append(text[pos:].strip())
            return segments
        segments.append(text[pos:nxt].strip())
        uid += 1
        pos = nxt + len(fmt.format(uid))


def parse_input(inputs):
    """Splits an `inputs` string into source sentences and evidence items."""
    if CONTEXT_MARKER in inputs:
        source_part, evidence_part = inputs.split(CONTEXT_MARKER, 1)
    else:
        source_part, evidence_part = inputs, ""
    source_sentences = _sequential_split(source_part, "[{}]")
    evidence = []
    for j, segment in enumerate(_sequential_split(evidence_part, "({})")):
        title, _, rest = segment.partition(" ")
        evidence.append({
            "marker": j,
            # Page title as rendered (underscores for spaces). The section
            # name and passage text follow with no reliable separator, so
            # they are kept together in "content".
            "title": title,
            "content": rest.strip(),
            "raw": segment,
        })
    return source_sentences, evidence


def parse_target(targets, n_source, n_evidence):
    """Parses a `targets` string into a list of ops.

    Ops are, in output order:
      {"op": "copy", "source_idx": i}
      {"op": "generate", "text": str, "evidence_refs": [j, ...]}
    Markers with out-of-range indices are treated as plain text.
    """
    matches = []
    for m in _SENT_RE.finditer(targets):
        if int(m.group(1)) < n_source:
            matches.append(("copy", int(m.group(1)), m.start(), m.end()))
    for m in _EVID_RE.finditer(targets):
        if int(m.group(1)) < n_evidence:
            matches.append(("ref", int(m.group(1)), m.start(), m.end()))
    matches.sort(key=lambda t: t[2])

    ops = []
    pending_refs = []
    prev_end = 0

    def flush_text(upto):
        nonlocal pending_refs
        span = targets[prev_end:upto].strip()
        if span:
            ops.append({
                "op": "generate",
                "text": span,
                "evidence_refs": pending_refs,
            })
            pending_refs = []

    for kind, uid, start, end in matches:
        flush_text(start)
        if kind == "copy":
            ops.append({"op": "copy", "source_idx": uid})
            pending_refs = []
        else:
            pending_refs.append(uid)
        prev_end = end
    flush_text(len(targets))
    return ops


def reconstruct_target(ops, source_sentences):
    """Rebuilds the full target article text from parsed ops.

    Equivalent to rendering_utils.normalize()'s clean_edited output.
    """
    pieces = []
    for op in ops:
        if op["op"] == "copy":
            pieces.append(source_sentences[op["source_idx"]])
        else:
            pieces.append(op["text"])
    return re.sub(r"\s{2,}", " ", " ".join(pieces)).strip()


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def split_generated_sentences(text):
    """Approximate sentence split for generated spans (used for stats).

    The pipeline emitted one sentence per generated segment, but adjacent
    generated sentences whose evidence refs are absent merge into one span,
    so this regex split is a best-effort re-segmentation.
    """
    return [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def diff_opcodes(source_sentences, ops):
    """Difflib opcodes between source sentences and target sentence sequence.

    Reproduces the pipeline's diff (difflib.SequenceMatcher over stripped
    sentence texts); generated spans are sentence-split first.
    """
    target_texts = []
    for op in ops:
        if op["op"] == "copy":
            target_texts.append(source_sentences[op["source_idx"]].strip())
        else:
            target_texts.extend(
                s.strip() for s in split_generated_sentences(op["text"]))
    source_texts = [s.strip() for s in source_sentences]
    matcher = difflib.SequenceMatcher(a=source_texts, b=target_texts)
    return matcher.get_opcodes(), source_texts, target_texts


def parse_instance(features, split=None, index=None):
    """Converts raw TFRecord features into the normalized instance schema.

    Fields:
      split, index         provenance (which file, position in file)
      input_raw            original `inputs` string
      target_raw           original `targets` string
      generatable_surfaces original annotation string (JSON-encoded by the
                           pipeline; kept verbatim)
      source_sentences     list of source article sentences
      evidence             list of {marker, title, content, raw}
      target_ops           list of copy/generate ops (see parse_target)
      source_text          clean source article text
      target_text          clean reconstructed target article text
    """
    inputs = features["inputs"]
    targets = features["targets"]
    source_sentences, evidence = parse_input(inputs)
    ops = parse_target(targets, len(source_sentences), len(evidence))
    instance = {
        "split": split,
        "index": index,
        "input_raw": inputs,
        "target_raw": targets,
        "generatable_surfaces": features.get("generatable_surfaces"),
        "source_sentences": source_sentences,
        "evidence": evidence,
        "target_ops": ops,
        "source_text": re.sub(
            r"\s{2,}", " ", " ".join(source_sentences)).strip(),
        "target_text": reconstruct_target(ops, source_sentences),
    }
    return instance


def load_jsonl(path):
    """Yields instances from a .jsonl file produced by build_subset.py."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    import sys

    for path in sys.argv[1:]:
        n = 0
        keys = set()
        for feats in read_tfrecords(path):
            keys.update(feats)
            n += 1
        print(f"{path}: {n} records, features: {sorted(keys)}")
