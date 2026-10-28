# Evidence writer brief

You write the evidence passages of a synthetic dataset for a text-updating
model. The model reads an outdated article plus evidence passages and must
decide, per sentence, whether to keep it, edit it or drop it. Your passages
are the only place where the model can find out what changed, so they must
carry the facts without revealing the answer through their form.

## Input

Each line of a task file `in/chunk_NNNN.jsonl` is one article:

    {"id": "...", "sentences": [{"i": 0, "label": "KEEP|EDIT|DROP|CASCADE", "text": "...", "fixes": [...]}]}

- `text` is the correct sentence for KEEP, EDIT and CASCADE, and the sentence
  that is no longer valid for DROP.
- EDIT and CASCADE sentences have `fixes`, a list of `{"wrong": ..., "right": ...}`.
  The outdated article holds `wrong` where `text` holds `right`.
- CASCADE sentences also have `avoid`, a list of strings.

## Output

For task file `in/chunk_NNNN.jsonl` write `out/chunk_NNNN.jsonl` with the Write
tool, one line per article, same order, same ids:

    {"id": "...", "evidence": [{"title": "...", "text": "...", "about": [0, 2]}]}

`about` lists the sentence indices `i` that the passage bears on. Use `[]` for
a background passage that bears on none. Every line must be valid JSON on one
line. Escape double quotes inside strings.

## What a passage is

A passage reads as an excerpt of some OTHER document: a different
encyclopedia article, a news clipping, a press release, an obituary, a
government or company notice, a book or journal abstract, a match report.
Vary the kind from article to article. One to three sentences, 25 to 70 words.
`title` is the title of that other document (an article title or a headline),
never the title of the subject article itself and never a generic label such
as "Reference", "Fact Sheet" or "Correction".

## Rules per label

EDIT. At least one passage must state the correct fact. It must contain every
`right` string of that sentence, spelled exactly as given, and list the
sentence in `about`. No passage of the article may contain a `wrong` string.
Do not say that anything changed, was corrected or was previously different.
The passage states the fact as its own document would.

DROP. At least one passage must make the sentence untenable: it reports that
the claim was disproved, superseded, withdrawn, found to be mistaken, or that
the thing described no longer exists or never happened. Say it the way a news
report or a later article would, with a concrete reason. List the sentence in
`about`. Vary the wording between articles. Do not reuse a stock phrase.

KEEP. For about half of the KEEP sentences, write a passage that is consistent
with the sentence and names its main subject, and list the sentence in
`about`. Leave the other half without a passage. Never contradict a KEEP
sentence.

CASCADE. The sentence is wrong only as a consequence of an EDIT sentence of
the same article. Write no passage about it and never list it in `about`. No
passage may contain one of its `wrong` strings or one of its `avoid` strings.
Do not state its fact in other words either. A reader must be able to fix it
only by reasoning from the corrected EDIT sentence.

Background. In about one article in three, add one passage on the same topic
that bears on no sentence (`about: []`).

## Rules for all passages

- Use your own wording. Never copy seven or more consecutive words from any
  sentence of the task.
- Never quote a sentence and never refer to "the sentence", "the article",
  "the claim above", an edit, a label or this task.
- One passage may bear on several sentences when that reads naturally.
- Invented supporting detail (dates, places, names of reporters) is fine. It
  must not contradict any `text` of a KEEP or EDIT sentence.
- Sentences are sometimes fragments or lists. Write the best passage you can
  for them. Do not skip an article.
- Order the passages of an article arbitrarily, not in sentence order.

## Procedure

Process the task files you were given one at a time. Read the file, write its
output file, then go to the next. Do not write helper scripts that generate
passages from templates. Every passage is written by you. Do not run the
validator yourself. When all files are written, reply with one line listing
the output files.
