# Contributing

Thanks for helping improve the **AI Performance Engineering Cheatsheet**.

This repository is a single document: `README.md`. There is no code, build, or test suite.
Contributions are edits to that document.

## What belongs here

The cheatsheet opens with Knuth's "premature optimization" quote, and that sets its stance: it is
a reference for *measuring before optimizing*, not a list of tricks. A good addition:

- defines the metric,
- gives the formula and its units,
- says what it tells you and when it misleads.

Concrete, dated hardware numbers (`H100: 3.35 TB/s`) are welcome. Vague claims are not.

## Structure

Twenty numbered sections, ordered hardware upward, with 17-20 cross-cutting and closing.

**Section numbering is load-bearing** - the table of contents links to generated heading anchors.
A new metric almost always belongs as a `###` subsection of an existing section. Adding a whole
section means renumbering every section after it and updating the TOC.

Section 19 is a goal-to-metrics lookup table; add a row if you introduce a materially new
*category* of metric. Section 20 is the reference list.

## Entry format

A `###` subsection takes one of three shapes. Match the neighbours rather than inventing a fourth.

**Bare formula** - a display block, then a line or two of prose on what it means:

```markdown
### 10.1 Time to First Token (TTFT)

$$
\text{TTFT} = T_{\text{receive request}} \to T_{\text{first token generated}}
$$

Includes prompt/prefill processing. Critical for perceived responsiveness.
```

**Labeled fields** - a bullet list, formula inline on the bullet:

```markdown
- **Formula:** $\displaystyle \text{IPC} = \frac{\text{Retired instructions}}{\text{Clock cycles}}$
- **Interpretation:** Average completed instructions per clock cycle.
```

**Comparison table** - for taxonomies and trade-offs rather than formulas.

## Math: two rules that will bite you

GitHub renders LaTeX, but it is easy to write math that silently degrades to raw text. It never
errors - it just looks broken. Both rules below come from real breakage in this document.

**1. Where `$$` works.** A display block must start its own block: blank line before it, flush to
the left margin. GitHub only promotes `$$` to math when it opens a block, so a `$$` glued to the
previous line is a Markdown lazy continuation and stays literal text.

**Display math is never recognised inside a list item**, at any indentation, blank line or not.
Inside a bullet, use inline `$...$` and add `\displaystyle` to keep fractions full size. Two
inline formulas on consecutive lines of one bullet collapse onto a single rendered line - put a
blank line between them.

```markdown
Definition:                     <-- BROKEN: $$ is glued to this line
$$
x = y
$$

- **Formula:**                  <-- BROKEN: display math inside a list item
  $$
  x = y
  $$

- **Formula:** $\displaystyle x = y$    <-- correct
```

**2. Double the backslash on punctuation macros.** Markdown unescapes backslash + ASCII
punctuation *before* the math is parsed, so `\%` arrives at the renderer as a bare `%`, which
starts a LaTeX comment and eats the rest of the line. `\,` `\;` `\!` arrive as literal `, ; !`.

Write `\\%`, `\\,`, `\;`, `\\!`, `\\$`. Control words (`\times`, `\frac`, `\text`) are
unaffected - never double those.

## House style

Plain ASCII. Use `'` not a curly quote, `->` and `=>` not arrows, `+/-` not the sign, `~` for
approximately, `x10^38` rather than a superscript, `us` for microseconds outside math. Normalize
anything pasted in from elsewhere.

Separate top-level sections with a `---` rule. Wrap words in `\text{}` inside formulas.

## Before you open a pull request

There are no tests; these checks stand in for them. Run from the repository root with `bash`:

```bash
# TOC anchors match the ## headings (no output = OK)
diff <(grep '^## ' README.md | sed 's/^## //' | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9 -]//g; s/ /-/g; s/^/#/') \
     <(grep -oP '^- \[.*?\]\(\K#[^)]*' README.md)

# section numbers are contiguous from 1
grep -o '^## [0-9]*' README.md | grep -o '[0-9]*' | awk 'NR!=$1{print "GAP at",NR,"->",$1}'

# non-ASCII drift
grep -nP '[^\x00-\x7F]' README.md

# math that will render as raw text: $$ opener glued to the previous line
awk '/^\$\$$/{ if(!b){ if(p!="") print "L"NR" glued to: "p; b=1 } else b=0; p=$0; next } {p=$0}' README.md

# display math inside a list item (never renders)
grep -n '^[[:space:]]\+\$\$' README.md

# punctuation macros needing a doubled backslash
grep -nP '(?<!\\)\\[!"#$%&'"'"'()*+,./:;<=>?@\[\]^_`{|}~]' README.md
```

To check math the way GitHub actually renders it, ask GitHub:

```bash
jq -Rs '{text:., mode:"gfm"}' README.md > payload.json
gh api -X POST /markdown --input payload.json > out.html
```

Recognised formulas come back wrapped in `<math-renderer class="js-display-math">` or
`js-inline-math`, and whatever sits inside that element is exactly what the math renderer
receives. Any `$$` left in the page text outside such an element will render as raw LaTeX.

## Pull requests

Keep them focused - one concern per branch. In the description, say what changed and how you
checked it. Please do not force-push after review has started.
