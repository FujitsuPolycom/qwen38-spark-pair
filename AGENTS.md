# Writing standard for this repository

Applies to every piece of prose the repository carries: documentation, code
comments, docstrings, commit messages, pull request text, plans, reports,
identifier names, and error strings. It binds human contributors and automated
agents equally.

## The requirement

All repository prose must make sense to a technically capable reader who has the
repository and none of the conversation or development history that produced it.

- **Describe the system as it exists.** State purpose, behavior, invariants,
  interfaces, evidence, and limitations directly. Do not narrate the journey.
- **Do not use lifecycle labels as identities.** `pilot`, `next`, `current`,
  `new`, `old`, and `latest` are not technical names.
- **Do not use false definite references.** Phrases such as "the 1M-token
  capture", "the experiment", or "this approach" are invalid unless the exact
  object was introduced locally and unambiguously. Counts, dates, and versions
  are attributes, not identities.
- **Give the semantic role on first reference**, together with the durable
  identifier when a reader needs it: artifact name, path, schema, revision,
  manifest, or hash.
- **Explain concepts before identifiers.** Internal codenames, experiment
  labels, profile numbers, and implementation shorthand are not the vocabulary
  of a design. Introduce what a thing does, then name it, and only when the
  reader must use the name.
- **Canonical documentation is a present-state specification, not a changelog.**
  Replace stale claims rather than layering history on top of them. Chronology,
  rejected alternatives, and retrospectives belong only in documents that
  declare themselves historical.
- **Label status explicitly** as `implemented`, `qualified`, `research-only`, or
  `unsupported`. State evidence as conditions, measurement, result, and
  conclusion, not as a story.
- **Comments explain invariants, intent, and non-obvious constraints**, never
  change history. A TODO names the missing condition and the criterion for its
  removal.
- **Commits and pull requests state resulting behavior**, technical reason,
  compatibility impact, and validation. They do not recount attempts or pivots.

**Final test:** if understanding a sentence requires having been present for the
work, rewrite it.

## Worked distinctions

Each pair below is drawn from prose this repository carries.

| Fails the standard | Satisfies it |
|---|---|
| "We then discovered the servers hold the KV cache." | "A cache server holds the engine's KV allocation through CUDA IPC until the engine unregisters it or the reaper expires it." |
| "The new launcher forwards more gates." | "`start-stack.sh` forwards every gate the serve script reads, including `RECON_M`, `FP8PREFILL`, and `SPEC`." |
| "Fixed the L1 sizing issue." | "`--l1-size-gb` must be at least the largest replayed prefix: a lookup counts an L2 hit only after the chunk stages into L1, and the default trim policy truncates at the first staging failure." |
| "This approach was validated." | "Measured on two DGX Spark nodes at TP2, greedy decoding, `GPUMEM=0.70`: cold time to first token 61.69 s, replay after engine restart 4.09 s, external prefix cache hit rate 97.9%." |

A status label is a claim about evidence, so it carries an obligation. Write
`qualified` only where the repository records the conditions, the measurement,
and the result. Where a configuration runs but nothing measured it, `implemented`
is the honest label.
