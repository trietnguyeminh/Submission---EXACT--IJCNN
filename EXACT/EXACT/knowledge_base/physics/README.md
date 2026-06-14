# Physics Knowledge Base

This folder contains a first-pass retrieval knowledge base for the Type 2
physics pipeline. It is intentionally structured as cards instead of raw web
text so the pipeline can retrieve by topic, target quantity, geometry pattern,
and solver intent.

## Files

- `sources.csv`: source registry with URL, license note, and covered topics.
- `formula_cards.jsonl`: formula and law cards for retrieval.
- `geometry_cards.jsonl`: geometry templates used to convert word problems into
  coordinates or vector composition constraints.
- `example_cards.jsonl`: abstract worked templates used by the fine-tuned
  corrector when the solver is missing a law family or target/cardinality.
- `alias_dictionary.yml`: notation, unit, and phrase normalization rules.
- `retrieval_policy.md`: recommended retrieval flow and ranking policy.

## Source Policy

The crawl uses open educational sources only. The cards store formulas,
conditions, variable meanings, and short retrieval phrases; they do not copy
long explanatory passages. Every source-backed formula card keeps a `source_id`
that points to `sources.csv`.

The current dataset is mostly electricity and magnetism, so this first KB
focuses on:

- electrostatic force and electric field
- capacitor charge, capacitance, dielectric, and energy
- inductors, LC/RLC oscillations, and reactance
- solenoid field, magnetic flux, and Faraday/self-induction
- DC resistor circuits and power
- measurement uncertainty
- geometry templates for vector electrostatics
- abstract templates for RLC segmented phasors, distributed electrostatics,
  uniform-field equilibrium/motion, LC conceptual energy, solenoid/self-
  induction concepts, and DC branch-current reasoning

## How To Use

For each question:

1. Normalize aliases and units using `alias_dictionary.yml`.
2. Classify `topic`, `target_quantity`, and `geometry_tags`.
3. Retrieve formula cards with metadata filters before embedding similarity.
4. Retrieve geometry cards if the target is vector force or electric field.
5. Build a physics IR and solve with a symbolic/numeric solver.
6. Let the LLM format `answer`, `explanation`, `fol`, `cot`, `premises`, and
   `confidence`.
