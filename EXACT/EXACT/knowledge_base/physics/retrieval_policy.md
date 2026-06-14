# Retrieval Policy

## Goal

Improve Type 2 physics accuracy by retrieving the correct law, geometry model,
and notation normalization before asking the solver/LLM to produce the final
JSON answer.

## Retrieval Order

1. **Alias normalization**
   - Normalize `q0`, `qo`, `q'`, `q prime` as target/test-charge candidates.
   - Normalize `U` and `V` as voltage.
   - Normalize units before numeric parsing.

2. **Lightweight query classifier**
   - Predict `topic`.
   - Predict `target_quantity`.
   - Predict `geometry_tags`.
   - Predict whether the problem is inverse/constraint solving.

3. **Metadata-filtered retrieval**
   - Formula cards: filter by topic and target first.
   - Geometry cards: filter by geometry phrase and target type.
   - Example cards: filter by abstract IR shape and use them when the solver is
     missing, low-confidence, or likely to miss answer cardinality.

4. **Hybrid score**
   - `score = 0.45 * metadata_match + 0.30 * BM25 + 0.25 * embedding`
   - Boost cards whose `target_quantities` match the target.
   - Boost geometry cards for force/electric-field vector tasks.
   - Penalize capacitor cards when the target is electric field from point
     charges, and vice versa.

5. **Solver-first execution**
   - Convert the question to a physics IR.
   - Use SymPy/numeric vector solver for formulas and geometry.
   - LLM is used to repair parsing, format explanation, and create final JSON,
     not to invent numeric answers.

## Recommended Top-K

- Formula cards: `top_k = 3`
- Geometry cards: `top_k = 2`
- Worked examples: `top_k = 3`

## Current Failure Buckets To Target

The first retrieval expansion should target:

- electric field vs force confusion
- q0/qo/q-prime target notation
- collinear charges
- midpoint cases
- equilateral triangle force/field vectors
- perpendicular bisector cases
- inverse zero-field or unknown-charge constraints
- segmented AC/RLC phasor circuits with `LComega^2=1`
- distributed electrostatics: ring, rod, disk, sheet, wire, semicircle
- uniform electric-field motion and charged equilibrium
- conceptual answer canonicalization for LC, solenoid, self-induction, and DC

## Do Not

- Do not pass raw crawled paragraphs directly into the model.
- Do not let formula retrieval alone choose a scalar formula for vector tasks.
- Do not cancel vectors by symmetry unless charges, distances, and signs are
  explicitly symmetric.
- Do not allow the LLM to output a long sentence in `answer`; `answer` should
  be the computed value plus unit only.
