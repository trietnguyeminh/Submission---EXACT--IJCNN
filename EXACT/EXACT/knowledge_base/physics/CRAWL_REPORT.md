# Physics KB Crawl Report

Created for the Type 2 physics retrieval pipeline.

## Crawl Scope

The local dataset is dominated by electricity and magnetism, so the crawl was
limited to sources and cards that support:

- Coulomb force and electric-field superposition
- capacitor charge, capacitance, dielectric, and energy
- inductor energy, self-induction, LC/RLC oscillation, and reactance
- solenoid magnetic field, magnetic flux, and Faraday induction
- simple DC resistor circuits and electric power
- measurement uncertainty
- vector geometry patterns for electrostatics

## Generated KB Assets

- `formula_cards.jsonl`: 24 formula cards
- `geometry_cards.jsonl`: 10 geometry cards
- `alias_dictionary.yml`: notation/unit/phrase normalization
- `sources.csv`: 10 source records
- `retrieval_policy.md`: retrieval and ranking policy

## Source Notes

The crawl uses LibreTexts/OpenStax educational pages and the local dataset for
template mining. The KB stores structured formulas, metadata, solver hints, and
short retrieval phrases. It does not store long copied passages.

Primary external source families:

- Physics LibreTexts electric charges and fields summary
- Physics LibreTexts capacitance summary
- Physics LibreTexts solenoid section
- Physics LibreTexts Faraday law section
- Physics LibreTexts LC oscillation section
- Physics LibreTexts resistor series/parallel section
- Physics LibreTexts units and measurement summary
- OpenStax RLC series circuits page

## Retrieval Design

Recommended lookup order:

1. Normalize notation and units with `alias_dictionary.yml`.
2. Classify the query into `topic`, `target_quantity`, and `geometry_tags`.
3. Retrieve formula cards with metadata filters.
4. Retrieve geometry cards for force/electric-field/vector problems.
5. Build physics IR.
6. Solve with SymPy/numeric solver.
7. Ask the LLM only to format the final JSON and explanation.

## Immediate Integration Targets

The first integration pass should focus on these cards:

- `coulomb_force_superposition`
- `electric_field_point_charges`
- `geometry_collinear_two_sources_target`
- `geometry_midpoint_ab`
- `geometry_perpendicular_bisector`
- `geometry_equilateral_triangle`
- `geometry_triangle_by_three_sides`
- `zero_electric_field_two_charges`

These map directly to the high-error buckets from the current Type 2 test run:
electric-field/force confusion, q0/q-prime target notation, collinear cases,
midpoint cases, equilateral geometry, perpendicular bisectors, and inverse
zero-field constraints.

