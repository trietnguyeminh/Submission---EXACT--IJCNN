# Prompt for Claude: Generate More Logic-Based Educational Query Records

You are generating additional JSON data for a logic-based educational reasoning dataset.

## Goal

Generate **100 new records** in the exact same schema as the provided dataset. Each record must contain:

```json
{
  "idx": [[...], [...]],
  "premises-FOL": ["...", "..."],
  "premises-NL": ["...", "..."],
  "questions": ["...", "..."],
  "answers": ["...", "..."],
  "explanation": ["...", "..."]
}
```

Return a **JSON array only**. Do not include Markdown, comments, prose, or code fences.

## Required schema rules

1. Each record must have exactly these fields:
   - `idx`
   - `premises-FOL`
   - `premises-NL`
   - `questions`
   - `answers`
   - `explanation`

2. `premises-FOL` and `premises-NL` must have the same length.

3. Each record should have **2 questions**:
   - one multiple-choice question with options A/B/C/D
   - one Yes/No/Unknown question

4. `answers` must contain only:
   - `"A"`, `"B"`, `"C"`, `"D"`
   - `"Yes"`, `"No"`, `"Unknown"`

5. `idx[i]` must list the 1-based premise numbers actually used to answer `questions[i]`.

6. Every index in `idx` must be valid:
   - minimum 1
   - maximum `len(premises-FOL)`

7. Do not duplicate records.

8. Do not repeat identical FOL premises within the same record.

## Logic quality rules

You must reason using only valid inference.

Allowed:

- Forward chaining:
  - `A → B`, `A` ⟹ `B`
- Multi-step forward chaining:
  - `A → B`, `B → C`, `A` ⟹ `C`
- Valid contrapositive:
  - `A → B` ⟹ `¬B → ¬A`
- Existential propagation:
  - `∃x A(x)`, `∀x(A(x) → B(x))` ⟹ `∃x B(x)`
- Universal instantiation / chaining:
  - `∀x A(x)`, `∀x(A(x) → B(x))` ⟹ `∀x B(x)`
- Conjunction introduction:
  - `A`, `B` ⟹ `A ∧ B`
- Conjunction elimination:
  - `A ∧ B` ⟹ `A`; `A ∧ B` ⟹ `B`

Forbidden:

- Affirming the consequent:
  - `A → B`, `B` ⟹ `A` is invalid.
- Denying the antecedent:
  - `A → B`, `¬A` ⟹ `¬B` is invalid.
- Invalid reverse inference:
  - `A → B` does not mean `B → A`.
- Invalid positive inference from contrapositive:
  - `¬B → ¬A` does not mean `B → A`.
- Assuming existence from a universal premise unless your logic setting explicitly allows non-empty domains. Prefer to include an explicit existential premise when deriving existential conclusions.

## Validation method before final output

For every question, internally build an inference graph from the FOL premises:

- Add forward edges from each implication:
  - `A → B`
- Add valid contrapositive edges:
  - `¬B → ¬A`
- For chains, run BFS/DFS from known facts.
- For Unknown answers, verify that no valid chain reaches the queried conclusion.
- For No answers, ensure the negation is explicitly derived or universally stated.

Do not output the graph, but use it to check correctness.

## Content requirements

Use educational, workplace training, software learning, research, or academic-administration contexts.

Avoid sensitive or risky topics.

Use varied predicates and scenarios. Example predicate names:

- `AttendSeminar(x)`
- `SubmitProject(x)`
- `ReceiveFeedback(x)`
- `CompleteModule(x)`
- `PassQuiz(x)`
- `EarnBadge(x)`
- `JoinResearchGroup(x)`
- `AccessDataset(x)`
- `WriteReport(x)`

## Desired answer distribution

Across 100 records / 200 questions, aim for a balanced mix:

- MCQ answers: roughly balanced among A/B/C/D
- Yes/No/Unknown: include all three, with at least 25 Unknown answers

## Explanation requirements

Each explanation must:

1. Mention the relevant premise numbers.
2. Explain why the answer follows.
3. For Unknown answers, explicitly say which reverse/unsupported inference is not allowed.
4. Avoid claiming a conclusion that contradicts the answer.
5. Avoid referencing premise numbers that do not exist.

## Output format

Return only a JSON array:

```json
[
  {
    "idx": [[1, 2, 3], [1, 2]],
    "premises-FOL": [
      "∀x (AttendSeminar(x) → TakeNotes(x))",
      "∀x (TakeNotes(x) → UnderstandTopic(x))",
      "∀x (AttendSeminar(x))"
    ],
    "premises-NL": [
      "If a student attends the seminar, then the student takes notes.",
      "If a student takes notes, then the student understands the topic.",
      "Every student attends the seminar."
    ],
    "questions": [
      "Based on the premises, which statement can be inferred?\nA. Every student understands the topic.\nB. No student understands the topic.\nC. Only some students attend the seminar.\nD. The premises are contradictory.",
      "According to the premises, is the following statement true?\nStatement: If a student attends the seminar, then the student understands the topic."
    ],
    "answers": ["A", "Yes"],
    "explanation": [
      "Premise 3 states that every student attends the seminar. Premise 1 maps attending the seminar to taking notes, and premise 2 maps taking notes to understanding the topic. Therefore every student understands the topic, so option A is correct.",
      "Premise 1 gives AttendSeminar→TakeNotes and premise 2 gives TakeNotes→UnderstandTopic. By forward chaining, AttendSeminar implies UnderstandTopic, so the statement is true."
    ]
  }
]
```

Now generate 100 new valid records.
