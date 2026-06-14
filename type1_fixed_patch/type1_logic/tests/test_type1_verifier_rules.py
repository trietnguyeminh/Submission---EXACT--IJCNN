from app.type1_logic.verifier_v35 import verify
from app.type1_logic.prompt import build_prompt


def test_positive_existential_from_universal():
    ans, prem, reason = verify(
        "Does at least one student receive a scholarship?",
        ["Every student receives a scholarship."],
        "Unknown",
    )
    assert ans == "Yes"
    assert prem == [0]
    assert reason.startswith("PE:")


def test_negative_existential_from_no_premise():
    ans, prem, reason = verify(
        "Does at least one student pass the exam?",
        ["No student passes the exam."],
        "Unknown",
    )
    assert ans == "No"
    assert prem == [0]
    assert reason.startswith("E1:")


def test_prompt_matches_artifact_style_ynu():
    prompt = build_prompt(["No student passes the exam."], "Does at least one student pass the exam?", ["Yes", "No", "Uncertain"])
    assert "Use only the given premises. Do not use outside knowledge." in prompt
    assert "Question:\nDoes at least one student pass the exam?" in prompt
    assert "Final Answer: <Yes, No, or Unknown>" in prompt
    assert "Task:" not in prompt
