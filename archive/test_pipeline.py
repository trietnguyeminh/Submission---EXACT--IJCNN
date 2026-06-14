#!/usr/bin/env python3
"""
test_pipeline.py -- Unit tests cho cac thanh phan pipeline.

Chay KHONG can GPU / model -- chi test logic tat dinh:
  - JSON parser (safe_json)
  - Z3 compiler (compile_ast, verify_with_z3)
  - Hallucination check
  - Config validation

Usage:
  python test_pipeline.py
  python -m pytest test_pipeline.py -v
"""

import json
import sys
import traceback


# ══════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════

_passed = 0
_failed = 0


def test(name: str, func):
    """Run a test function, report pass/fail."""
    global _passed, _failed
    try:
        func()
        print(f"  [PASS] {name}")
        _passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        _failed += 1


# ══════════════════════════════════════════════════════════════════
# TEST: JSON Parser
# ══════════════════════════════════════════════════════════════════

def test_json_parser():
    from json_parser import safe_json

    # Test 1: Pure JSON
    def t1():
        r = safe_json('{"a": 1, "b": [2, 3]}')
        assert r == {"a": 1, "b": [2, 3]}, f"Got: {r}"

    test("safe_json: pure JSON", t1)

    # Test 2: Code fence
    def t2():
        text = 'Some text\n```json\n{"key": "val"}\n```\nMore text'
        r = safe_json(text)
        assert r == {"key": "val"}, f"Got: {r}"

    test("safe_json: code fence extraction", t2)

    # Test 3: JSON + trailing text
    def t3():
        text = '{"answer": "A"} Here is my explanation...'
        r = safe_json(text)
        assert r["answer"] == "A", f"Got: {r}"

    test("safe_json: JSON with trailing text", t3)

    # Test 4: Nested braces
    def t4():
        text = 'prefix {"a": {"b": {"c": 1}}} suffix'
        r = safe_json(text)
        assert r == {"a": {"b": {"c": 1}}}, f"Got: {r}"

    test("safe_json: nested braces", t4)

    # Test 5: Escaped quotes in strings
    def t5():
        text = '{"text": "He said \\"hello\\""}'
        r = safe_json(text)
        assert r["text"] == 'He said "hello"', f"Got: {r}"

    test("safe_json: escaped quotes", t5)

    # Test 6: Invalid JSON raises ValueError
    def t6():
        try:
            safe_json("no json here at all")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    test("safe_json: raises ValueError on no JSON", t6)


# ══════════════════════════════════════════════════════════════════
# TEST: Z3 Compiler
# ══════════════════════════════════════════════════════════════════

def test_z3_compiler():
    from z3_compiler import compile_ast, verify_with_z3, _func_cache

    # Test 1: Simple predicate
    def t1():
        _func_cache.clear()
        node = {"type": "predicate", "name": "Student", "arguments": ["x"]}
        result = compile_ast(node, {})
        assert result is not None

    test("compile_ast: simple predicate", t1)

    # Test 2: Forall quantifier
    def t2():
        _func_cache.clear()
        node = {
            "type": "quantifier",
            "operator": "forall",
            "bound_variables": ["x"],
            "body": {
                "type": "connective",
                "operator": "implies",
                "operands": [
                    {"type": "predicate", "name": "Student", "arguments": ["x"]},
                    {"type": "predicate", "name": "Person", "arguments": ["x"]},
                ],
            },
        }
        result = compile_ast(node, {})
        assert result is not None

    test("compile_ast: forall quantifier", t2)

    # Test 3: Nested connectives
    def t3():
        _func_cache.clear()
        node = {
            "type": "connective",
            "operator": "and",
            "operands": [
                {"type": "predicate", "name": "A", "arguments": ["x"]},
                {
                    "type": "connective",
                    "operator": "or",
                    "operands": [
                        {"type": "predicate", "name": "B", "arguments": ["x"]},
                        {"type": "predicate", "name": "C", "arguments": ["x"]},
                    ],
                },
            ],
        }
        from z3 import Int
        result = compile_ast(node, {"x": Int("x")})
        assert result is not None

    test("compile_ast: nested connectives", t3)

    # Test 4: NOT with single operand
    def t4():
        _func_cache.clear()
        node = {
            "type": "connective",
            "operator": "not",
            "operands": [
                {"type": "predicate", "name": "Rich", "arguments": ["x"]},
            ],
        }
        from z3 import Int
        result = compile_ast(node, {"x": Int("x")})
        assert result is not None

    test("compile_ast: NOT single operand", t4)

    # Test 5: NOT with multiple operands -> error
    def t5():
        _func_cache.clear()
        node = {
            "type": "connective",
            "operator": "not",
            "operands": [
                {"type": "predicate", "name": "A", "arguments": ["x"]},
                {"type": "predicate", "name": "B", "arguments": ["x"]},
            ],
        }
        try:
            from z3 import Int
            compile_ast(node, {"x": Int("x")})
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "1 operand" in str(e)

    test("compile_ast: NOT multiple operands -> error", t5)

    # Test 6: BUG FIX #4 -- bound_variables as dict
    def t6():
        _func_cache.clear()
        node = {
            "type": "quantifier",
            "operator": "forall",
            "bound_variables": [{"type": "variable", "name": "x"}],
            "body": {"type": "predicate", "name": "P", "arguments": ["x"]},
        }
        result = compile_ast(node, {})
        assert result is not None

    test("compile_ast: BUG FIX #4 bound_variables as dict", t6)

    # Test 7: BUG FIX #3 -- predicate args as dict
    def t7():
        _func_cache.clear()
        node = {
            "type": "predicate",
            "name": "Knows",
            "arguments": [
                {"type": "variable", "name": "x"},
                {"type": "constant", "name": "John"},
            ],
        }
        result = compile_ast(node, {})
        assert result is not None

    test("compile_ast: BUG FIX #3 predicate args as dict", t7)

    # Test 8: verify_with_z3 -- satisfiable
    def t8():
        _func_cache.clear()
        premises = [
            {
                "premise_id": 0,
                "source_nl": "All students are people",
                "ast": {
                    "type": "quantifier",
                    "operator": "forall",
                    "bound_variables": ["x"],
                    "body": {
                        "type": "connective",
                        "operator": "implies",
                        "operands": [
                            {"type": "predicate", "name": "Student", "arguments": ["x"]},
                            {"type": "predicate", "name": "Person", "arguments": ["x"]},
                        ],
                    },
                },
            },
            {
                "premise_id": 1,
                "source_nl": "John is a student",
                "ast": {
                    "type": "predicate",
                    "name": "Student",
                    "arguments": ["John"],
                },
            },
        ]
        result = verify_with_z3(premises)
        assert result["status"] == "sat", f"Expected 'sat', got: {result}"
        assert result["compiled_count"] == 2
        assert not result["errors"]

    test("verify_with_z3: satisfiable system", t8)

    # Test 9: verify_with_z3 -- compile error
    def t9():
        _func_cache.clear()
        premises = [
            {
                "premise_id": 0,
                "source_nl": "Bad node",
                "ast": {"type": "INVALID_TYPE"},
            },
        ]
        result = verify_with_z3(premises)
        assert result["status"] == "compile_error"
        assert len(result["errors"]) > 0

    test("verify_with_z3: compile error handling", t9)

    # Test 10: IFF connective
    def t10():
        _func_cache.clear()
        node = {
            "type": "connective",
            "operator": "iff",
            "operands": [
                {"type": "predicate", "name": "A", "arguments": ["x"]},
                {"type": "predicate", "name": "B", "arguments": ["x"]},
            ],
        }
        from z3 import Int
        result = compile_ast(node, {"x": Int("x")})
        assert result is not None

    test("compile_ast: IFF connective", t10)

    # Test 11: Exists quantifier
    def t11():
        _func_cache.clear()
        node = {
            "type": "quantifier",
            "operator": "exists",
            "bound_variables": ["x"],
            "body": {"type": "predicate", "name": "Happy", "arguments": ["x"]},
        }
        result = compile_ast(node, {})
        assert result is not None

    test("compile_ast: exists quantifier", t11)


# ══════════════════════════════════════════════════════════════════
# TEST: Hallucination Check
# ══════════════════════════════════════════════════════════════════

def test_hallucination():
    from ontology import hallucination_check

    # Test 1: No hallucination
    def t1():
        local = [
            {"predicate": "Student", "arity": 1},
            {"predicate": "Person", "arity": 1},
        ]
        ast_list = [
            {
                "premise_id": 0,
                "ast": {"type": "predicate", "name": "Student", "arguments": ["x"]},
            },
        ]
        warnings = hallucination_check(local, ast_list)
        assert len(warnings) == 0, f"Unexpected warnings: {warnings}"

    test("hallucination_check: no hallucination", t1)

    # Test 2: Hallucinated predicate
    def t2():
        local = [{"predicate": "Student", "arity": 1}]
        ast_list = [
            {
                "premise_id": 0,
                "ast": {"type": "predicate", "name": "Unknown", "arguments": ["x"]},
            },
        ]
        warnings = hallucination_check(local, ast_list)
        assert len(warnings) == 1, f"Expected 1 warning, got: {warnings}"
        assert "Unknown" in warnings[0]

    test("hallucination_check: detects hallucinated predicate", t2)


# ══════════════════════════════════════════════════════════════════
# TEST: Config Validation
# ══════════════════════════════════════════════════════════════════

def test_config():
    from config import PipelineConfig

    # Test 1: Default config
    def t1():
        cfg = PipelineConfig()
        assert cfg.quantization == "8bit"
        assert cfg.n_samples == 50
        assert cfg.max_retries == 3

    test("PipelineConfig: default values", t1)

    # Test 2: Invalid quantization
    def t2():
        try:
            PipelineConfig(quantization="16bit")
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass

    test("PipelineConfig: rejects invalid quantization", t2)

    # Test 3: Summary
    def t3():
        cfg = PipelineConfig(n_samples=10, quantization="4bit")
        s = cfg.summary()
        assert "4bit" in s
        assert "10" in s

    test("PipelineConfig: summary format", t3)


# ══════════════════════════════════════════════════════════════════
# TEST: Full Integration (without model)
# ══════════════════════════════════════════════════════════════════

def test_integration():
    """Test the full pipeline flow with mock Qwen output."""
    from z3_compiler import verify_with_z3
    from ontology import hallucination_check

    # Simulate what Qwen would produce for a simple sample
    def t1():
        mock_formalization = {
            "step1_local_ontology": [
                {"source_text": "student", "predicate": "Student", "arity": 1, "description": "is a student"},
                {"source_text": "completed coursework", "predicate": "CompletedCoursework", "arity": 1, "description": "completed required coursework"},
                {"source_text": "passing grade", "predicate": "PassingGrade", "arity": 1, "description": "received passing grade"},
                {"source_text": "can enroll", "predicate": "CanEnroll", "arity": 1, "description": "can enroll next semester"},
            ],
            "step2_premises_ast": [
                {
                    "premise_id": 0,
                    "source_nl": "All students who complete coursework receive a passing grade",
                    "ast": {
                        "type": "quantifier",
                        "operator": "forall",
                        "bound_variables": ["x"],
                        "body": {
                            "type": "connective",
                            "operator": "implies",
                            "operands": [
                                {
                                    "type": "connective",
                                    "operator": "and",
                                    "operands": [
                                        {"type": "predicate", "name": "Student", "arguments": ["x"]},
                                        {"type": "predicate", "name": "CompletedCoursework", "arguments": ["x"]},
                                    ],
                                },
                                {"type": "predicate", "name": "PassingGrade", "arguments": ["x"]},
                            ],
                        },
                    },
                },
                {
                    "premise_id": 1,
                    "source_nl": "Any student with passing grade can enroll",
                    "ast": {
                        "type": "quantifier",
                        "operator": "forall",
                        "bound_variables": ["x"],
                        "body": {
                            "type": "connective",
                            "operator": "implies",
                            "operands": [
                                {
                                    "type": "connective",
                                    "operator": "and",
                                    "operands": [
                                        {"type": "predicate", "name": "Student", "arguments": ["x"]},
                                        {"type": "predicate", "name": "PassingGrade", "arguments": ["x"]},
                                    ],
                                },
                                {"type": "predicate", "name": "CanEnroll", "arguments": ["x"]},
                            ],
                        },
                    },
                },
                {
                    "premise_id": 2,
                    "source_nl": "John is a student who completed coursework",
                    "ast": {
                        "type": "connective",
                        "operator": "and",
                        "operands": [
                            {"type": "predicate", "name": "Student", "arguments": ["John"]},
                            {"type": "predicate", "name": "CompletedCoursework", "arguments": ["John"]},
                        ],
                    },
                },
            ],
        }

        local_onto = mock_formalization["step1_local_ontology"]
        premises_ast = mock_formalization["step2_premises_ast"]

        # Hallucination check
        warnings = hallucination_check(local_onto, premises_ast)
        assert len(warnings) == 0, f"Unexpected warnings: {warnings}"

        # Z3 verify
        z3_result = verify_with_z3(premises_ast)
        assert z3_result["status"] == "sat", f"Expected sat, got: {z3_result}"
        assert z3_result["compiled_count"] == 3
        assert z3_result["total_count"] == 3
        assert len(z3_result["errors"]) == 0

    test("integration: mock formalization -> Z3 sat", t1)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  Neuro-Symbolic Pipeline -- Unit Tests")
    print("=" * 55)

    test_json_parser()
    test_z3_compiler()
    test_hallucination()
    test_config()
    test_integration()

    print("\n" + "=" * 55)
    print(f"  Results: {_passed} passed, {_failed} failed")
    print("=" * 55)

    sys.exit(1 if _failed > 0 else 0)
