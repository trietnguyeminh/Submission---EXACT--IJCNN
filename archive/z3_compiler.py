"""
z3_compiler.py -- Stage 3: Bo dich AST -> Z3 (Tat dinh, khong dung AI).

compile_ast(): Duyet cay JSON de quy -> anh xa sang Z3 API.
verify_with_z3(): Kiem tra satisfiability toan bo he logic.
"""

from z3 import (
    Int, IntSort, IntVal, BoolSort,
    Function, ForAll, Exists,
    And, Or, Not, Implies,
    Solver, sat, unsat,
)


# ══════════════════════════════════════════════════════════════════
# Predicate Function Cache (reset moi sample)
# ══════════════════════════════════════════════════════════════════

_func_cache: dict = {}


def get_z3_func(name: str, arity: int):
    """Tao hoac lay tu cache Z3 Function voi dung arity.

    Args:
        name: Ten predicate.
        arity: So arguments.

    Returns:
        Z3 Function declaration.
    """
    key = f"{name}_{arity}"
    if key not in _func_cache:
        sorts = [IntSort()] * arity + [BoolSort()]
        _func_cache[key] = Function(name, *sorts)
    return _func_cache[key]


def _resolve_bound_var_name(bv) -> str:
    """BUG FIX #4: bound_variables co the la string hoac dict {type, name}.

    Qwen doi khi sinh {"type":"variable","name":"x"} thay vi chuoi "x"
    trong danh sach bound_variables.
    """
    if isinstance(bv, dict):
        return bv.get("name", str(bv))
    return str(bv)


def _resolve_predicate_arg(a, var_map: dict):
    """BUG FIX #3: Predicate arguments co the la string HOAC dict node.

    Qwen doi khi sinh {"type":"variable","name":"x"} thay vi chuoi "x"
    trong arguments. Neu truyen dict vao `a in var_map` -> TypeError.
    """
    if isinstance(a, str):
        if a in var_map:
            return var_map[a]
        # Treat as constant
        return IntVal(abs(hash(a)) % 100000)

    if isinstance(a, dict):
        atype = a.get("type", "")
        name = a.get("name", "")
        if atype == "variable":
            if name in var_map:
                return var_map[name]
            v = Int(name)
            var_map[name] = v
            return v
        if atype == "constant":
            if name in var_map:
                return var_map[name]
            return IntVal(abs(hash(name)) % 100000)
        raise ValueError(f"Argument khong hop le (type={atype!r}) trong predicate")

    # Fallback: so nguyen
    return IntVal(abs(hash(str(a))) % 100000)


# ══════════════════════════════════════════════════════════════════
# COMPILER: AST JSON -> Z3 Expression
# ══════════════════════════════════════════════════════════════════

def compile_ast(node: dict, var_map: dict):
    """Bien dich 1 AST node -> Z3 expression (tat dinh, khong AI).

    Recursive descent qua 4 loai node:
      - quantifier (forall / exists)
      - connective (and / or / implies / iff / not)
      - predicate
      - variable / constant

    Args:
        node: Dict node tu cay AST JSON.
        var_map: { ten_bien_str -> Z3 Int variable } (scope hien tai).

    Returns:
        Z3 expression.

    Raises:
        ValueError: Khi node khong hop le.
    """
    if not isinstance(node, dict):
        raise ValueError(f"Expected dict node, got {type(node)}: {node!r}")

    ntype = node.get("type", "")

    # ── quantifier ────────────────────────────────────────────────
    if ntype == "quantifier":
        op = node.get("operator", "").lower()
        bvs = node.get("bound_variables", [])
        if not bvs:
            raise ValueError("quantifier thieu bound_variables")

        bv_names = [_resolve_bound_var_name(bv) for bv in bvs]
        z3_bvs = [Int(v) for v in bv_names]
        child_map = {**var_map, **{v: z3_bvs[i] for i, v in enumerate(bv_names)}}
        body = compile_ast(node["body"], child_map)

        if op == "forall":
            return ForAll(z3_bvs, body)
        elif op in ("exists", "exist"):
            return Exists(z3_bvs, body)
        else:
            raise ValueError(f"Quantifier khong hop le: {op!r}")

    # ── connective ────────────────────────────────────────────────
    elif ntype == "connective":
        op = node.get("operator", "").lower()
        ops = [compile_ast(o, var_map) for o in node.get("operands", [])]

        if op == "and":
            return And(*ops)
        elif op == "or":
            return Or(*ops)
        elif op == "implies":
            if len(ops) != 2:
                raise ValueError(f"implies can dung 2 operands, nhan {len(ops)}")
            return Implies(ops[0], ops[1])
        elif op == "iff":
            if len(ops) != 2:
                raise ValueError(f"iff can dung 2 operands, nhan {len(ops)}")
            return And(Implies(ops[0], ops[1]), Implies(ops[1], ops[0]))
        elif op == "not":
            if len(ops) != 1:
                raise ValueError(f"not can dung 1 operand, nhan {len(ops)}")
            return Not(ops[0])
        else:
            raise ValueError(f"Connective khong hop le: {op!r}")

    # ── predicate ─────────────────────────────────────────────────
    elif ntype == "predicate":
        name = node.get("name", "")
        args = node.get("arguments", [])
        if not name:
            raise ValueError('predicate thieu truong "name"')
        func = get_z3_func(name, len(args))
        z3_args = [_resolve_predicate_arg(a, var_map) for a in args]
        return func(*z3_args)

    # ── variable / constant ───────────────────────────────────────
    elif ntype in ("variable", "constant"):
        name = node.get("name", "")
        if name in var_map:
            return var_map[name]
        if ntype == "constant":
            return IntVal(abs(hash(name)) % 100000)
        v = Int(name)
        var_map[name] = v
        return v

    else:
        raise ValueError(f"AST node type khong hop le: {ntype!r}")


# ══════════════════════════════════════════════════════════════════
# VERIFIER: Z3 Satisfiability Check
# ══════════════════════════════════════════════════════════════════

def verify_with_z3(premises_ast: list) -> dict:
    """Bien dich toan bo premises AST -> Z3, kiem tra consistency.

    Args:
        premises_ast: List cua { premise_id, source_nl, ast }.

    Returns:
        Dict voi keys: status, errors, compiled_count, total_count.
        status co the la: 'sat', 'unsat', 'unknown', 'compile_error', 'solver_error'.
    """
    _func_cache.clear()  # reset de tranh arity conflict giua cac samples

    solver = Solver()
    errors = []
    compiled = 0

    for item in premises_ast:
        pid = item.get("premise_id", "?")
        try:
            ast = item.get("ast", {})
            if not ast:
                errors.append(f"Premise {pid}: AST rong")
                continue
            expr = compile_ast(ast, {})
            solver.add(expr)
            compiled += 1
        except Exception as e:
            errors.append(f"Premise {pid}: {str(e)[:250]}")

    if errors:
        return {
            "status": "compile_error",
            "errors": errors,
            "compiled_count": compiled,
            "total_count": len(premises_ast),
        }

    try:
        result = solver.check()
        return {
            "status": str(result),
            "errors": [],
            "compiled_count": compiled,
            "total_count": len(premises_ast),
        }
    except Exception as e:
        return {
            "status": "solver_error",
            "errors": [str(e)],
            "compiled_count": compiled,
            "total_count": len(premises_ast),
        }
