#!/usr/bin/env python3
"""
colab_pipeline.py -- Neuro-Symbolic Pipeline (All-in-One for Google Colab)

EXACT 2026 -- XAI Challenge @ IJCNN
Qwen2.5-7B + Z3 | Local, No Cloud API

Copy-paste vao Google Colab va chay tung cell (## la ranh gioi cell).
Hoac chay truc tiep: python colab_pipeline.py

Pipeline 5 giai doan:
  Stage 0: Cai dat Dependencies & Load Qwen2.5-7B
  Stage 1: Data Grounding + Dual-Layer Ontology
  Stage 2: Local Ontology Generation + AST FOL (Qwen)
  Stage 3: Deterministic Z3 Compilation & Verification
  Stage 4: Feedback Loop (Z3 -> Qwen) + Answer Extraction
  Stage 5: Evaluation & Export

Dual-Layer Ontology:
  - Global (bat bien, do ban dinh nghia)
  - Local (Qwen tu sinh moi sample)

AST JSON: Cay de quy 4 loai node:
  quantifier / connective / predicate / variable-constant

Feedback Loop:
  Z3 loi -> gui error ve Qwen -> retry toi khi compile thanh cong hoac het MAX_RETRIES
"""

# ══════════════════════════════════════════════════════════════════
# STAGE 0 -- Cai dat Dependencies & Load Qwen2.5-7B
# ══════════════════════════════════════════════════════════════════

import subprocess, sys

pkgs = [
    "z3-solver",
    "transformers>=4.40.0",
    "bitsandbytes>=0.43.0",
    "accelerate>=0.27.0",
    "sentencepiece",
    "protobuf",
]
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages"]
    + pkgs,
    check=True,
)
print("All packages installed OK")

import json, os, re, time, traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from z3 import (
    Int,
    IntSort,
    IntVal,
    BoolSort,
    Function,
    ForAll,
    Exists,
    And,
    Or,
    Not,
    Implies,
    Solver,
    sat,
    unsat,
)

print(f"PyTorch  : {torch.__version__}")
print(f"CUDA OK  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    print(f"GPU      : {props.name}  ({total_gb:.1f} GB)")
print("Imports OK")

# ==================================================================
# CAU HINH -- Chinh sua o day
# ==================================================================
QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # model chinh
QUANTIZATION = "8bit"  # '8bit' (~8-10 GB VRAM) | '4bit' (~5-6 GB VRAM)
DATASET_PATH = "Logic_Based_Educational_Queries-2.json"  # upload len Colab truoc
N_SAMPLES = 50  # so samples danh gia
MAX_RETRIES = 3  # so lan Qwen duoc phep sua lai khi Z3 loi
OUTPUT_PATH = "pipeline_results_qwen.json"
MAX_NEW_TOKENS = 4096  # token sinh ra toi da moi lan goi Qwen (formalization)
ANS_MAX_TOKENS = 512  # token cho answer extraction
# ==================================================================

print(f"Config OK | Model: {QWEN_MODEL_ID} | Quant: {QUANTIZATION}")
print(f"N_SAMPLES={N_SAMPLES}  MAX_RETRIES={MAX_RETRIES}")

# -- Load Qwen2.5-7B voi quantization
print(f"Loading {QWEN_MODEL_ID} ({QUANTIZATION})...")
print("  (lan dau can tai ~15 GB tu HuggingFace, sau do cache lai)")

if QUANTIZATION == "8bit":
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
elif QUANTIZATION == "4bit":
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
else:
    bnb_cfg = None  # full precision (chi dung neu GPU >= 28 GB)

tokenizer = AutoTokenizer.from_pretrained(
    QWEN_MODEL_ID,
    trust_remote_code=True,
    padding_side="left",
)

model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16 if bnb_cfg is None else None,
)
model.eval()

if torch.cuda.is_available():
    used_gb = torch.cuda.memory_allocated() / 1024**3
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Model loaded | VRAM: {used_gb:.1f} / {total_gb:.1f} GB")
else:
    print("Model loaded (CPU mode -- very slow, test only)")


# ══════════════════════════════════════════════════════════════════
# STAGE 1 -- Dual-Layer Ontology & Data Grounding
# ══════════════════════════════════════════════════════════════════

# -- TANG 1: GLOBAL ONTOLOGY (Static, bat bien)
GLOBAL_ONTOLOGY = {
    "quantifiers": ["forall", "exists"],
    "logical_operators": ["and", "or", "implies", "iff", "not"],
    "ast_node_types": [
        "quantifier",
        "connective",
        "predicate",
        "variable",
        "constant",
    ],
}

GLOBAL_ONTOLOGY_TEXT = """
## GLOBAL ONTOLOGY -- BAT BUOC TUAN THU (KHONG duoc sua doi)

### Luong tu (Quantifiers):
  forall  -> forall  (voi moi)
  exists  -> exists  (ton tai)

### Toan tu logic (Logical Operators):
  and     -> AND
  or      -> OR
  implies -> IMPLIES (keo theo)
  iff     -> IFF (tuong duong)
  not     -> NOT (phu dinh)

### So do 4 loai AST Node (phai dung DUNG nhu duoi):
  quantifier : { "type":"quantifier",  "operator":"forall|exists",
                 "bound_variables":["x",...], "body":{...} }
  connective : { "type":"connective",  "operator":"and|or|implies|iff|not",
                 "operands":[{...},{...},...] }
  predicate  : { "type":"predicate",   "name":"PredicateName",
                 "arguments":["x","y",...] }
  variable   : { "type":"variable",    "name":"x" }
  constant   : { "type":"constant",    "name":"SomeName" }

### QUY TAC CUNG (vi pham -> Z3 loi):
  1. Chi dung 4 node type tren, KHONG sang tao them
  2. 'not' chi co DUNG 1 operand
  3. 'implies' co DUNG 2 operands (ve trai, ve phai)
  4. bound_variables phai la list (du chi 1 bien)
  5. Bien dung: x, y, z (lowercase); hang so: PascalCase
"""

print("Global Ontology loaded:")
for k, v in GLOBAL_ONTOLOGY.items():
    print(f"  {k}: {v}")

# -- Load Dataset
with open(DATASET_PATH, encoding="utf-8") as f:
    full_dataset = json.load(f)

samples = full_dataset[:N_SAMPLES]

print(f"Dataset: {len(full_dataset)} samples -> dung {len(samples)}")
print(f"Fields: {list(samples[0].keys())}")

q_counts = [len(s["questions"]) for s in samples]
p_counts = [len(s["premises-NL"]) for s in samples]
print(f"Avg premises/sample : {sum(p_counts)/len(p_counts):.1f}")
print(f"Avg questions/sample: {sum(q_counts)/len(q_counts):.1f}")


# ══════════════════════════════════════════════════════════════════
# STAGE 2 -- Qwen2.5-7B: Sinh Local Ontology + Cay AST FOL
# ══════════════════════════════════════════════════════════════════


def call_qwen(system: str, user: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Goi Qwen voi chat template, tra ve raw text."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.05,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def safe_json(text: str) -> dict:
    """Trich xuat JSON tu response -- robust multi-strategy parser.

    BUG FIX #2: Thay rfind('}') bang brace-balancing chinh xac de khong
    bi nham dau '}' trong phan text giai thich Qwen them sau JSON.
    """
    text = text.strip()

    # 1) Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) ```json ... ``` hoac ``` ... ``` code fence
    for pattern in [
        r"```json\s*([\s\S]+?)\s*```",
        r"```\s*([\s\S]+?)\s*```",
    ]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except Exception:
                pass

    # 3) Brace-balancing: tim { ket hop dung voi }
    start = text.find("{")
    if start != -1:
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        break

    # 4) Fallback: rfind
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    raise ValueError(f"Khong the parse JSON (400 ky tu dau):\n{text[:400]}")


print("call_qwen() va safe_json() san sang")


# -- Prompt Templates

FORMALIZATION_SYSTEM = (
    "Ban la chuyen gia hinh thuc hoa logic bac mot (First-Order Logic).\n"
    "Lam viec theo quy trinh 2 buoc CUC KY NGHIEM NGAT duoi day.\n"
    "\n"
    + GLOBAL_ONTOLOGY_TEXT
    + "\n"
    "================================================================\n"
    "BUOC 1 -- LOCAL DECLARATION (Xay dung Tu dien Cuc bo)\n"
    "================================================================\n"
    "Doc toan bo premises. Nhan dien moi khai niem, thuc the, thuoc tinh, quan he, hanh dong.\n"
    "Quy tac:\n"
    "  - Moi Predicate <-> mot cum tu nguon (One-to-One Grounding)\n"
    "  - Ten Predicate: PascalCase, tieng Anh, ro nghia\n"
    "  - KHONG hallucinate: chi dinh nghia Predicate cho khai niem THUC SU co trong van ban\n"
    "  - Arity nhat quan -- cung Predicate luon cung so arguments\n"
    "\n"
    'Output Buoc 1 (JSON array, key = "step1_local_ontology"):\n'
    "[\n"
    '  {"source_text": "cum tu goc", "predicate": "PredicateName", "arity": 1, "description": "mo ta ngan"}\n'
    "]\n"
    "\n"
    "================================================================\n"
    "BUOC 2 -- SELF-BINDING CONSTRAINT & AST JSON GENERATION\n"
    "================================================================\n"
    "CHI DUOC PHEP dung:\n"
    "  (1) Toan tu tu Global Ontology: forall, exists, and, or, implies, iff, not\n"
    "  (2) Predicate DUNG ten + dung arity nhu da khai bao o Buoc 1\n"
    "TUYET DOI KHONG sang tao them Predicate hay toan tu moi.\n"
    "\n"
    "Dich MOI premise sang cay AST JSON de quy. Bien dung: x, y, z (chu thuong).\n"
    "\n"
    'Output Buoc 2 (JSON array, key = "step2_premises_ast"):\n'
    "[\n"
    '  {"premise_id": 0, "source_nl": "cau text goc", "ast": {...cay AST JSON...}}\n'
    "]\n"
    "\n"
    "================================================================\n"
    "OUTPUT FORMAT -- JSON THUAN TUY (khong co text giai thich, khong co markdown)\n"
    "================================================================\n"
    "{\n"
    '  "step1_local_ontology": [...],\n'
    '  "step2_premises_ast": [...]\n'
    "}\n"
)

CORRECTION_SYSTEM = (
    "Ban la chuyen gia sua loi FOL. He thong Z3 da phat hien loi trong cay AST ban sinh ra.\n"
    "Nhiem vu: sua lai TOAN BO (ca Buoc 1 va Buoc 2) de khong con loi compile.\n"
    "\n"
    + GLOBAL_ONTOLOGY_TEXT
    + "\n"
    "Loi hay gap can sua:\n"
    "  - Arity khong nhat quan (cung Predicate dung so arguments khac nhau)\n"
    "  - Variable chua khai bao trong bound_variables nhung lai dung trong body\n"
    "  - Dung Predicate khong co trong Local Ontology (hallucination)\n"
    '  - "not" co nhieu hon 1 operand\n'
    '  - "implies" khong du 2 operands\n'
    "\n"
    "Output JSON thuan tuy -- format GIONG HET lan dau (step1_local_ontology + step2_premises_ast).\n"
)

ANSWER_SYSTEM = (
    "Ban la chuyen gia suy luan logic. Dua vao cac tien de FOL da duoc xac minh boi Z3, hay tra loi cau hoi.\n"
    "\n"
    "Quy tac:\n"
    "  - Cau hoi trac nghiem (A/B/C/D): tra ve dung 1 chu cai HOA\n"
    '  - Cau hoi Yes/No: tra ve "Yes", "No", hoac "Unknown"\n'
    "  - Suy luan chat che tu tien de, khong suy doan ngoai pham vi\n"
    "\n"
    "Output JSON THUAN TUY:\n"
    '{"answer": "A|B|C|D|Yes|No|Unknown", "reasoning": "giai thich 1-2 cau ngan gon"}\n'
)

print("Prompt templates san sang")
print(f"  FORMALIZATION_SYSTEM: {len(FORMALIZATION_SYSTEM)} chars")


def run_formalization(premises_nl: list) -> dict:
    """Goi Qwen 2 buoc: NER + Local Ontology -> AST FOL de quy."""
    numbered = "\n".join(
        f"Premise {i+1}: {p}" for i, p in enumerate(premises_nl)
    )

    user_msg = (
        "Hay hinh thuc hoa cac tien de sau theo dung quy trinh 2 buoc.\n\n"
        + numbered
        + "\n\nNho:\n"
        "  Buoc 1: khai bao Local Ontology -- tat ca khai niem quan trong trong van ban\n"
        "  Buoc 2: dung DUNG ten + arity Predicate tu Buoc 1, sinh cay AST JSON de quy cho tung premise\n"
        "  Chi tra ve JSON thuan tuy -- khong co text, khong co markdown."
    )

    raw = call_qwen(FORMALIZATION_SYSTEM, user_msg)
    return safe_json(raw)


def hallucination_check(local_ontology: list, premises_ast: list) -> list:
    """Kiem tra One-to-One Grounding: moi Predicate trong AST phai co trong Local Ontology."""
    declared = {item["predicate"] for item in local_ontology}
    warnings = []

    def collect_predicates(node: dict, found: set):
        if not isinstance(node, dict):
            return
        if node.get("type") == "predicate":
            found.add(node.get("name", ""))
        for v in node.values():
            if isinstance(v, dict):
                collect_predicates(v, found)
            elif isinstance(v, list):
                for sub in v:
                    collect_predicates(sub, found)

    for item in premises_ast:
        used = set()
        collect_predicates(item.get("ast", {}), used)
        hallucinated = used - declared - {""}
        if hallucinated:
            warnings.append(
                f"Premise {item.get('premise_id','?')}: "
                f"predicates not in Local Ontology -> {hallucinated}"
            )
    return warnings


print("run_formalization() va hallucination_check() san sang")


# ══════════════════════════════════════════════════════════════════
# STAGE 3 -- Bo dich AST -> Z3 (Tat dinh, khong dung AI)
# ══════════════════════════════════════════════════════════════════

_func_cache: dict = {}


def get_z3_func(name: str, arity: int):
    """Tao hoac lay tu cache Z3 Function voi dung arity."""
    key = f"{name}_{arity}"
    if key not in _func_cache:
        sorts = [IntSort()] * arity + [BoolSort()]
        _func_cache[key] = Function(name, *sorts)
    return _func_cache[key]


def _resolve_bound_var_name(bv) -> str:
    """BUG FIX #4: bound_variables co the la string hoac dict {type, name}."""
    if isinstance(bv, dict):
        return bv.get("name", str(bv))
    return str(bv)


def _resolve_predicate_arg(a, var_map: dict):
    """BUG FIX #3: predicate arguments co the la string HOAC dict node."""
    if isinstance(a, str):
        if a in var_map:
            return var_map[a]
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
        raise ValueError(
            f"Argument khong hop le (type={atype!r}) trong predicate"
        )
    return IntVal(abs(hash(str(a))) % 100000)


def compile_ast(node: dict, var_map: dict):
    """Bien dich 1 AST node -> Z3 expression (tat dinh)."""
    if not isinstance(node, dict):
        raise ValueError(f"Expected dict node, got {type(node)}: {node!r}")

    ntype = node.get("type", "")

    # -- quantifier
    if ntype == "quantifier":
        op = node.get("operator", "").lower()
        bvs = node.get("bound_variables", [])
        if not bvs:
            raise ValueError("quantifier thieu bound_variables")
        bv_names = [_resolve_bound_var_name(bv) for bv in bvs]
        z3_bvs = [Int(v) for v in bv_names]
        child_map = {
            **var_map,
            **{v: z3_bvs[i] for i, v in enumerate(bv_names)},
        }
        body = compile_ast(node["body"], child_map)
        if op == "forall":
            return ForAll(z3_bvs, body)
        elif op in ("exists", "exist"):
            return Exists(z3_bvs, body)
        else:
            raise ValueError(f"Quantifier khong hop le: {op!r}")

    # -- connective
    elif ntype == "connective":
        op = node.get("operator", "").lower()
        ops = [compile_ast(o, var_map) for o in node.get("operands", [])]
        if op == "and":
            return And(*ops)
        elif op == "or":
            return Or(*ops)
        elif op == "implies":
            if len(ops) != 2:
                raise ValueError(
                    f"implies can dung 2 operands, nhan {len(ops)}"
                )
            return Implies(ops[0], ops[1])
        elif op == "iff":
            if len(ops) != 2:
                raise ValueError(
                    f"iff can dung 2 operands, nhan {len(ops)}"
                )
            return And(Implies(ops[0], ops[1]), Implies(ops[1], ops[0]))
        elif op == "not":
            if len(ops) != 1:
                raise ValueError(
                    f"not can dung 1 operand, nhan {len(ops)}"
                )
            return Not(ops[0])
        else:
            raise ValueError(f"Connective khong hop le: {op!r}")

    # -- predicate
    elif ntype == "predicate":
        name = node.get("name", "")
        args = node.get("arguments", [])
        if not name:
            raise ValueError('predicate thieu truong "name"')
        func = get_z3_func(name, len(args))
        z3_args = [_resolve_predicate_arg(a, var_map) for a in args]
        return func(*z3_args)

    # -- variable / constant
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


def verify_with_z3(premises_ast: list) -> dict:
    """Bien dich toan bo premises AST -> Z3, kiem tra consistency."""
    _func_cache.clear()

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


print("compile_ast() va verify_with_z3() san sang")


# ══════════════════════════════════════════════════════════════════
# STAGE 4 -- Vong Feedback Z3 -> Qwen + Trich xuat Cau tra loi
# ══════════════════════════════════════════════════════════════════


def run_correction(
    premises_nl: list,
    prev_formalization: dict,
    z3_info: dict,
    hall_warnings: list,
) -> dict:
    """Gui thong bao loi Z3 ve Qwen de sua lai.

    BUG FIX #1b: them None-guard cho prev_formalization va z3_info.
    """
    numbered = "\n".join(
        f"Premise {i+1}: {p}" for i, p in enumerate(premises_nl)
    )

    safe_z3 = z3_info or {}
    safe_prev = prev_formalization or {}

    z3_errors = (
        "\n".join(safe_z3.get("errors", []))
        or "(khong co loi compile cu the)"
    )
    hall_errs = "\n".join(hall_warnings) if hall_warnings else "(khong co)"
    prev_local = json.dumps(
        safe_prev.get("step1_local_ontology", []),
        ensure_ascii=False,
        indent=2,
    )

    correction_user = (
        "He thong Z3 da phat hien loi khi compile cay AST cua ban.\n\n"
        "===================================================\n"
        "THONG TIN LOI TU Z3\n"
        "===================================================\n"
        f'Z3 status: {safe_z3.get("status", "N/A")}\n'
        f'So premise compile duoc: {safe_z3.get("compiled_count", 0)} '
        f'/ {safe_z3.get("total_count", 0)}\n\n'
        f"Loi compile chi tiet:\n{z3_errors}\n\n"
        f"Loi Hallucination (Predicate khong khai bao):\n{hall_errs}\n\n"
        "===================================================\n"
        "LOCAL ONTOLOGY LAN TRUOC (de tham khao)\n"
        "===================================================\n"
        f"{prev_local}\n\n"
        "===================================================\n"
        "PREMISES GOC\n"
        "===================================================\n"
        f"{numbered}\n\n"
        "Hay sua lai TOAN BO (Buoc 1 + Buoc 2) de khong con loi.\n"
        "Chi tra ve JSON thuan tuy."
    )

    raw = call_qwen(CORRECTION_SYSTEM, correction_user, max_new_tokens=MAX_NEW_TOKENS)
    return safe_json(raw)


def extract_answers(
    premises_nl: list,
    fol_context: list,
    questions: list,
) -> list:
    """Dung Qwen tra loi cau hoi dua tren FOL da xac minh."""
    p_text = "\n".join(f"P{i+1}: {p}" for i, p in enumerate(premises_nl))
    fol_text = "\n".join(
        f"FOL P{i+1}: {f}" for i, f in enumerate(fol_context)
    )

    answers = []
    for i, q in enumerate(questions):
        user_msg = (
            "## Tien de (Natural Language):\n"
            f"{p_text}\n\n"
            "## Tien de (FOL da xac minh qua Z3):\n"
            f"{fol_text}\n\n"
            f"## Cau hoi {i+1}:\n"
            f"{q}\n\n"
            "Tra loi JSON thuan tuy."
        )

        raw = call_qwen(ANSWER_SYSTEM, user_msg, max_new_tokens=ANS_MAX_TOKENS)
        try:
            ans = safe_json(raw)
            answers.append(
                {
                    "question_id": i,
                    "answer": ans.get("answer", "Unknown"),
                    "reasoning": ans.get("reasoning", ""),
                }
            )
        except Exception:
            answers.append(
                {"question_id": i, "answer": "Unknown", "reasoning": raw[:200]}
            )
    return answers


print("run_correction() va extract_answers() san sang")


# ══════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════


@dataclass
class PipelineResult:
    sample_id: int
    status: str = "pending"
    z3_status: str = "pending"
    z3_compiled: int = 0
    z3_total: int = 0
    z3_attempts: int = 0
    z3_errors: list = field(default_factory=list)
    local_ontology: list = field(default_factory=list)
    premises_ast: list = field(default_factory=list)
    hallucination_warn: list = field(default_factory=list)
    predicted_answers: list = field(default_factory=list)
    ground_truth: list = field(default_factory=list)
    correct_count: int = 0
    total_questions: int = 0
    time_sec: float = 0.0
    error_log: list = field(default_factory=list)


def run_pipeline(idx: int, sample: dict) -> PipelineResult:
    """Chay toan bo pipeline cho 1 sample."""
    premises_nl = sample["premises-NL"]
    questions = sample["questions"]
    gt_answers = sample["answers"]

    result = PipelineResult(
        sample_id=idx,
        ground_truth=gt_answers,
        total_questions=len(questions),
    )
    t0 = time.time()
    print(
        f"  [Sample {idx:02d}] {len(premises_nl)} premises, {len(questions)} Q",
        end="",
        flush=True,
    )

    formalization = None
    z3_info = None

    # -- Vong lap Qwen -> Z3 -> Feedback
    for attempt in range(1, MAX_RETRIES + 1):
        result.z3_attempts = attempt
        try:
            # BUG FIX #1
            if attempt == 1 or formalization is None or z3_info is None:
                if attempt > 1:
                    print(f" [retry {attempt-1}]", end="", flush=True)
                formalization = run_formalization(premises_nl)
            else:
                print(f" [retry {attempt-1}]", end="", flush=True)
                formalization = run_correction(
                    premises_nl,
                    formalization,
                    z3_info,
                    result.hallucination_warn,
                )

            local_onto = formalization.get("step1_local_ontology", [])
            premises_ast = formalization.get("step2_premises_ast", [])

            if not premises_ast:
                raise ValueError(
                    "step2_premises_ast rong -- Qwen chua sinh AST"
                )

            hw = hallucination_check(local_onto, premises_ast)
            z3_info = verify_with_z3(premises_ast)

            result.local_ontology = local_onto
            result.premises_ast = premises_ast
            result.hallucination_warn = hw
            result.z3_status = z3_info["status"]
            result.z3_errors = z3_info.get("errors", [])
            result.z3_compiled = z3_info.get("compiled_count", 0)
            result.z3_total = z3_info.get("total_count", 0)

            if z3_info["status"] != "compile_error":
                break

        except Exception as e:
            result.error_log.append(
                f"Attempt {attempt}: {traceback.format_exc()[-500:]}"
            )
            # BUG FIX #5
            if attempt == MAX_RETRIES:
                print(" [all retries failed]", end="", flush=True)
                break

    # Fallback
    if not result.premises_ast:
        result.premises_ast = [
            {"premise_id": i, "source_nl": p, "ast": {}}
            for i, p in enumerate(premises_nl)
        ]
        if result.z3_status == "pending":
            result.z3_status = "no_ast"

    # -- Stage 4b
    fol_ctx = [item.get("source_nl", "") for item in result.premises_ast]
    ans_results = extract_answers(premises_nl, fol_ctx, questions)
    result.predicted_answers = ans_results

    correct = sum(
        1
        for i, ar in enumerate(ans_results)
        if i < len(gt_answers)
        and str(ar["answer"]).strip().upper()
        == str(gt_answers[i]).strip().upper()
    )
    result.correct_count = correct

    if result.z3_status in ("sat", "unsat", "unknown"):
        result.status = "success"
    elif result.z3_compiled > 0:
        result.status = "partial"
    else:
        result.status = "failed"

    result.time_sec = round(time.time() - t0, 2)

    badge = {
        "sat": "sat OK",
        "unsat": "unsat(!)",
        "unknown": "unknown?",
        "compile_error": "ERR_COMPILE",
        "solver_error": "ERR_SOLVER",
        "no_ast": "NO_AST",
    }.get(result.z3_status, result.z3_status)

    print(
        f" | {badge} | {correct}/{len(questions)} correct | {result.time_sec}s"
    )
    return result


print("PipelineResult + run_pipeline() san sang")


# ══════════════════════════════════════════════════════════════════
# RUN -- Danh gia toan bo samples
# ══════════════════════════════════════════════════════════════════

all_results = []

print("=" * 65)
print("  Neuro-Symbolic Pipeline -- Qwen2.5-7B + Z3")
print(f"  Model   : {QWEN_MODEL_ID}  ({QUANTIZATION})")
print(f"  Samples : {N_SAMPLES}   Max retries: {MAX_RETRIES}")
print("=" * 65)

for idx, sample in enumerate(samples):
    try:
        r = run_pipeline(idx, sample)
        all_results.append(r)
    except Exception as fatal:
        print(f"  [Sample {idx:02d}] FATAL: {fatal}")
        r = PipelineResult(
            sample_id=idx,
            status="failed",
            ground_truth=sample["answers"],
            total_questions=len(sample["questions"]),
            error_log=[str(fatal)],
        )
        all_results.append(r)

    if (idx + 1) % 10 == 0:
        done = idx + 1
        correct = sum(r.correct_count for r in all_results)
        total_q = sum(r.total_questions for r in all_results)
        acc = correct / total_q if total_q else 0
        sat_cnt = sum(1 for r in all_results if r.z3_status == "sat")
        print(
            f"\n  -- Checkpoint {done}/{N_SAMPLES} | "
            f"Acc: {acc:.1%} | Z3-sat: {sat_cnt}/{done} --\n"
        )

    time.sleep(0.1)

print(f'\n{"=" * 65}')
print(f"  Pipeline xong -- {len(all_results)} samples")
print(f'{"=" * 65}')


# ══════════════════════════════════════════════════════════════════
# STAGE 5 -- Evaluation & Tong hop ket qua
# ══════════════════════════════════════════════════════════════════


def evaluate(results: list) -> dict:
    n = len(results)
    if n == 0:
        return {}
    total_q = sum(r.total_questions for r in results)
    total_ok = sum(r.correct_count for r in results)
    status_ct = {"success": 0, "partial": 0, "failed": 0}
    z3_ct = {
        "sat": 0,
        "unsat": 0,
        "unknown": 0,
        "compile_error": 0,
        "solver_error": 0,
        "other": 0,
    }
    for r in results:
        status_ct[r.status] = status_ct.get(r.status, 0) + 1
        key = r.z3_status if r.z3_status in z3_ct else "other"
        z3_ct[key] += 1
    hall_total = sum(len(r.hallucination_warn) for r in results)
    avg_retries = sum(r.z3_attempts for r in results) / n
    avg_time = sum(r.time_sec for r in results) / n
    avg_comp = sum(r.z3_compiled for r in results) / n
    avg_tot_p = sum(r.z3_total for r in results) / n
    return {
        "n_samples": n,
        "total_questions": total_q,
        "total_correct": total_ok,
        "accuracy": round(total_ok / total_q, 4) if total_q else 0,
        "status_breakdown": status_ct,
        "z3_breakdown": z3_ct,
        "hallucination_warnings": hall_total,
        "avg_z3_retries": round(avg_retries, 2),
        "avg_time_sec": round(avg_time, 2),
        "avg_compiled_pct": (
            round(avg_comp / avg_tot_p * 100, 1) if avg_tot_p else 0
        ),
    }


metrics = evaluate(all_results)

W = 58
print("=" * W)
print("  NEURO-SYMBOLIC PIPELINE -- EVALUATION SUMMARY")
print(f"  Model: Qwen2.5-7B ({QUANTIZATION})")
print("=" * W)
print(f'  Samples evaluated  : {metrics["n_samples"]}')
print(f'  Total questions    : {metrics["total_questions"]}')
print(f'  Correct answers    : {metrics["total_correct"]}')
print(f'  Accuracy           : {metrics["accuracy"]:.1%}')
print("-" * W)
print("  Pipeline Status:")
for k, v in metrics["status_breakdown"].items():
    print(f"    {k:14}: {v:3d}  {'#' * v}")
print("-" * W)
print("  Z3 Verification:")
for k, v in metrics["z3_breakdown"].items():
    if v > 0:
        print(f"    {k:16}: {v:3d}  {'#' * v}")
print("-" * W)
print(f'  Hallucination warns: {metrics["hallucination_warnings"]}')
print(f'  Avg Z3 retries     : {metrics["avg_z3_retries"]}')
print(f'  Avg compile rate   : {metrics["avg_compiled_pct"]}%')
print(f'  Avg time / sample  : {metrics["avg_time_sec"]}s')
print("=" * W)

# -- Per-sample breakdown
header = (
    f"{'ID':>3} | {'Status':>8} | {'Z3':>13} | "
    f"{'Corr':>6} | {'Retry':>5} | {'Time':>6} | Hall | Predicted"
)
print(header)
print("-" * len(header))

for r in all_results:
    pred_ans = [a["answer"] for a in r.predicted_answers]
    paired = "  ".join(
        f'{"v" if str(p).upper()==str(g).upper() else "x"}{p}(gt:{g})'
        for p, g in zip(pred_ans, r.ground_truth)
    )
    hall = f"W{len(r.hallucination_warn)}" if r.hallucination_warn else "ok"
    print(
        f"{r.sample_id:>3} | {r.status:>8} | {r.z3_status:>13} | "
        f"{r.correct_count}/{r.total_questions:>4} | {r.z3_attempts:>5} | "
        f"{r.time_sec:>5.1f}s | {hall:>4} | {paired}"
    )


# -- Luu ket qua ra file JSON
def result_to_dict(r) -> dict:
    return {
        "sample_id": r.sample_id,
        "status": r.status,
        "z3_status": r.z3_status,
        "z3_compiled": r.z3_compiled,
        "z3_total": r.z3_total,
        "z3_attempts": r.z3_attempts,
        "z3_errors": r.z3_errors[:3],
        "hallucination_warns": r.hallucination_warn,
        "local_ontology": r.local_ontology,
        "correct_count": r.correct_count,
        "total_questions": r.total_questions,
        "predicted_answers": [a["answer"] for a in r.predicted_answers],
        "ground_truth": r.ground_truth,
        "per_question": [
            {
                "q_id": a["question_id"],
                "predicted": a["answer"],
                "gt": (
                    r.ground_truth[a["question_id"]]
                    if a["question_id"] < len(r.ground_truth)
                    else "?"
                ),
                "correct": (
                    str(a["answer"]).upper()
                    == str(r.ground_truth[a["question_id"]]).upper()
                    if a["question_id"] < len(r.ground_truth)
                    else False
                ),
                "reasoning": a.get("reasoning", ""),
            }
            for a in r.predicted_answers
        ],
        "time_sec": r.time_sec,
        "error_log": r.error_log[-1:],
    }


output_data = {
    "meta": {
        "model": QWEN_MODEL_ID,
        "quantization": QUANTIZATION,
        "n_samples": N_SAMPLES,
        "max_retries": MAX_RETRIES,
        "dataset": DATASET_PATH,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    },
    "metrics": metrics,
    "per_sample": [result_to_dict(r) for r in all_results],
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"Ket qua luu tai: {OUTPUT_PATH}")
print(f"  Dung luong: {Path(OUTPUT_PATH).stat().st_size / 1024:.1f} KB")
print(
    f'  Final Accuracy: {metrics["accuracy"]:.1%}  '
    f'({metrics["total_correct"]}/{metrics["total_questions"]})'
)
