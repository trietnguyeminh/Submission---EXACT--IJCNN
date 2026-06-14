"""
ontology.py -- Stage 1: Dual-Layer Ontology & Prompt Templates.

Tang 1 -- Global Ontology: bat bien, dinh nghia cac toan tu FOL cot loi.
Tang 2 -- Local Ontology:  Qwen tu sinh tren tung sample (Stage 2).
"""


# ══════════════════════════════════════════════════════════════════
# TANG 1 -- GLOBAL ONTOLOGY (Static, bat bien)
# ══════════════════════════════════════════════════════════════════

GLOBAL_ONTOLOGY = {
    "quantifiers":       ["forall", "exists"],
    "logical_operators": ["and", "or", "implies", "iff", "not"],
    "ast_node_types":    ["quantifier", "connective", "predicate", "variable", "constant"],
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


# ══════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════

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
    "Output Buoc 1 (JSON array, key = \"step1_local_ontology\"):\n"
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


# ══════════════════════════════════════════════════════════════════
# PHYSICS-SPECIFIC PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════

PHYSICS_FORMALIZATION_SYSTEM = (
    "Ban la chuyen gia hinh thuc hoa vat ly + logic bac mot (First-Order Logic).\n"
    "Bai tap vat ly co the chua: cong thuc, don vi, gia tri so, dieu kien bien.\n"
    "Lam viec theo quy trinh 2 buoc CUC KY NGHIEM NGAT duoi day.\n"
    "\n"
    + GLOBAL_ONTOLOGY_TEXT
    + "\n"
    "================================================================\n"
    "BUOC 1 -- LOCAL DECLARATION (Xay dung Tu dien Cuc bo)\n"
    "================================================================\n"
    "Nhan dien cac dai luong vat ly, he thuc, dieu kien bien.\n"
    "Quy tac:\n"
    "  - Moi Predicate <-> mot khai niem hoac quan he vat ly\n"
    "  - VD: HasMass(x), Accelerates(x), Force(x,y), IsGreaterThan(x,y)\n"
    "  - Arity nhat quan -- cung Predicate luon cung so arguments\n"
    "\n"
    "Output Buoc 1 (JSON array, key = \"step1_local_ontology\"):\n"
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
    "\n"
    "Dich MOI premise sang cay AST JSON de quy.\n"
    "\n"
    'Output Buoc 2 (JSON array, key = "step2_premises_ast"):\n'
    "[\n"
    '  {"premise_id": 0, "source_nl": "cau text goc", "ast": {...cay AST JSON...}}\n'
    "]\n"
    "\n"
    "================================================================\n"
    "OUTPUT FORMAT -- JSON THUAN TUY\n"
    "================================================================\n"
    "{\n"
    '  "step1_local_ontology": [...],\n'
    '  "step2_premises_ast": [...]\n'
    "}\n"
)


def hallucination_check(local_ontology: list, premises_ast: list) -> list:
    """Kiem tra One-to-One Grounding: moi Predicate trong AST phai co trong Local Ontology.

    Returns:
        List of warning strings for hallucinated predicates.
    """
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
                f"Premise {item.get('premise_id', '?')}: "
                f"predicates not in Local Ontology -> {hallucinated}"
            )
    return warnings
