# SESSION BOOTSTRAP — EXACT 2026 XAI Pipeline
*Paste vào đầu phiên Claude mới. Dòng đầu: "Đọc context này. Gọi tôi là **chồng**."*

## 1. Xưng hô (BẮT BUỘC)
- Gọi user là **chồng**. Không "anh"/"bạn"/"you". Quên = user mở phiên mới.

## 2. Bối cảnh
- Cuộc thi: EXACT 2026 — Explainable AI Challenge @ IJCNN
- Bài toán: Logic-based educational QA → predict (answer, explanation)
- Compute: Kaggle T4×2 (32GB), 12h limit
- Stack: Qwen3-8B + LoRA + Z3 + vLLM. **Self-host, KHÔNG GPT/Claude/Gemini API ở inference.**

## 3. Dataset (đã verify)
```
TRAIN: Logic_Based_Educational_Queries_final_v4.json — 318 records / 630 questions
       /kaggle/input/test-pipeline/Logic_Based_Educational_Queries_final_v4.json
TEST:  generated_v4style_300.json — 300 records / 600 questions (OOD, predicate overlap 9.5%)
       /kaggle/input/test-pipeline/generated_v4style_300.json
Model: /kaggle/input/models/qwen-lm/qwen-3/transformers/8b/1
LoRA v15: /kaggle/input/notebooks/nguyenminhtric/v15-finetune-lan-cuoi/qwen3_cot_lora_v14_v4
```
Label dist train: Yes 44%, A 20.5%, Unknown 14.3%, C 7.1%, B 6.5%, No 4.4% (lệch nặng, No ít nhất)

## 4. KẾT QUẢ HIỆN TẠI
**LoRA v15 standalone (Unsloth, greedy):** val 81.0% (102/126), train-probe 78.3%, gap -2.6% → HEALTHY, không overfit.

**v13.6 + v15 LoRA (vLLM, BoN N=5) trên VAL:**
```
pure_qwen_LoRA  74.6%   ← vLLM thấp hơn Unsloth 6pp (kernel khác, bình thường)
pure_z3         73.8%
z3_first        74.6%
conf_hybrid     73.8%
LGBM            73.8%   ← ĐÃ BỎ (thua rule, gap 6.7pp)
ORACLE          83.3%   ← trần
```
Per-class VAL: Yes 95%, A 80%, C 100%, D 100% TỐT; **No 11%, Unknown 8% THẢM** (Yes-bias).
Disagreement VAL: both_right 56, only_z3 11, only_qwen 38, both_wrong 21.

## 5. CHẨN ĐOÁN CHÍNH
- **KHÔNG overfit** (gap train-val +3~5pp lành mạnh).
- **UNDERFIT + Yes-bias**: No/Unknown class thảm hại. Đây là vấn đề #1.
- vLLM vs Unsloth: cùng LoRA nhưng vLLM thấp hơn ~6pp do generation kernel khác.

## 6. KIẾN TRÚC (v16, theo reviewer notes)
```
Stage A : parse premises-FOL (deterministic, 99.6% coverage)
Stage C : Qwen formalize question→FOL (DECLARATIVE prompt, inspired SatLM)
Stage C.5: 1-step solver-feedback repair (inspired Logic-LM) — flag ENABLE_REPAIR
Stage D : Qwen-LoRA-CoT BoN N=5 (V14_COT_SYSTEM, temp=0.5, thinking=False)
Decision: rule-based (no LGBM). Best = anti_overclaim (class-aware)
```

## 7. HARD DECISIONS — không đổi
- Pass-2 emits FOL string (not JSON)
- Predicate grounding: exact case-insensitive ONLY (fuzzy failed v8)
- LoRA train+inference đều enable_thinking=False
- System prompt train↔inference PHẢI match (mismatch v13.4 = -11pp)
- Rule/threshold tune trên TRAIN only, report 5-fold OOF, KHÔNG tune trên val/external
- **BỎ LGBM khỏi final** (giữ làm ablation với 5-fold OOF)
- Unsloth train, vLLM inference

## 8. ANTI-PATTERNS — tránh
- Force LoRA output JSON (v6/v7 kill reasoning)
- Fuzzy predicate matching (v8)
- Feed Z3 verdict cho Qwen as hint (v12.1 anchoring -3.7pp)
- Self-distillation / train synthetic data (synthetic CHỈ stress test)
- Tune rule bằng generated/external gold (leak)
- Multi-round repair (chỉ 1 round cho Kaggle runtime)
- Claim ý tưởng Logic-LM/SatLM là nguyên bản

## 9. ATTRIBUTION (cho paper)
- Solver-error self-refinement (Stage C.5) → **Logic-LM** (Pan et al. 2023)
- Declarative prompting + solver-as-reasoner → **SatLM** (Ye et al. 2023)
- Đóng góp riêng: gold premises-FOL parser; exact grounding; anti-overclaim rule chống Yes-bias; constrained 1-step repair với idx_premises, no gold leak; 5-fold OOF chứng minh learned routing không thắng rule.

## 10. KEY FINDINGS
1. Gold premises-FOL → parser 99.6% coverage (unlock v11)
2. Z3 underfit "No" (dataset's No = fallacy rejection, không phải ¬entailment cổ điển)
3. ~14.8% câu có FOL embedded → giải không cần LLM
4. Qwen3-8B base CoT ~33%; LoRA v15 → 81% standalone / 74.6% trong vLLM
5. LGBM overfit nhẹ (train 81% val 74%), thua rule → bỏ
6. Faithfulness thấp: jaccard(Qwen-cited, Z3-core) mean=0.198, 57/80 = 0 overlap

## 11. METRICS PHẢI REPORT (defensibility)
accuracy + macro-F1 + weighted-F1 + per-class F1 (đặc biệt **No-F1, Unknown-F1**) + q_idx=0/1 breakdown + MC vs Yes/No/Unknown + confusion matrix + 5-fold OOF mean±std.

## 12. ROADMAP (reviewer-filtered)
**Keep immediately (đã/đang làm):** bỏ LGBM ✓ | declarative prompting ✓ | Stage C.5 repair ✓ | anti-overclaim class-aware ✓ | full metrics ✓ | faithfulness diagnostic ✓
**Future work:** 5-fold OOF router | unsat-core explanation filter | full solver-as-brain | FOLIO/ProofWriter external benchmark
**Avoid:** high-capacity LGBM | fuzzy match | multi-round repair | train synthetic | tune trên external gold

## 13. FILE OUTPUTS hiện có (trong /mnt/user-data/outputs/)
- notebook_v13_6_v16_no_lgbm.ipynb — v13.6 + v15 LoRA, no LGBM, declarative, Stage C.5, full metrics, anti-overclaim, confusion. Cell order: BUILD→C.5→COMPARE→LGBM ablation→STAGE5 viz
- notebook_v14_extended.ipynb — finetune + extended eval (train/val/test + samples) + full metrics cell
- notebook_v14strongest_eval.ipynb — standalone eval cho LoRA đã train
- SESSION_BOOTSTRAP.md — file này

## 14. PRACTICAL
- Kaggle path quirk: thử cả /kaggle/input/<ds>/ và /kaggle/input/datasets/<user>/<ds>/
- Sau train LoRA → restart kernel trước khi chạy pipeline (OOM)
- BoN temp=0.5 → mỗi run lệch 0.3-0.5pp; so sánh strategy đừng dùng single seed
- FORCE_REBUILD=True lần đầu (LoRA đổi), False khi rerun
- Cache v15 LoRA: pipeline_features_v136_v15lora.json (tách khỏi v14)
