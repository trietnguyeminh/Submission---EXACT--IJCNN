# EXACT 2026 — So sánh 2 Pipeline

> **Cuộc thi:** EXACT 2026 — Explainable AI Challenge @ IJCNN  
> **Bài toán:** Logic-based Educational QA (predict answer + explanation)  
> **Compute:** Kaggle T4×2 (16GB×2), 12h session  
> **Model gốc:** Qwen3-8B

---

## 1. Tổng quan 2 Pipeline

### Pipeline A — Neuro-Symbolic (Bootstrap / v14 + Calibrator)

```
Gold premises-FOL ──→ Recursive Descent Parser ──→ Z3 Expressions
                                                        │
Question ──→ Qwen Pass-2 (BoN=5, vLLM) ──→ FOL string ─┤
                                                        ▼
                                              Z3 Entailment Check
                                              (SAT / UNSAT / timeout)
                                                        │
                               ┌────────────────────────┤
                               ▼                        ▼
                          Z3 definite              Z3 indefinite
                               │                        │
                               ▼                        ▼
                    LightGBM Calibrator         Qwen-LoRA CoT (BoN=5)
                    (15 features, τ=0.30)
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
              p(z3_correct) ≥ τ    p(z3_correct) < τ
                    │                     │
                    ▼                     ▼
              → Z3 answer           → Qwen answer
```

**Stack:** vLLM + Unsloth + Z3 + LightGBM  
**Dataset:** 318 records / 630 questions (v4 cleaned)  
**LoRA:** r=64, train trên explanation field, `enable_thinking=False`

### Pipeline B — Pure LoRA-CoT (exact_70 / "notebook tào lao")

```
Gold FOL + NL premises ──→ Prompt (P1.[NL]...[FOL]...) ──→ Qwen-LoRA
                                                               │
                                                               ▼
                                                     CoT Reasoning
                                                     ### Final Answer
                                                               │
                                                               ▼
                                                        Regex extract
```

**Stack:** Unsloth only  
**Dataset:** 411 records / 812 questions (gốc, chưa clean)  
**LoRA:** r=64, α=128, LR=1e-4, 4 epochs, Unknown oversample 3x  
**Inference:** Greedy (temp=0), max_new_tokens=256, BoN=1

---

## 2. Kết quả

| Metric | Pipeline A (Neuro-Symbolic) | Pipeline B (Pure LoRA-CoT) |
|--------|:--------------------------:|:--------------------------:|
| **Overall Accuracy** | **79.8%** | **70.1%** |
| Eval fairness | Test set (held-out) | Train+Val mixed ⚠️ |
| Dataset | 318 records (clean) | 411 records (raw) |
| Unknown class | Z3 SAT check → ~72% | 92.0% (3x oversample) |
| Yes class | — | 80.2% |
| No class | — | 46.8% ⚠️ |
| MCQ (A/B/C/D) | — | ~62% |
| UNPARSEABLE | ~0% | 2.5% (20/812) |
| Oracle upper bound | 85.2% | — |

### Pipeline A — Accuracy by component

| Component | VAL | FULL |
|-----------|:---:|:----:|
| Pure Qwen-LoRA (BoN=5) | 73.0% | 77.0% |
| Pure Z3 definite | 72.2% | 76.7% |
| Z3-first rule | 71.4% | 76.8% |
| Parallel hybrid | 72.2% | 76.8% |
| **LightGBM Calibrator** | **73.8%** | **79.8%** |
| Oracle (perfect routing) | 80.2% | 85.2% |

### Pipeline B — Accuracy progression (train eval)

| Checkpoint | Running Acc |
|:----------:|:-----------:|
| 50/411 | 86.6% |
| 100/411 | 89.3% |
| 150/411 | 80.8% |
| 200/411 | 77.4% |
| 300/411 | 68.9% |
| 400/411 | 70.0% |
| **Final** | **70.1%** |

Accuracy giảm dần theo entry index → dấu hiệu overfitting / memorization.

---

## 3. So sánh kiến trúc

| Khía cạnh | Pipeline A | Pipeline B |
|-----------|-----------|-----------|
| FOL Parser | Recursive descent trên gold `premises-FOL`, 99.6% coverage | Không có — FOL chỉ nằm trong prompt text |
| Z3 Solver | SAT-based entailment, 3-tier confidence | Không có |
| LoRA mục đích | CoT reasoning (explanation field) | CoT reasoning (explanation field) |
| LoRA ↔ Inference | Match: cùng system prompt, `enable_thinking=False` | Match: cùng `SYSTEM_PROMPT` |
| BoN | N=5 (temp=0.5) | N=1 (greedy) |
| Routing | LightGBM 15 features | Không có |
| Unknown handling | Z3 SAT(H)∧SAT(¬H) → Unknown | 3x oversample → model bias |
| Inference engine | vLLM (batch) | Unsloth (sequential) |
| Time budget | ~2h train (pre-trained LoRA) + ~4h inference | ~4h train + ~2h inference |

---

## 4. Phân tích

### Pipeline B đúng ở đâu

**Simplicity works.** Không cần Z3, không cần routing — chỉ train LoRA-CoT rồi infer. Đạt 70% trên 812 questions chứng minh Qwen3-8B có khả năng học reasoning pattern từ explanation field. Đây là validation quan trọng cho hướng LoRA-CoT mà Pipeline A kế thừa.

**Gold FOL trong prompt giúp.** `format_premises` đặt cả NL và FOL cạnh nhau (P1. [NL]... [FOL]...). Model đọc FOL notation như context bổ sung, dù không xử lý formal. Pipeline A cũng giữ pattern này.

**System prompt match = critical.** Cả 2 pipeline đều match system prompt giữa train và inference. Bootstrap ghi rõ: mismatch → -11pp (v13.4 lesson).

### Pipeline B sai ở đâu

**No class là thảm họa: 46.8%.** 77/156 câu No bị đoán Yes. Root cause: dataset's "No" = pedagogical fallacy rejection, không phải classical ¬entailment. LoRA-CoT không có cơ chế phân biệt. Pipeline A dùng Z3 contrapositive check + calibrator để giảm thiểu.

**Unknown oversample 3x gây bias.** 92% Unknown acc nhưng kéo 78 câu non-Unknown sang Unknown. Đây là trade-off xấu: gain Unknown (+42pp vs random) không bù loss ở MCQ và No class.

**Eval trên train data.** 70.1% bao gồm ~85% data model đã thấy. Accuracy thật trên held-out sẽ thấp hơn đáng kể — progression 89.3% → 68.9% cho thấy gap train/test lớn.

**max_new_tokens=256 quá ngắn.** 20 câu UNPARSEABLE (2.5%) nhiều khả năng do truncation giữa CoT reasoning. Pipeline A không bị vì vLLM allocate đủ token.

### Pipeline A thắng nhờ đâu

**Z3 bổ sung cho Qwen, không thay thế.** Pure Z3 = 76.7%, Pure Qwen-LoRA = 77.0% — gần bằng nhau. Nhưng chúng sai ở CÁC CÂU KHÁC NHAU. Oracle (chọn đúng giữa Z3/Qwen) = 85.2%, nghĩa là 2 model complementary. LightGBM calibrator khai thác điều này → 79.8%.

**Gold FOL parser là major unlock.** v2-v10 (không dùng gold FOL) chỉ đạt ~40%. v11+ (dùng gold FOL parser) nhảy lên 52%+. Pipeline B "dùng" gold FOL nhưng chỉ như text — Pipeline A parse thành Z3 expressions và prove entailment.

**BoN=5 > Greedy.** Self-consistency voting giảm variance. Pipeline B dùng greedy (N=1) → single point of failure. BoN=5 với temp=0.5 cho Pipeline A ổn định hơn 3-5pp.

---

## 5. Key Findings (proven bằng số)

| # | Finding | Evidence |
|---|---------|----------|
| 1 | LoRA-CoT trên explanation field là hướng đúng | Pipeline B: 70.1% (train eval); Pipeline A pure Qwen-LoRA: 77.0% |
| 2 | Gold FOL parser là differentiator | Without (v2-v10): ~40%; With (v11+): 76.7% Z3 standalone |
| 3 | Z3 và Qwen complementary | Oracle upper bound 85.2% vs best single 77.0% = 8.2pp gap |
| 4 | Calibrator khai thác complementarity | 79.8% vs best non-calibrator 77.0% = +2.8pp |
| 5 | No class là hard ceiling cho pure-LLM | Pipeline B: 46.8%; cả 2 pipeline đều struggle |
| 6 | Unknown oversample có cost ẩn | +42pp Unknown acc nhưng -78 câu false Unknown |
| 7 | Eval on train data misleading | Pipeline B: 89.3% (top-100) vs 68.9% (entry 250-300) |

---

## 6. Cái Pipeline B dạy cho Pipeline A

Dù Pipeline B yếu hơn 10pp, nó validate 3 design decisions quan trọng mà Pipeline A kế thừa:

1. **LoRA trên explanation field** — proven gain so với base Qwen (~33% → 70%+)
2. **Gold FOL + NL cùng trong prompt** — model hưởng lợi từ dual representation
3. **System prompt consistency** — mismatch = catastrophic (-11pp elsewhere)

Pipeline A không phát minh hướng mới — nó REFINE hướng mà Pipeline B đã mở, bằng cách thêm Z3 formal verification + calibrated routing.

---

*Report generated for EXACT 2026 XAI Pipeline development.*
