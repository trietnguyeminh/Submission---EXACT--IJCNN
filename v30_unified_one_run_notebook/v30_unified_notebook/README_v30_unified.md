# v30 unified one-run notebook

Run only this notebook instead of four separate v30 notebooks.

## Required input

`v27_standard_preds.json` must be available in one of:

- `/kaggle/working`
- `/kaggle/input/datasets/yixuanisthebest/v2333333`
- `/kaggle/input/datasets/nguyenminhtric/test-pipeline`
- anywhere under `/kaggle/input`

`v27_standard_summary.json` is optional.

## Outputs

The notebook writes:

- `v30_standard_preds.json`
- `v30_standard_summary.json`
- `v30_a_preds.json`
- `v30_a_summary.json`
- `v30_b_preds.json`
- `v30_b_summary.json`
- `v30_full_preds.json`
- `v30_full_summary.json`
- `v30_1_full_preds.json`
- `v30_1_full_summary.json`

## Expected final result

- `selection_status = SELECT_V30`
- `macro_f1 = 0.5934206145879246`
- `flipped_indices_from_v27 = [71, 109, 125]`

The notebook verifies actual saved `v30_standard_preds.json`, not just the summary, so it catches the old artifact bug.
