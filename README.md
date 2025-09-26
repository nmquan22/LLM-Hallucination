# Vietnamese Hallucination Detection (Cross‑Encoder Reranker)

> Phát hiện “hallucination” cho câu trả lời LLM trên tiếng Việt bằng **cross‑encoder** (reranker) fine‑tune end‑to‑end: phân loại **`no`**, **`intrinsic`**, **`extrinsic`**.

## TL;DR
- **Input**: cặp văn bản *(hypothesis, evidence)* trong đó  
  - `hypothesis` = `response` (có thể nối thêm `"[Prompt]: {prompt}"`)  
  - `evidence`   = `context`
- **Model**: encoder‑only/cross‑encoder (ví dụ: `AITeamVN/Vietnamese_Reranker`, `Qwen/Qwen3-Reranker-0.6B`)  
  → thay **linear head** thành 3 lớp.
- **Loss**: `CrossEntropyLoss` (có **class weights** khi dữ liệu lệch)
- **Output**: softmax → `{no, intrinsic, extrinsic}`
- **Submit**: file **`submit.csv`** chỉ gồm **`id`, `predict_label`**, nén thành **`submit.zip`**.

---

## 1) Bài toán & Nhãn

Phân loại 3 nhãn cho mỗi *(context, prompt, response)*:
- **no** – câu trả lời **được hỗ trợ** bởi `context` (không ảo giác).
- **intrinsic** – câu trả lời **mâu thuẫn/sai nội tại** so với `context`.
- **extrinsic** – câu trả lời đưa thông tin **không có/không đủ bằng chứng** trong `context`.

> Gợi ý nhớ: *intrinsic = trái ngược với bằng chứng; extrinsic = thêm ngoài bằng chứng.*

---

## 2) Kiến trúc & Dòng dữ liệu

### Cross‑encoder là gì?
Tokenizer ghép **hai đoạn** vào **một chuỗi** để encoder xử lý **chung**, cho phép self‑attention **chéo** giữa token của `hypothesis` và `evidence`.

- BERT: `[CLS] hypothesis [SEP] evidence [SEP]` (+ token_type_ids 0/1)  
- RoBERTa/XLM‑R/Qwen‑reranker: `<s> hypothesis </s></s> evidence </s>` (không dùng `token_type_ids`)

### Luồng tính toán
```
(hypothesis, evidence)
   └─ tokenizer(...) → input_ids, attention_mask, (token_type_ids?)
       └─ Embeddings (token + position (+ segment))
           └─ Transformer encoder blocks (self‑attention chéo)
               └─ Pooling/CLS → Linear(H→3) → logits (B,3)
                   └─ softmax → probs → argmax → label
```

### Vì sao cần tokenizer?
Backbone **không ăn text raw**, mà cần `input_ids` + special tokens + mask/padding **đúng chuẩn** của checkpoint để học/khởi tạo embedding chính xác.

---

## 3) Định dạng dữ liệu

- **Train CSV**: bắt buộc có cột `context`, `response`, `label` (và **có thể** có `prompt`).  
- **Validation**: tách từ train theo `VAL_SPLIT` (stratified).  
- **Test CSV**: bắt buộc có cột `id`, `context`, `response` (và **có thể** có `prompt`).

> Nhãn train (`label`) là `{no, intrinsic, extrinsic}`. Dùng `LabelEncoder` để map ↔ chuỗi.

---

## 4) Mô hình & Cấu hình

### Tuỳ chọn model (khuyến nghị)
- ✅ **`AITeamVN/Vietnamese_Reranker`** – nhẹ, tiếng Việt tốt, batch lớn hơn.
- ✅ **`Qwen/Qwen3-Reranker-0.6B`** – mạnh, cần thêm `pad_token` và batch nhỏ hơn.
  ```python
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
  if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token or tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
      model.resize_token_embeddings(len(tokenizer))  # nếu add token mới
  model.config.pad_token_id = tokenizer.pad_token_id
  ```
- ❌ Các model *Instruct/Chat/Decoder‑only* (Qwen‑7B, Qwen2‑7B‑Instruct, …) **không drop‑in** cho `AutoModelForSequenceClassification`.
- ❌ Model *Embedding* (bi‑encoder) **không phải cross‑encoder**.

### Head 3 lớp
Hầu hết reranker checkpoint gốc trả về **1 score**. Ta thay **Linear** cuối thành **3 classes**:
```python
# Sau khi from_pretrained(..., ignore_mismatched_sizes=True, trust_remote_code=True)
_replace_final_linear_head(model, num_labels=3)
model.config.num_labels = 3
model.config.problem_type = "single_label_classification"
# Lưu ý: thay head xong mới model.to(device)
```

### Hyperparams gợi ý
- `MAX_LENGTH = 384` (nếu OOM: 256).  
- `BATCH_SIZE = 8` (AITeamVN) | `2~4` (Qwen3‑0.6B).  
- `LR = 5e-6 ~ 1e-5` (Qwen3 hợp 8e-6~1e-5).  
- `EPOCHS = 3` là đủ với encoder‑only đã pretrain.  
- `WEIGHTED_LOSS = True` nếu dữ liệu lệch lớp.  
- (Tuỳ phiên bản) `fp16=True`, `warmup_ratio=0.06`, `lr_scheduler_type="cosine"`.

---

## 5) Train/Eval/Submit (những điểm mấu chốt)

### Sanity check (1 batch)
- Chỉ feed `input_ids/attention_mask/(token_type_ids)` vào model, **không** truyền `labels`; tự tính `CrossEntropy` để chắc logits `(B,3)` và backward OK.

### Trainer & Loss
- Ghi đè `compute_loss`: `labels = inputs.pop("labels")` → gọi `model(**inputs)` → lấy `logits` → `CrossEntropyLoss` (có `weight=class_weights` nếu bật).
- **Không** để model tự tính loss (tránh rẽ sang MSE khi config hiểu nhầm regression).

### Logging
- `logging_steps` nhỏ (5–20) + callback `on_log` để in loss/epoch rõ ràng.  
- Tắt W&B nếu môi trường tự bật: `os.environ["WANDB_MODE"] = "disabled"`.

### Evaluate
- In **accuracy** + **macro‑F1** cho **validation**.  
- Có thể in **confusion matrix** để xem lỗi nhầm lớp.

### Submission
- Tạo **`submit.csv`** với **2 cột**: `id`, `predict_label` (∈ {`no`,`intrinsic`,`extrinsic`}).  
- Nén thành **`submit.zip`** và submit.

Ví dụ tạo file nộp:
```python
submit = pd.DataFrame({
    "id": test_df["id"].astype(str),
    "predict_label": le.inverse_transform(np.argmax(test_logits, axis=-1))
})
submit.to_csv("/kaggle/working/submit.csv", index=False)

import zipfile
with zipfile.ZipFile("/kaggle/working/submit.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write("/kaggle/working/submit.csv", arcname="submit.csv")
```

---

## 6) Troubleshooting (các lỗi hay gặp & cách sửa)

1) **Không thấy GPU/CPU chạy, không in log**  
   - Nguyên nhân: tạo nhiều `Trainer` (sanity/test) → **Accelerate state kẹt**.  
   - Cách sửa: trước khi train thật, chạy:
     ```python
     from accelerate.state import AcceleratorState
     AcceleratorState._reset_state()
     ```
     rồi tạo **Trainer mới** → `trainer.train()` **một lệnh duy nhất**.

2) **`RuntimeError: Cannot handle batch sizes > 1 if no padding token is defined.`** (Qwen3)  
   - Chưa set `pad_token_id`. Sửa như phần **Mô hình & Cấu hình** ở trên.

3) **`size mismatch`/`mismatched head` khi load checkpoint**  
   - Dùng `ignore_mismatched_sizes=True` và **thay head 3 lớp** sau khi load.

4) **Model tự tính MSE thay vì CE** (lỗi broadcast khi có `labels`)  
   - Trong `compute_loss`, **pop** `labels` khỏi `inputs` rồi tự tính `CrossEntropyLoss` trên `logits`.

5) **`cpu vs cuda` device mismatch**  
   - Nhớ: **thay head xong mới** `model.to(device)` (hoặc `.to(device)` cho head mới).

6) **Checkpoint ghi lỗi “PytorchStreamWriter failed writing file …”**  
   - Optimizer state quá lớn. Tắt lưu optimizer/scheduler:  
     ```python
     args.save_strategy = "no"          # nếu có
     args.save_only_model = True        # nếu có
     # hoặc:
     trainer._save_optimizer_and_scheduler = lambda *a, **k: None
     ```
   - Sau train, dùng `trainer.save_model(save_dir)` + `tokenizer.save_pretrained(save_dir)`.

7) **Tokenizer xoá mất `labels`**  
   - Khi `map(...)`, chỉ `remove_columns=["hypothesis","evidence"]` để **giữ `labels`**.

---

## 7) Cách đổi model (drop‑in)
Chỉ cần đổi `MODEL_NAME` sang model cross‑encoder tương thích và đảm bảo:
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, ignore_mismatched_sizes=True, trust_remote_code=True)

# Nếu thiếu pad token (Qwen3)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

_replace_final_linear_head(model, 3)
model.config.num_labels = 3
model.config.problem_type = "single_label_classification"
model.to(device)
```

> Điều chỉnh `BATCH_SIZE`, `MAX_LENGTH`, `LR` theo VRAM (P100 16GB: Qwen3‑0.6B → BS=2~4, MaxLen=256~384).

---

## 8) Yêu cầu môi trường
- Python 3.10+
- PyTorch 2.1+ (CUDA nếu có GPU)
- `transformers` (≥ 4.40), `datasets`, `accelerate`, `scikit‑learn`, `pandas`, `numpy`

```bash
pip install -U torch torchvision torchaudio
pip install -U "transformers[torch]" datasets accelerate scikit-learn pandas numpy
```

---

## 9) Kết quả tham khảo (tuỳ dữ liệu)
- `AITeamVN/Vietnamese_Reranker`: Val Acc ~ **73–75%** (đã báo ~**73.77%**).  
- `Qwen/Qwen3-Reranker-0.6B`: Val Acc ~ **71–75%** (có thể nhỉnh hơn sau khi tune LR/warmup).  
  > Lưu ý: so sánh công bằng cần **cùng split** và cùng preprocess.

---

## 10) Cấu trúc repo (gợi ý)
```
.
├── README.md
├── train.ipynb              # notebook chính
├── src/
│   ├── data_utils.py        # load/mapping/encode
│   ├── model_utils.py       # build tokenizer/model, replace head
│   ├── trainer_utils.py     # WeightedTrainer, callbacks
│   └── eval_submit.py       # eval & tạo submit
└── assets/                  # ảnh minh hoạ / sơ đồ
```

---

## 11) License & Credits
- Dữ liệu: theo nguồn bạn sử dụng.  
- Model: theo giấy phép của từng checkpoint (`AITeamVN/Vietnamese_Reranker`, `Qwen/Qwen3-Reranker-0.6B`, …).  
- Mã tham khảo dựa trên Hugging Face `transformers`/`datasets`/`accelerate`.
