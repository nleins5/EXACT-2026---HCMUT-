# fine_tune/

Thư mục này chứa code fine-tune cho các model trong pipeline EXACT-2026.

## Cấu trúc

```text
fine_tune/
├── README.md                 # File hướng dẫn (bạn đang đọc)
└── Fine-tune_LLM.ipynb       # Notebook dùng chung để fine-tune cả Coder và Instruct
```

## Mục đích

Notebook Google Colab này được thiết kế để fine-tune 2 model chuyên biệt:

| Model Đích   | Tên model HuggingFace (`MODEL`)  | Dataset (`TRAIN_FILE`) | Mục đích                       |
| ------------ | -------------------------------- | ---------------------- | ------------------------------ |
| **Coder**    | `Qwen/Qwen2.5-Coder-7B-Instruct` | `coder.jsonl`          | Chuyên sinh code Z3/SymPy      |
| **Instruct** | `Qwen/Qwen2.5-7B-Instruct`       | `instruct.jsonl`       | Chuyên sinh JSON ExactResponse |

## Cách dùng

1. **Chuẩn bị Dữ liệu:** Upload thư mục `data/finetune/` (gồm `coder.jsonl` và `instruct.jsonl`) lên Google Drive của bạn.
2. **Khởi chạy Notebook:** Mở file `Fine-tune_LLM.ipynb` trên Google Colab.
3. **Cấu hình Model & Dataset:** Tại Cell khai báo biến, điều chỉnh các biến sau tùy thuộc vào model bạn muốn fine-tune:
   ```python
   DRIVE_DATA_DIR = "/content/drive/MyDrive/.../finetune" # Trỏ đến folder chứa data
   MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"               # Hoặc "Qwen/Qwen2.5-7B-Instruct"
   TRAIN_FILE = f"{DRIVE_DATA_DIR}/coder.jsonl"           # Hoặc "instruct.jsonl"
   ```
4. **Chạy Fine-tune:** Run all các cell trong notebook. Notebook sẽ lo toàn bộ từ việc cài đặt Unsloth, load model, chuẩn bị dataset, train, và cuối cùng là đẩy model lên Hugging Face.
5. **Convert sang GGUF (Tùy chọn):** Sau khi có LoRA adapter hoặc model merge trên Hugging Face, bạn có thể export ra định dạng GGUF.
6. **Sử dụng:** Tải file `.gguf` về, thả vào thư mục `models/` của project EXACT-2026.

## Dataset

- **Coder**: `data/finetune/coder.jsonl` (~1,391 records)
- **Instruct**: `data/finetune/instruct.jsonl` (~2,518 records)

_(Xem `data/finetune/README.md` để biết chi tiết cấu trúc dataset)._

## Cấu hình sau khi Fine-tune

Sau khi đã có model GGUF mới, hãy cập nhật đường dẫn trong `config/setting.yaml` để hệ thống `LlamaServerSupervisor` có thể load và swap giữa các model của bạn:

```yaml
llm:
  coder:
    model_path: models/qwen2.5-coder-7b-instruct-tuned.Q4_K_M.gguf
  instruct:
    model_path: models/qwen2.5-7b-instruct-tuned.Q4_K_M.gguf
```
