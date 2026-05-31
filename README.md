# ⚡ AI-Quantum Portfolio Optimizer using QAOA

Dự án tối ưu hóa danh mục đầu tư tài chính sử dụng thuật toán lượng tử **QAOA (Quantum Approximate Optimization Algorithm)** kết hợp với AI, được phát triển cho cuộc thi **AI-Quantum Challenge 2026**.

## 📌 Tổng Quan Dự Án
Tối ưu hóa danh mục đầu tư (Portfolio Optimization) là bài toán lựa chọn tỷ lệ vốn phân bổ vào các tài sản sao cho tối đa hóa lợi nhuận kỳ vọng và tối thiểu hóa rủi ro (lý thuyết Markowitz Modern Portfolio Theory - MPT). 
Bài toán này được chuyển đổi thành dạng **QUBO (Quadratic Unconstrained Binary Optimization)** và giải quyết bằng thuật toán lượng tử biến phân **QAOA** chạy trên các trình giả lập hoặc phần cứng lượng tử của IBM Quantum thông qua Qiskit.

---

## 📂 Cấu Trúc Thư Mục Khuyên Dùng (Optimal Directory Structure)

Để đảm bảo dự án dễ bảo trì, mở rộng và tuân thủ các tiêu chuẩn của một dự án phần mềm lượng tử chuyên nghiệp, cấu trúc sau được thiết lập:

```text
d:/[AOF] AI Quantum 2026/
├── .venv/                      # Thư mục môi trường ảo Python
├── requirements.txt            # Danh sách các thư viện cần thiết và phiên bản tương thích
├── README.md                   # Hướng dẫn thiết lập và mô tả dự án
├── app.py                      # File chạy chính của ứng dụng Streamlit (Giao diện người dùng)
├── src/                        # Thư mục chứa mã nguồn xử lý logic chính
│   ├── __init__.py
│   ├── data_loader.py          # Tải dữ liệu chứng khoán từ Yahoo Finance (yfinance) và tiền xử lý
│   ├── portfolio.py            # Chuyển đổi bài toán danh mục đầu tư sang mô hình toán học (QUBO)
│   ├── quantum_solver.py       # Xây dựng mạch QAOA, cấu hình Qiskit Primitives (Sampler) và tối ưu hóa lượng tử
│   └── classical_solver.py     # Giải pháp cổ điển (SciPy/Min-Variance) để đối chiếu kết quả (Benchmark)
├── utils/                      # Thư mục chứa các hàm tiện ích trợ giúp
│   ├── __init__.py
│   └── visualization.py        # Vẽ biểu đồ (Efficient Frontier, phân bổ tỷ trọng tài sản, so sánh lợi nhuận)
└── docs/                       # Tài liệu thuyết minh toán học & lượng tử
    └── theory.md               # Giải thích chi tiết về Hamiltonian, biểu diễn Ising Model và QAOA
```

---

## 🛠️ Hướng Dẫn Khởi Tạo Môi Trường & Cài Đặt

### Bước 1: Mở Terminal tại thư mục dự án
Đảm bảo bạn đang ở thư mục làm việc của dự án:
```bash
cd "d:\[AOF] AI Quantum 2026"
```

### Bước 2: Tạo môi trường ảo (Virtual Environment)
Tạo một môi trường độc lập (ví dụ đặt tên là `.venv`) bằng Python:
```bash
python -m venv .venv
```

### Bước 3: Kích hoạt môi trường ảo
Tùy thuộc vào terminal bạn đang dùng trên Windows:

* **Trong PowerShell:**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
  *(Lưu ý: Nếu gặp lỗi quyền thực thi, hãy chạy lệnh `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` trước khi kích hoạt)*

* **Trong Command Prompt (cmd):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

* **Trong Git Bash / WSL:**
  ```bash
  source .venv/Scripts/activate
  ```

### Bước 4: Nâng cấp pip và cài đặt thư viện
Khi môi trường ảo đã được kích hoạt (bạn sẽ thấy `(.venv)` hiển thị đầu dòng lệnh), tiến hành cài đặt các gói thư viện từ file `requirements.txt`:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Hướng Dẫn Chạy Ứng Dụng
Sau khi cài đặt thành công và hoàn thiện các code logic, ứng dụng Streamlit có thể được khởi chạy bằng lệnh:
```bash
streamlit run app.py
```
Ứng dụng sẽ tự động mở giao diện web trên trình duyệt tại địa chỉ `http://localhost:8501`.
