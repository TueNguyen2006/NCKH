# Hướng dẫn thiết lập nhanh (Quick Setup)

Sao chép toàn bộ đoạn mã dưới đây và dán vào Terminal để tự động thực hiện: Clone project, tạo môi trường ảo, cài đặt thư viện và chạy chương trình.

```bash
# 1. Clone repo và truy cập thư mục
git clone https://github.com/TueNguyen2006/NCKH.git && cd NCKH

# 2. Khởi tạo và kích hoạt môi trường ảo (venv)
python -m venv venv
venv\Scripts\activate

# 3. Cập nhật pip và cài đặt dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Chạy chương trình
python inference1.py
