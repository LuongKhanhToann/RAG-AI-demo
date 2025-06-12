### Hướng dẫn các bước xây dựng hệ thống RAG AI agent

Bước 1: Chuẩn bị file dữ liệu về cửa hàng (`ABC_Store.txt`), tiền xử lý bằng chatbot để thêm section (loại dữ liệu) và category (loại sản phẩm):

 python preprocess_data.py => preprocess_data.txt  
 
Bước 2: Chuyển data đã thêm section và category (`preprocess_data.txt`) thành vecto và lưu vào vecto db:

python save_to_db.py  

Bước 3: Hỏi dữ liệu liên quan, có thể thay câu hỏi vào biến question:

python main.py
