import openai
import pinecone
# import numpy as np
import os
import uuid
from dotenv import load_dotenv
load_dotenv()
# Nhập API key Pinecone từ biến môi trường (hoặc trực tiếp như trước)
openai.api_key = os.getenv("OPENAI_API_KEY")
api_key_pinecone = os.getenv("PINECONE_API_KEY")
environment = 'us-west1-gcp'

# Khởi tạo Pinecone thông qua lớp Pinecone mới
from pinecone import Pinecone, ServerlessSpec

# Tạo đối tượng Pinecone
pc = Pinecone(api_key=api_key_pinecone)

# Kiểm tra nếu chỉ mục đã tồn tại, nếu chưa tạo mới
index_name = "shop-info-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Chỉnh lại kích thước vector (ví dụ: GPT-3 embeddings)
        metric='euclidean',  # Hoặc 'cosine', tùy vào yêu cầu
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Lấy index
index = pc.Index(index_name)

# Hàm tạo embedding cho văn bản
def create_embedding(text):
    embedding_response = openai.Embedding.create(
        model="text-embedding-ada-002",  # Chọn mô hình embedding
        input=text
    )
    return embedding_response['data'][0]['embedding']

# Hàm đọc và tiền xử lý dữ liệu từ file
def read_and_preprocess_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Phân tách nội dung thành các section và category
    data = []
    sections = content.split("Section: ")
    
    for section in sections[1:]:
        lines = section.split("\n")
        
        section_name = lines[0].strip()  # Lấy tên section
        category_line = lines[1].strip()
        category_name = category_line.split(":")[1].strip() if ":" in category_line else "Không áp dụng"
        
        text = "\n".join(lines[2:]).strip()  # Lấy toàn bộ nội dung còn lại là text
        
        # Thêm vào danh sách dữ liệu
        data.append({
            "section": section_name,
            "category": category_name,
            "text": text
        })
    
    return data

# Hàm để đưa dữ liệu vào Pinecone
def upsert_to_pinecone(data):
    vectors = []
    for entry in data:
        embedding = create_embedding(entry['text'])  # Tạo embedding cho văn bản
        metadata = {
            "section": entry['section'],
            "category": entry['category'],
            "text": entry['text']
        }
        # Thêm vector và metadata vào Pinecone
        vector_id = str(uuid.uuid4())
        vectors.append((vector_id, embedding, metadata))
    
    # Upsert vectors vào Pinecone
    index.upsert(vectors)
    print("Dữ liệu đã được thêm vào Pinecone.")

if __name__ == "__main__":
    # Đọc và tiền xử lý dữ liệu từ file 'preprocess_data.txt'
    data_from_file = read_and_preprocess_file("preprocess_data.txt")

    # Gọi hàm để thêm dữ liệu vào Pinecone
    upsert_to_pinecone(data_from_file)
