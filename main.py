import openai
import pinecone
import os 
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from save_to_db import read_and_preprocess_file, create_embedding
load_dotenv()
api_key_pinecone = os.getenv("PINECONE_API_KEY")
index_name = "shop-info-index"

def get_all_sections_and_categories(data):
    sections = set()
    categories = set()
    for item in data:
        sections.add(item.get("section", "").strip())
        categories.add(item.get("category", "").strip())
    return list(sections), list(categories)

# Giả sử bạn đã load dữ liệu từ file (preprocess_data.txt)
data_from_file = read_and_preprocess_file("preprocess_data.txt")
all_sections, all_categories = get_all_sections_and_categories(data_from_file)
print(">>>all_sections", all_sections)
print(">>>all_categories", all_categories)

# Load Pinecone index
pc = Pinecone(api_key=api_key_pinecone)
index = pc.Index(index_name)
question = '''
Vậy tôi có thể thanh toán bằng hình thức nào?
'''

# Bước 1: Extract metadata (section & category) từ câu hỏi
def extract_metadata_from_question(question, sections, categories):
    prompt = f"""
    Ngữ cảnh: Bạn đang hỗ trợ phân loại câu hỏi của khách hàng trong hệ thống bán hàng.

    Câu hỏi của người dùng: "{question}"

    Dưới đây là danh sách các Section hiện có:
    {', '.join(sections)}

    Và các Category hiện có:
    {', '.join(categories)}

    Nhiệm vụ của bạn:
    1. Dự đoán section và category phù hợp với câu hỏi.
    2. Nếu câu hỏi không nói rõ là dành cho nam hay nữ (ví dụ: "quần jean", "giày size 42") thì hãy **bao gồm cả các khả năng phù hợp**, ví dụ: cả "Thời trang nam" và "Thời trang nữ".
    3. Có thể có **nhiều section và category** trong câu hỏi, hãy đưa ra đầy đủ.
    4. Trả về kết quả dưới dạng JSON với định dạng chuẩn sau:

    {{
    "section": ["<tên section 1>", "<tên section 2>", ...],
    "category": ["<tên category 1>", "<tên category 2>", ...]
    }}

    Lưu ý:
    - Nếu không xác định được thì để là "Không áp dụng".
    - Không thêm bất kỳ lời giải thích nào, chỉ trả về JSON hợp lệ.
    """.strip()

    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    import json
    try:
        result = json.loads(response.choices[0].message.content)
    except:
        result = {"section": "unknown", "category": "unknown"}
    print(">>>section&category of users: ", result)
    return result


def get_all_vectors_by_section_category(section, category):
    print(">>>category", category)
    # Dummy vector để kích hoạt query
    dummy_vector = [0.0] * 1536
    print(type(section))
    # Tách các giá trị nếu chứa dấu phẩy và loại bỏ khoảng trắng thừa
    section_list = [s.strip() for s in section] if section else []
    category_list = [c.strip() for c in category] if category else []

    # Xây dựng filter sử dụng $in nếu có nhiều giá trị
    filter_obj = {}
    if section_list:
        filter_obj["section"] = {"$in": section_list} if len(section_list) > 1 else section_list[0]
    if category_list:
        filter_obj["category"] = {"$in": category_list} if len(category_list) > 1 else category_list[0]

    # Truy vấn Pinecone
    result = index.query(
        vector=dummy_vector,
        top_k=1000,  # Số lượng kết quả lớn để lấy tất cả
        filter=filter_obj,
        include_metadata=True
    )

    return result.matches



def rag_generate_response(question, section, category):
    results = get_all_vectors_by_section_category(section, category)
    print(">>> similar result:", results)
    # Ghép văn bản lại
    context = "\n\n".join([match['metadata'].get('text', '') for match in results if match.metadata])
    if context is None:
        context == "None"
    # Đưa vào LLM để tạo câu trả lời
    prompt = f"""
    Bạn là một nhân viên bán hàng chuyên nghiệp và chỉ được trả lời dựa trên dữ liệu nội bộ.

    Dưới đây là danh sách sản phẩm hoặc thông tin hiện có của cửa hàng:
    {context}

    Câu hỏi của khách hàng: "{question}"

    Yêu cầu:
    - **Chỉ sử dụng thông tin có trong danh sách trên để trả lời**. Không được suy đoán hoặc trả lời theo hiểu biết chung nếu không có dữ liệu hỗ trợ.
    - Trả lời một cách tự nhiên, rõ ràng, đúng trọng tâm theo câu hỏi của khách.
    - Nếu **không tìm thấy thông tin phù hợp** trong danh sách trên, hãy trả lời đúng một câu:  
    **"Hiện tại cửa hàng không có sản phẩm hoặc thông tin phù hợp với yêu cầu của bạn."**
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response['choices'][0]['message']['content']

result_exact = extract_metadata_from_question(question, all_sections, all_categories)
sections = result_exact["section"] 
categories = result_exact["category"] 
# if isinstance(result_exact["category"], list) and len(result_exact["category"]) > 1:
#     categories = result_exact["category"] = ', '.join(result_exact["category"])

response = rag_generate_response(question,sections,categories)
print(">>>response: ", response)