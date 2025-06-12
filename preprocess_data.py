import openai
import os
from dotenv import load_dotenv
# Nhập API key của bạn
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# models = openai.Model.list()
# print(models)

with open("ABC_Store.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Hàm phân loại section và category
def classify_section_and_category(text):
    prompt = f"""
    Bạn là một trợ lý AI giúp phân loại thông tin và sản phẩm từ văn bản.

    Yêu cầu:
    - Dựa vào nội dung sau, hãy phân loại mỗi đoạn thông tin theo:
    + Section: là nhóm thông tin có nội dung liên quan, do bạn tự xác định (ví dụ: thông tin shop, sản phẩm, vận chuyển, điều khoản...).
    + Category: là loại sản phẩm cụ thể nếu có (ví dụ: Thời trang nam, Thời trang nữ, Phụ kiện,...). Nếu phần đó không nói về sản phẩm thì để là "Không áp dụng".

    - Mỗi phần output phải theo đúng định dạng sau:
    Section: <Tên Section>
    Category: <Tên Category>
    <Dữ liệu liên quan được giữ nguyên dòng và format gốc>

    - Ngăn cách giữa các bản ghi bằng một dòng trống (xuống dòng 2 lần).

    - Nếu format đầu vào không chuẩn, bạn vẫn phải cố gắng phân tích và giữ kết quả rõ ràng, không tự chế lại định dạng.

    Dữ liệu đầu vào:
    {text}
    """.strip()
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Bạn là một trợ lý giúp phân loại sản phẩm và thông tin shop."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
    )
    
    return response['choices'][0]['message']['content'].strip()

result = classify_section_and_category(text)

with open("preprocess_data.txt", "w", encoding="utf-8") as file:
    file.write(result)

print(result)