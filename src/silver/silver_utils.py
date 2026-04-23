"""
silver_utils.py — Utilities for Silver Layer
"""

from pyspark.sql import functions as F
from pyspark.sql.column import Column

def advanced_clean_text(col: Column) -> Column:
    """
    Pipeline làm sạch văn bản toàn diện bằng PySpark Regex.
    Áp dụng tuần tự các phép biến đổi để loại bỏ nhiễu.
    """
    
    # 1. Loại bỏ các phần tử do việc nối chuỗi tạo ra (Artifacts)
    # Xóa " | {}" ở cuối câu (đặc trưng của Item Text)
    cleaned = F.regexp_replace(col, r" \|\s*\{\}\s*$", "")
    # Xóa các dấu ngoặc nhọn và ngoặc kép thừa từ chuỗi JSON
    cleaned = F.regexp_replace(cleaned, r"[\{\}\"]", "")
    
    # 2. Xóa thẻ HTML
    cleaned = F.regexp_replace(cleaned, r"<[^>]+>", " ")
    
    # 3. Xóa URL (http/https)
    cleaned = F.regexp_replace(cleaned, r"http[s]?://\S+", " ")
    
    # 4. Giữ lại Emojis (LLaMA-3 có thể khai thác sentiment từ emoji)
    # 5. Loại bỏ Ký tự đặc biệt gây nhiễu, GIỮ LẠI dấu câu cơ bản, Unicode Tiếng Việt, tiếng Anh và Emojis
    cleaned = F.regexp_replace(cleaned, r"[*_~^\\#@%\+=\|<>`]", " ")
    
    # 6. Chuyển thành chữ thường (Lowercase)
    cleaned = F.lower(cleaned)
    
    # 7. Chuẩn hóa khoảng trắng (Thực hiện cuối cùng)
    cleaned = F.regexp_replace(cleaned, r"\s{2,}", " ")
    cleaned = F.trim(cleaned)
    
    return cleaned
