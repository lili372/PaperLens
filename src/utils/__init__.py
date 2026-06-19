import hashlib
import os
import time

from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


def is_text_pdf(pdf_path):
    import fitz

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    if total_pages == 0:
        return False

    text_pages = 0
    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text.strip():  # 检查是否有文本内容
            text_pages += 1

    # 计算有文本内容的页面比例
    text_ratio = text_pages / total_pages
    # 如果超过50%的页面有文本内容，则认为是文本PDF
    return text_ratio > 0.5


def hashstr(input_string, length=None, with_salt=False):
    """生成字符串的哈希值
    Args:
        input_string: 输入字符串
        length: 截取长度，默认为None，表示不截取
        with_salt: 是否加盐，默认为False
    """
    try:
        # 尝试直接编码
        encoded_string = str(input_string).encode("utf-8")
    except UnicodeEncodeError:
        # 如果编码失败，替换无效字符
        encoded_string = str(input_string).encode("utf-8", errors="replace")

    if with_salt:
        salt = str(time.time())
        encoded_string = (encoded_string.decode("utf-8") + salt).encode("utf-8")

    hash = hashlib.md5(encoded_string).hexdigest()
    if length:
        return hash[:length]
    return hash
