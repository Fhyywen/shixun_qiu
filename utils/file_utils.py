import os
import logging
from typing import Optional
import PyPDF2
import docx

logger = logging.getLogger(__name__)


def load_document(file_path: str) -> Optional[str]:
    """加载文档内容"""
    try:
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        elif file_path.endswith('.pdf'):
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text

        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])

        else:
            logger.warning(f"Unsupported file format: {file_path}")
            return None

    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}")
        return None


def split_text(text: str, chunk_size: int, chunk_overlap: int = 0) -> List[str]:
    """分割文本为块"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)

        # 确保不在单词中间分割
        if end < len(text):
            while end > start and text[end] not in ' \n\t.,!?;:':
                end -= 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap if chunk_overlap > 0 else end

        if start >= len(text):
            break

    return chunks


def get_supported_formats() -> List[str]:
    """获取支持的文档格式"""
    return ['.txt', '.pdf', '.docx', '.md', '.csv', '.json']