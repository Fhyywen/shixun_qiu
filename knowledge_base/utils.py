import os
import hashlib
from datetime import datetime


def get_file_hash(file_path):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def format_timestamp(timestamp):
    """格式化时间戳"""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def safe_join(base_path, *paths):
    """安全的路径拼接，防止目录遍历攻击"""
    final_path = os.path.join(base_path, *paths)
    final_path = os.path.normpath(final_path)

    if not final_path.startswith(os.path.abspath(base_path)):
        return None  # 路径超出基目录

    return final_path


def chunk_text_by_sentences(text, max_words=500, overlap=50):
    """按句子分块文本，保持语义完整性"""
    import re

    # 分割句子
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)

        if current_word_count + word_count > max_words and current_chunk:
            # 保存当前块
            chunks.append(' '.join(current_chunk))

            # 保留重叠部分
            if overlap > 0:
                overlap_text = ' '.join(current_chunk)
                overlap_words = overlap_text.split()[-overlap:]
                current_chunk = [' '.join(overlap_words)]
                current_word_count = len(overlap_words)
            else:
                current_chunk = []
                current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += word_count

    # 添加最后一个块
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks