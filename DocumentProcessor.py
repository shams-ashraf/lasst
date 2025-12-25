# <DOCUMENT filename="DocumentProcessor.py"> (أكبر تحسين: إضافة OCR كامل بـ PyMuPDF built-in + تحسين chunking)
import streamlit as st
import re
import fitz  # PyMuPDF
import docx
import glob
import os
import pickle
import hashlib

PDF_PASSWORD = "mbe2025"
DOCS_FOLDER = "/mount/src/chatbotlast/documents"  # أو غيره حسب البيئة
CACHE_FOLDER = os.getenv("CACHE_FOLDER", "./cache")

os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

def get_file_hash(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_cache(cache_key):
    cache_file = os.path.join(CACHE_FOLDER, f"{cache_key}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None

def save_cache(cache_key, data):
    cache_file = os.path.join(CACHE_FOLDER, f"{cache_key}.pkl")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        st.warning(f"⚠️ Cache save error: {str(e)}")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def structure_text_into_paragraphs(text):
    # نفس الكود السابق مع تحسينات طفيفة (شيلنا التكرار)
    if not text.strip():
        return ""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    paragraphs = []
    current = []
    for line in lines:
        if re.match(r'^[\d]+[\.\)]\s|^[•\-\*]\s|^\n*🔹', line):
            if current:
                paragraphs.append(' '.join(current))
                current = []
            paragraphs.append(line)
        else:
            current.append(line)
    if current:
        paragraphs.append(' '.join(current))
    return '\n\n'.join(paragraphs)

def create_smart_chunks(text, chunk_size=800, overlap=100, page_num=None, source_file=None, is_table=False, table_num=None):
    words = text.split()
    chunks = []
    metadata = {
        'page': str(page_num) if page_num is not None else "N/A",
        'source': source_file or "Unknown",
        'is_table': str(is_table),
        'table_number': str(table_num) if table_num else "N/A"
    }
    if len(words) <= chunk_size:
        if text.strip():
            chunks.append({'content': text.strip(), 'metadata': metadata})
        return chunks

    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        if len(chunk_words) >= 50:  # تجنب chunks صغيرة جداً
            chunks.append({'content': chunk_text, 'metadata': metadata.copy()})
    return chunks

def format_table_as_structured_text(table, table_number=None):
    if not table or len(table) == 0:
        return ""
    headers = [str(cell).strip() or f"Col_{i+1}" for i, cell in enumerate(table[0])]
    text = f"\n📊 Table {table_number or ''}\n\n"
    text += "| " + " | ".join(headers) + " |\n"
    text += "| " + " --- |" * len(headers) + " |\n"
    for row in table[1:]:
        cells = [str(cell).strip() for cell in row]
        if any(cells):
            text += "| " + " | ".join(cells) + " |\n"
    return text

def extract_pdf_detailed(filepath):
    try:
        doc = fitz.open(filepath)
        if doc.is_encrypted and not doc.authenticate(PDF_PASSWORD):
            return None, "❌ Wrong PDF password"
    except Exception as e:
        return None, f"❌ PDF open error: {str(e)}"

    filename = os.path.basename(filepath)
    file_info = {'chunks': [], 'total_pages': len(doc), 'total_tables': 0}

    for page_num in range(len(doc)):
        page = doc[page_num]

        # OCR كامل للصفحة إذا كان النص قليل (يعني scanned محتمل)
        text = page.get_text("text")
        if len(text.strip()) < 100:  # إذا نص قليل → OCR كامل
            textpage = page.get_textpage_ocr(flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE, full=True)
            text = page.get_text("text", textpage=textpage)

        # استخراج blocks للـ structuring
        blocks = page.get_text("dict")["blocks"]
        page_text = f"# {filename} - Page {page_num + 1}\n\n"
        for block in blocks:
            if block.get("type") == 0:
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "")
                if block_text.strip():
                    page_text += structure_text_into_paragraphs(block_text) + "\n\n"

        # Tables
        tables = page.find_tables()
        if tables:
            for t_num, table in enumerate(tables.tables, 1):
                file_info['total_tables'] += 1
                extracted = table.extract()
                if extracted:
                    table_text = format_table_as_structured_text(extracted, file_info['total_tables'])
                    page_text += table_text + "\n\n"
                    table_chunks = create_smart_chunks(table_text, chunk_size=1500, overlap=0, page_num=page_num+1,
                                                      source_file=filename, is_table=True, table_num=file_info['total_tables'])
                    file_info['chunks'].extend(table_chunks)

        # Chunk الصفحة كاملة
        page_chunks = create_smart_chunks(page_text, chunk_size=800, overlap=100, page_num=page_num+1, source_file=filename)
        file_info['chunks'].extend(page_chunks)

    doc.close()
    return file_info, None

# extract_docx_detailed و extract_txt_detailed بدون تغيير كبير (كانت كويسة)
# ... (نفس الكود السابق مع chunk_size=800, overlap=100)

def get_files_from_folder():
    return glob.glob(os.path.join(DOCS_FOLDER, "*.[pP][dD][fF]")) + \
           glob.glob(os.path.join(DOCS_FOLDER, "*.[dD][oO][cC][xX]")) + \
           glob.glob(os.path.join(DOCS_FOLDER, "*.txt"))
