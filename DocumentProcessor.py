import re
import fitz
import glob
import os
import pickle
import hashlib
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PDF_PASSWORD = os.getenv("PDF_PASSWORD", "")
DOCS_FOLDER = "/mount/src/lasst/documents"
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
        st.warning(str(e))

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def structure_text_into_paragraphs(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    paragraphs = []
    current = []
    for line in lines:
        if re.match(r'^[\d]+[\.\)]\s|^[•\-\*]\s', line):
            if current:
                paragraphs.append(' '.join(current))
                current = []
            paragraphs.append(line)
        else:
            current.append(line)
    if current:
        paragraphs.append(' '.join(current))
    return '\n\n'.join(paragraphs)

def create_smart_chunks(text, chunk_size=800, overlap=100, page_num=None, source_file=None, is_table=False, table_num=None, catalog=None, section=None):
    words = text.split()
    chunks = []
    metadata = {
        "page": str(page_num),
        "source": source_file,
        "is_table": str(is_table),
        "table_number": str(table_num) if table_num else "N/A",
        "catalog": catalog,
        "section": section
    }
    if len(words) <= chunk_size:
        if text.strip():
            chunks.append({"content": text.strip(), "metadata": metadata})
        return chunks
    for i in range(0, len(words), chunk_size - overlap):
        part = words[i:i + chunk_size]
        if len(part) >= 40:
            chunks.append({"content": " ".join(part), "metadata": metadata.copy()})
    return chunks

def format_table_as_structured_text(table, table_number):
    headers = [str(c).strip() for c in table[0]]
    text = f"\nTable {table_number}\n\n"
    text += "| " + " | ".join(headers) + " |\n"
    text += "| " + " --- |" * len(headers) + "\n"
    for row in table[1:]:
        text += "| " + " | ".join(str(c).strip() for c in row) + " |\n"
    return text

def extract_pdf_detailed(filepath):
    try:
        doc = fitz.open(filepath)
        if doc.is_encrypted and not doc.authenticate(PDF_PASSWORD):
            return None, "Wrong PDF password"
    except Exception as e:
        return None, str(e)

    filename = os.path.basename(filepath)
    chunks = []
    current_catalog = None
    current_section = None
    table_counter = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block.get("type") != 0:
                continue
            block_text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")
            clean_block = block_text.strip()
            if not clean_block:
                continue
            if re.search(r'katalog\s*a', clean_block, re.IGNORECASE):
                current_catalog = "Katalog A"
            if re.search(r'katalog\s*b', clean_block, re.IGNORECASE):
                current_catalog = "Katalog B"
            if re.search(r'anlage\s*\d+', clean_block, re.IGNORECASE):
                current_section = clean_block
            structured = structure_text_into_paragraphs(clean_block)
            chunks.extend(create_smart_chunks(
                structured,
                page_num=page_num + 1,
                source_file=filename,
                catalog=current_catalog,
                section=current_section
            ))

        tables = page.find_tables()
        for table in tables.tables:
            extracted = table.extract()
            if extracted:
                table_counter += 1
                table_text = format_table_as_structured_text(extracted, table_counter)
                chunks.extend(create_smart_chunks(
                    table_text,
                    chunk_size=1500,
                    overlap=0,
                    page_num=page_num + 1,
                    source_file=filename,
                    is_table=True,
                    table_num=table_counter,
                    catalog=current_catalog,
                    section=current_section
                ))

    doc.close()
    return {"chunks": chunks}, None

def extract_docx_detailed(filepath):
    return {"chunks": []}, None

def extract_txt_detailed(filepath):
    filename = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    structured = structure_text_into_paragraphs(text)
    chunks = create_smart_chunks(
        structured,
        page_num=1,
        source_file=filename
    )
    return {"chunks": chunks}, None

def get_files_from_folder():
    return glob.glob(os.path.join(DOCS_FOLDER, "*.[pP][dD][fF]")) + \
           glob.glob(os.path.join(DOCS_FOLDER, "*.[dD][oO][cC][xX]")) + \
           glob.glob(os.path.join(DOCS_FOLDER, "*.txt"))
