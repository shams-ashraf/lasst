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
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for c in iter(lambda: f.read(4096), b""):
            h.update(c)
    return h.hexdigest()

def load_cache(key):
    p = os.path.join(CACHE_FOLDER, f"{key}.pkl")
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except:
            return None
    return None

def save_cache(key, data):
    with open(os.path.join(CACHE_FOLDER, f"{key}.pkl"), "wb") as f:
        pickle.dump(data, f)

def structure_text(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    out = []
    cur = []
    for l in lines:
        if re.match(r'^[\d]+[\.\)]\s|^[•\-\*]\s', l):
            if cur:
                out.append(" ".join(cur))
                cur = []
            out.append(l)
        else:
            cur.append(l)
    if cur:
        out.append(" ".join(cur))
    return "\n\n".join(out)

def chunk(text, page, source, is_table=False, table_number=None):
    words = text.split()
    meta = {
        "page": str(page),
        "source": source,
        "is_table": str(is_table),
        "table_number": str(table_number) if table_number else "N/A"
    }
    if len(words) <= 800:
        return [{"content": text, "metadata": meta}]
    out = []
    for i in range(0, len(words), 700):
        part = words[i:i+800]
        if len(part) >= 40:
            out.append({"content": " ".join(part), "metadata": meta.copy()})
    return out

def format_table(t, n):
    h = [str(c).strip() for c in t[0]]
    s = f"\nTable {n}\n\n"
    s += "| " + " | ".join(h) + " |\n"
    s += "| " + " --- |" * len(h) + "\n"
    for r in t[1:]:
        s += "| " + " | ".join(str(c).strip() for c in r) + " |\n"
    return s

def extract_pdf_detailed(path):
    doc = fitz.open(path)
    if doc.is_encrypted and not doc.authenticate(PDF_PASSWORD):
        return None, "Wrong PDF password"
    name = os.path.basename(path)
    chunks = []
    tcount = 0
    for i in range(len(doc)):
        page = doc[i]
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b.get("type") != 0:
                continue
            txt = ""
            for l in b["lines"]:
                for s in l["spans"]:
                    txt += s["text"]
            txt = txt.strip()
            if not txt:
                continue
            chunks.extend(chunk(structure_text(txt), i+1, name))
        tables = page.find_tables()
        for tb in tables.tables:
            data = tb.extract()
            if data:
                tcount += 1
                chunks.extend(chunk(format_table(data, tcount), i+1, name, True, tcount))
    doc.close()
    return {"chunks": chunks}, None

def extract_docx_detailed(path):
    return {"chunks": []}, None

def extract_txt_detailed(path):
    name = os.path.basename(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    return {"chunks": chunk(structure_text(txt), 1, name)}, None

def get_files_from_folder():
    return glob.glob(os.path.join(DOCS_FOLDER, "*.pdf")) + \
           glob.glob(os.path.join(DOCS_FOLDER, "*.docx")) + \
           glob.glob(os.path.join(DOCS_FOLDER, "*.txt"))
