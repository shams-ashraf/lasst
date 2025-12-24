"""
Document Processing Module - مسؤول عن استخراج ومعالجة المستندات
"""
import re
import fitz
import docx
import glob
import os
import pickle
import hashlib
from typing import List, Dict, Tuple, Optional

class Processor:
    def __init__(self, config):
        self.config = config
        self.cache_folder = config['cache_folder']
        self.docs_folder = config['docs_folder']
        self.pdf_password = config['pdf_password']
    
    def get_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def load_cache(self, cache_key: str) -> Optional[Dict]:
        """Load cached processed data"""
        cache_file = os.path.join(self.cache_folder, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
    
    def save_cache(self, cache_key: str, data: Dict):
        """Save processed data to cache"""
        cache_file = os.path.join(self.cache_folder, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
        return text.strip()
    
    def structure_text_into_paragraphs(self, text: str) -> str:
        """Convert raw text into structured paragraphs"""
        if not text or not text.strip():
            return ""
        
        text = self.clean_text(text)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return ""
        
        paragraphs = []
        current_paragraph = []
        
        for i, line in enumerate(lines):
            words_in_line = line.split()
            
            if len(words_in_line) < 3 and not (line[0].isupper() or re.match(r'^[\d]+[\.\):]', line)):
                continue
            
            is_heading = (
                (line.isupper() and len(words_in_line) <= 10) or
                (len(words_in_line) <= 6 and line[0].isupper() and line.endswith(':'))
            )
            
            if is_heading:
                if current_paragraph:
                    paragraph_text = ' '.join(current_paragraph)
                    paragraph_text = re.sub(r'\s+', ' ', paragraph_text)
                    paragraph_text = re.sub(r'\s+([.,!?;:])', r'\1', paragraph_text)
                    paragraphs.append(paragraph_text.strip())
                    current_paragraph = []
                paragraphs.append(f"\n🔹 {line}\n")
                continue
            
            is_list_item = re.match(r'^[\d]+[\.\)]\s', line) or re.match(r'^[•\-\*]\s', line)
            
            if is_list_item:
                if current_paragraph:
                    paragraph_text = ' '.join(current_paragraph)
                    paragraph_text = re.sub(r'\s+', ' ', paragraph_text)
                    paragraph_text = re.sub(r'\s+([.,!?;:])', r'\1', paragraph_text)
                    paragraphs.append(paragraph_text.strip())
                    current_paragraph = []
                paragraphs.append(f" {line}")
                continue
            
            current_paragraph.append(line)
            
            ends_with_punctuation = line.endswith(('.', '!', '?', '؟', '!', '。'))
            next_is_new_section = False
            
            if i < len(lines) - 1:
                next_line = lines[i + 1]
                next_words = next_line.split()
                next_is_new_section = (
                    re.match(r'^[\d]+[\.\)]\s', next_line) or
                    re.match(r'^[•\-\*]\s', next_line) or
                    (len(next_words) <= 6 and next_line[0].isupper()) or
                    next_line.isupper()
                )
            
            is_last_line = (i == len(lines) - 1)
            
            if (ends_with_punctuation or next_is_new_section or is_last_line):
                if current_paragraph:
                    paragraph_text = ' '.join(current_paragraph)
                    paragraph_text = re.sub(r'\s+', ' ', paragraph_text)
                    paragraph_text = re.sub(r'\s+([.,!?;:])', r'\1', paragraph_text)
                    paragraphs.append(paragraph_text.strip())
                    current_paragraph = []
        
        if paragraphs:
            structured_text = ""
            for para in paragraphs:
                if para.startswith('\n🔹'):
                    structured_text += para
                elif para.startswith(' '):
                    structured_text += para + "\n"
                else:
                    structured_text += para + "\n\n"
            return structured_text.strip()
        
        return text
    
    def create_smart_chunks(self, text: str, chunk_size: int = None, 
                           overlap: int = None, page_num: int = None, 
                           source_file: str = None, is_table: bool = False, 
                           table_num: int = None) -> List[Dict]:
        """Create intelligent chunks with metadata"""
        chunk_size = chunk_size or self.config['chunk_size']
        overlap = overlap or self.config['overlap']
        
        words = text.split()
        chunks = []
        
        metadata = {
            'page': str(page_num) if page_num is not None else "N/A",
            'source': source_file if source_file else "Unknown",
            'is_table': str(is_table),
            'table_number': str(table_num) if table_num is not None else "N/A"
        }
        
        if len(words) <= chunk_size:
            if text.strip():
                return [{
                    'content': text,
                    'metadata': metadata
                }]
            return []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = " ".join(chunk_words)
            if len(chunk.split()) >= 30:
                chunks.append({
                    'content': chunk,
                    'metadata': metadata.copy()
                })
        
        return chunks
    
    def format_table_as_structured_text(self, extracted_table: List[List], 
                                       table_number: int = None) -> str:
        """Format extracted table as markdown"""
        if not extracted_table or len(extracted_table) == 0:
            return ""
        
        headers = [str(cell).strip() if cell else f"Column_{i+1}" 
                  for i, cell in enumerate(extracted_table[0])]
        headers = [self.clean_text(h) for h in headers]
        
        text = f"\n📊 Table {table_number or ''}\n\n"
        if headers:
            text += "| " + " | ".join(headers) + " |\n"
            text += "| " + " --- |" * len(headers) + " |\n"
        
        row_count = 0
        for row in extracted_table[1:]:
            cells = [str(cell).strip() if cell else "" for cell in row]
            cells = [self.clean_text(c) for c in cells]
            if any(cells):
                text += "| " + " | ".join(cells) + " |\n"
                row_count += 1
        
        text += f"\n**Summary**: {row_count} rows, {len(headers)} columns.\n"
        return text
    
    def extract_pdf(self, filepath: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Extract content from PDF file"""
        try:
            doc = fitz.open(filepath)
            if doc.is_encrypted:
                if not doc.authenticate(self.pdf_password):
                    doc.close()
                    return None, "Invalid PDF password"
        except Exception as e:
            return None, f"Error opening PDF: {str(e)}"
        
        filename = os.path.basename(filepath)
        file_info = {
            'chunks': [],
            'total_pages': len(doc),
            'total_tables': 0,
            'pages_with_tables': [],
        }
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            all_elements = []
            
            # Extract text blocks
            text_blocks = page.get_text("dict")["blocks"]
            for block in text_blocks:
                if block.get('type') == 0:
                    y_pos = block.get('bbox', [0, 0, 0, 0])[1]
                    text_content = ""
                    for line in block.get('lines', []):
                        for span in line.get('spans', []):
                            text_content += span.get('text', '') + ' '
                    if text_content.strip():
                        structured = self.structure_text_into_paragraphs(text_content)
                        all_elements.append({
                            'type': 'text',
                            'y_position': y_pos,
                            'content': structured,
                            'page': page_num + 1
                        })
            
            # Extract tables
            tables = page.find_tables()
            if tables and len(tables.tables) > 0:
                file_info['pages_with_tables'].append(page_num + 1)
                
                for table_num, table in enumerate(tables.tables, 1):
                    file_info['total_tables'] += 1
                    extracted = table.extract()
                    if extracted:
                        table_text = self.format_table_as_structured_text(
                            extracted, file_info['total_tables']
                        )
                        all_elements.append({
                            'type': 'table',
                            'y_position': table.bbox[1] if table.bbox else 0,
                            'content': table_text,
                            'page': page_num + 1,
                            'table_num': file_info['total_tables']
                        })
            
            all_elements.sort(key=lambda x: x['y_position'])
            
            # Build page text
            page_text = f"\n# Document: {filename}\n\n"
            page_text += f"\n{'═' * 60}\n📄 Page {page_num + 1}\n{'═' * 60}\n\n"
            for element in all_elements:
                page_text += element['content'] + "\n\n"
            
            # Create chunks
            page_chunks = self.create_smart_chunks(
                page_text,
                page_num=page_num + 1,
                source_file=filename
            )
            file_info['chunks'].extend(page_chunks)
            
            # Add table chunks separately
            for element in all_elements:
                if element['type'] == 'table':
                    table_chunks = self.create_smart_chunks(
                        element['content'],
                        chunk_size=2000,
                        overlap=0,
                        page_num=element['page'],
                        source_file=filename,
                        is_table=True,
                        table_num=element.get('table_num')
                    )
                    file_info['chunks'].extend(table_chunks)
        
        doc.close()
        return file_info, None
    
    def extract_docx(self, filepath: str) -> Tuple[Dict, Optional[str]]:
        """Extract content from DOCX file"""
        doc = docx.Document(filepath)
        filename = os.path.basename(filepath)
        
        file_info = {
            'chunks': [],
            'total_pages': 1,
            'total_tables': 0,
            'pages_with_tables': [],
        }
        
        all_text = []
        table_counter = 0
        
        for element in doc.element.body:
            if element.tag.endswith('p'):
                for para in doc.paragraphs:
                    if para._element == element:
                        text = self.clean_text(para.text)
                        if text:
                            structured = self.structure_text_into_paragraphs(text)
                            if structured:
                                all_text.append(structured)
                        break
            
            elif element.tag.endswith('tbl'):
                for table in doc.tables:
                    if table._element == element:
                        file_info['total_tables'] += 1
                        table_counter += 1
                        table_text = self.format_table_as_structured_text(
                            [[cell.text for cell in row.cells] for row in table.rows],
                            table_counter
                        )
                        if table_text:
                            all_text.append(table_text)
                            table_chunks = self.create_smart_chunks(
                                table_text,
                                chunk_size=2000,
                                overlap=0,
                                page_num=1,
                                source_file=filename,
                                is_table=True,
                                table_num=table_counter
                            )
                            file_info['chunks'].extend(table_chunks)
                        break
        
        complete_text = "\n\n".join(all_text)
        text_chunks = self.create_smart_chunks(
            complete_text,
            page_num=1,
            source_file=filename
        )
        file_info['chunks'].extend(text_chunks)
        
        if file_info['total_tables'] > 0:
            file_info['pages_with_tables'] = [1]
        
        return file_info, None
    
    def extract_txt(self, filepath: str) -> Tuple[Dict, Optional[str]]:
        """Extract content from TXT file"""
        filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        structured_text = self.structure_text_into_paragraphs(text)
        chunks = self.create_smart_chunks(
            structured_text,
            page_num=1,
            source_file=filename
        )
        
        file_info = {
            'chunks': chunks,
            'total_pages': 1,
            'total_tables': 0,
            'pages_with_tables': [],
        }
        return file_info, None
    
    def get_available_files(self) -> List[str]:
        """Get list of supported files in docs folder"""
        supported_extensions = ['*.pdf', '*.docx', '*.doc', '*.txt']
        files = []
        for ext in supported_extensions:
            files.extend(glob.glob(os.path.join(self.docs_folder, ext)))
        return files
    
    def process_file(self, filepath: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Process a single file with caching"""
        filename = os.path.basename(filepath)
        file_ext = filename.split('.')[-1].lower()
        
        # Check cache
        file_hash = self.get_file_hash(filepath)
        cache_key = f"{file_hash}_{file_ext}"
        cached_data = self.load_cache(cache_key)
        
        if cached_data:
            return cached_data, None
        
        # Process file based on type
        if file_ext == 'pdf':
            file_info, error = self.extract_pdf(filepath)
        elif file_ext in ['docx', 'doc']:
            file_info, error = self.extract_docx(filepath)
        elif file_ext == 'txt':
            file_info, error = self.extract_txt(filepath)
        else:
            return None, "Unsupported file type"
        
        if error:
            return None, error
        
        # Save to cache
        self.save_cache(cache_key, file_info)
        return file_info, None
