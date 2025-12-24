"""
MBE Document Assistant - Main Entry Point
"""
import os
from DocumentProcessor import Processor
from ChatEngine import Chat
from ui_manager import UIManager

class MBEAssistant:
    def __init__(self):
        self.config = {
            'groq_api_key': os.getenv("GROQ_API_KEY"),
            'groq_model': "llama-3.3-70b-versatile",
            'pdf_password': "mbe2025",
            'docs_folder': "/mount/src/lasst/documents",
            'cache_folder': os.getenv("CACHE_FOLDER", "./cache"),
            'chunk_size': 1500,
            'overlap': 250
        }
        
        # Ensure folders exist
        os.makedirs(self.config['docs_folder'], exist_ok=True)
        os.makedirs(self.config['cache_folder'], exist_ok=True)
        
        # Initialize components
        self.doc_processor = Processor(self.config)
        self.chat_engine = Chat(self.config)
        self.ui_manager = UIManager(self.doc_processor, self.chat_engine)
    
    def run(self):
        """Main execution method"""
        self.ui_manager.render()

if __name__ == "__main__":
    app = MBEAssistant()
    app.run()
