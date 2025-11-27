import os
from typing import Callable, Optional
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np
import faiss
from phishing_knowledge import phishing_patterns
from gemini_model_manager import GeminiModelManager

load_dotenv()

LogFn = Optional[Callable[[str], None]]

class PhishingRAG:
    def __init__(self, logger: LogFn = None):
        self.log = logger or print

        # Initialize Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model_manager = GeminiModelManager(
            preferred_models=['models/gemini-pro-latest'],
            logger=lambda msg: self.log(f"[PhishingRAG] {msg}")
        )
        self.log(
            f"[PhishingRAG] Candidate analysis models: "
            f"{', '.join(self.model_manager.candidate_models)}"
        )
        
        # Store patterns
        self.patterns = phishing_patterns
        self.pattern_texts = []
        self.embeddings = []
        
        # Load knowledge base
        self._load_knowledge()
    
    def _get_embedding(self, text):
        """Get embedding from Gemini"""
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    
    def _load_knowledge(self):
        """Load phishing patterns and create embeddings"""
        self.log("Loading knowledge base...")
        
        for pattern in self.patterns:
            text = f"{pattern['pattern']}: {pattern['description']} Example: {pattern['example']}"
            self.pattern_texts.append(text)
            
            # Get embedding
            embedding = self._get_embedding(text)
            self.embeddings.append(embedding)
        
        # Convert to numpy array
        self.embeddings = np.array(self.embeddings).astype('float32')
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        
        self.log(f"Loaded {len(self.patterns)} patterns into knowledge base")
    
    def retrieve(self, sms_message, n_results=3):
        """Retrieve relevant phishing patterns"""
        # Get embedding for the query
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=sms_message,
            task_type="retrieval_query"
        )
        query_embedding = np.array([result['embedding']]).astype('float32')
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, n_results)
        
        # Return the relevant texts
        results = [self.pattern_texts[i] for i in indices[0]]
        return results
    
    def generate_response(self, sms_message, context):
        """Use Gemini to analyze if SMS is phishing"""
        prompt = f"""You are a cybersecurity expert analyzing SMS messages for phishing.

        Context (known phishing patterns):
        {chr(10).join(context)}

        SMS Message to analyze:
        "{sms_message}"

        Analyze this message and determine if it's phishing. Provide:
        1. Classification (PHISHING or LEGITIMATE)
        2. Confidence level (0-100%)
        3. Reasoning based on the context provided

        Format your response as:
        Classification: [PHISHING/LEGITIMATE]
        Confidence: [0-100]%
        Reasoning: [Your explanation]"""

        response = self.model_manager.generate_content(prompt)
        return response.text
    
    def analyze(self, sms_message):
        """Complete RAG pipeline"""
        # Retrieve relevant context
        context = self.retrieve(sms_message)
        
        # Generate response
        response = self.generate_response(sms_message, context)
        
        return {
            "sms": sms_message,
            "context": context,
            "analysis": response
        }