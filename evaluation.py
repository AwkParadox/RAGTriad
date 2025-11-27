import google.generativeai as genai
import os
import time
from typing import Callable, Optional

from gemini_model_manager import GeminiModelManager

LogFn = Optional[Callable[[str], None]]

class RAGEvaluator:
    def __init__(self, rag_system, logger: LogFn = None):
        self.rag = rag_system
        self.log = logger or print
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model_manager = GeminiModelManager(
            preferred_models=[
                'models/gemini-2.0-flash-lite',
                'models/gemini-1.5-flash',
                'models/gemini-1.5-pro',
            ],
            logger=lambda msg: self.log(f"[RAGEvaluator] {msg}")
        )
        self.log(
            f"[RAGEvaluator] Candidate evaluator models: "
            f"{', '.join(self.model_manager.candidate_models)}"
        )
    
    def evaluate_context_relevance(self, sms_message, context):
        """Evaluate if retrieved context is relevant to the SMS"""
        prompt = f"""Rate how relevant these phishing patterns are to analyzing this SMS message.

        SMS: "{sms_message}"

        Retrieved Context:
        {chr(10).join([f"{i+1}. {c}" for i, c in enumerate(context)])}

        On a scale of 0-10, how relevant is this context for analyzing the SMS?
        Only respond with a number between 0-10."""

        response = self.model_manager.generate_content(prompt)
        
        try:
            score = float(response.text.strip())
            return min(max(score / 10, 0), 1)
        except:
            return 0.5
    
    def evaluate_groundedness(self, context, analysis):
        """Evaluate if the analysis is grounded in the retrieved context"""
        prompt = f"""Evaluate if this analysis is based on the provided context.

        Context:
        {chr(10).join([f"{i+1}. {c}" for i, c in enumerate(context)])}

        Analysis:
        {analysis}

        On a scale of 0-10, how well is the analysis grounded in the context? 
        (10 = completely based on context, 0 = ignores context or makes things up)
        Only respond with a number between 0-10."""

        response = self.model_manager.generate_content(prompt)
        
        try:
            score = float(response.text.strip())
            return min(max(score / 10, 0), 1)
        except:
            return 0.5
    
    def evaluate_answer_relevance(self, sms_message, analysis):
        """Evaluate if the analysis answers the phishing question"""
        prompt = f"""Evaluate if this analysis properly answers whether the SMS is phishing.

        SMS: "{sms_message}"

        Analysis:
        {analysis}

        On a scale of 0-10, how well does the analysis answer if this is phishing?
        (10 = clear classification with reasoning, 0 = doesn't answer the question)
        Only respond with a number between 0-10."""

        response = self.model_manager.generate_content(prompt)
        
        try:
            score = float(response.text.strip())
            return min(max(score / 10, 0), 1)
        except:
            return 0.5
    
    def evaluate_single(self, sms_message):
        """Evaluate a single SMS analysis with RAG Triad metrics"""
        self.log("")
        self.log("="*70)
        self.log(f"Analyzing SMS: {sms_message}")
        self.log("="*70)
        
        result = self.rag.analyze(sms_message)
        time.sleep(35)  # Wait to avoid rate limits (2 requests/min = 30+ sec wait)
        
        self.log("")
        self.log("📚 RETRIEVED CONTEXT:")
        for i, ctx in enumerate(result['context'], 1):
            self.log(f"  {i}. {ctx[:100]}...")
        
        self.log("")
        self.log("🔍 ANALYSIS:")
        self.log(f"  {result['analysis']}")
        
        self.log("")
        self.log("📊 RAG TRIAD EVALUATION:")
        self.log("  (Evaluating metrics - this takes ~10 seconds due to rate limits...)")
        
        context_rel = self.evaluate_context_relevance(sms_message, result['context'])
        time.sleep(35)  # Wait between API calls
        self.log(
            f"  ✓ Context Relevance:  {context_rel:.2f} "
            f"{'✅' if context_rel > 0.7 else '⚠️' if context_rel > 0.5 else '❌'}"
        )
        
        groundedness = self.evaluate_groundedness(result['context'], result['analysis'])
        time.sleep(35)  # Wait between API calls
        self.log(
            f"  ✓ Groundedness:       {groundedness:.2f} "
            f"{'✅' if groundedness > 0.7 else '⚠️' if groundedness > 0.5 else '❌'}"
        )
        
        answer_rel = self.evaluate_answer_relevance(sms_message, result['analysis'])
        time.sleep(35)  # Wait between API calls
        self.log(
            f"  ✓ Answer Relevance:   {answer_rel:.2f} "
            f"{'✅' if answer_rel > 0.7 else '⚠️' if answer_rel > 0.5 else '❌'}"
        )
        
        avg_score = (context_rel + groundedness + answer_rel) / 3
        self.log(
            f"📈 AVERAGE SCORE:     {avg_score:.2f} "
            f"{'✅' if avg_score > 0.7 else '⚠️' if avg_score > 0.5 else '❌'}"
        )
        
        return {
            **result,
            "metrics": {
                "context_relevance": context_rel,
                "groundedness": groundedness,
                "answer_relevance": answer_rel,
                "average": avg_score
            }
        }
    
    def evaluate_batch(self, sms_messages):
        """Evaluate multiple SMS messages"""
        results = []
        total = len(sms_messages)
        
        for idx, sms in enumerate(sms_messages, 1):
            self.log(f"[Processing {idx}/{total}]")
            result = self.evaluate_single(sms)
            results.append(result)
        
        self.log("")
        self.log("="*70)
        self.log("📊 BATCH EVALUATION SUMMARY")
        self.log("="*70)
        
        avg_context = sum(r['metrics']['context_relevance'] for r in results) / len(results)
        avg_ground = sum(r['metrics']['groundedness'] for r in results) / len(results)
        avg_answer = sum(r['metrics']['answer_relevance'] for r in results) / len(results)
        overall = (avg_context + avg_ground + avg_answer) / 3
        
        self.log(f"  Messages Analyzed: {len(results)}")
        self.log(f"  Avg Context Relevance:  {avg_context:.2f}")
        self.log(f"  Avg Groundedness:       {avg_ground:.2f}")
        self.log(f"  Avg Answer Relevance:   {avg_answer:.2f}")
        self.log(f"  Overall Score:          {overall:.2f}")
        self.log("="*70)
        
        return results