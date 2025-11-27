import json
from pathlib import Path

from rag_system import PhishingRAG
from evaluation import RAGEvaluator
from file_logger import FileLogger

OUTPUT_FILE = "evaluation_output.txt"
TEST_MESSAGES_FILE = Path("sms_test_messages.json")

DEFAULT_TEST_MESSAGES = [
    "URGENT: Your bank account has been compromised! Click here immediately to secure it: bit.ly/secure123"
]

def load_test_messages(log):
    if TEST_MESSAGES_FILE.exists():
        try:
            data = json.loads(TEST_MESSAGES_FILE.read_text(encoding="utf-8"))
            messages = [
                sms.strip() for sms in data
                if isinstance(sms, str) and sms.strip()
            ]
            if messages:
                log(f"Loaded {len(messages)} test SMS messages from {TEST_MESSAGES_FILE}")
                return messages
            else:
                log(f"No valid SMS messages found in {TEST_MESSAGES_FILE}, using defaults.")
        except Exception as exc:
            log(f"Failed to read {TEST_MESSAGES_FILE}: {exc}. Using defaults.")
    else:
        log(f"{TEST_MESSAGES_FILE} not found. Using default test SMS messages.")
    return DEFAULT_TEST_MESSAGES

def main():
    logger = FileLogger(OUTPUT_FILE)
    log = logger.log

    test_messages = load_test_messages(log)

    log("Initializing RAG system...")
    rag = PhishingRAG(logger=log)
    log(f"Active analysis model: {rag.model_manager.active_model_name}")
    
    log("Initializing evaluator...")
    evaluator = RAGEvaluator(rag, logger=log)
    log(f"Active evaluator model: {evaluator.model_manager.active_model_name}")
    
    log("="*60)
    log("EVALUATING SMS MESSAGES")
    log("="*60)
    
    results = evaluator.evaluate_batch(test_messages)
    
    log("="*60)
    log("EVALUATION COMPLETE")
    log(f"Analyzed {len(results)} messages")
    log(f"Run details written to {logger.path}")
    log("="*60)

if __name__ == "__main__":
    main()