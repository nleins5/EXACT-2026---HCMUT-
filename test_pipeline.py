"""Test LangGraph pipeline với fine-tuned models."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.agent.graph import build_graph
from src.agent.state import AgentState
from src.agent.llm.factory import LLMFactory
from src.agent.llm.server_supervisor import LlamaServerSupervisor
from src.utils.logger import logger


def test_pipeline():
    """Test toàn bộ pipeline với fine-tuned models."""
    
    # 1. Initialize supervisor và factory
    logger.info("Initializing LlamaServerSupervisor...")
    supervisor = LlamaServerSupervisor()
    LLMFactory.init(supervisor)
    
    # 2. Build graph
    logger.info("Building LangGraph pipeline...")
    graph = build_graph()
    
    # 3. Test Logic Problem (Type 1)
    logger.info("=" * 70)
    logger.info("TEST 1: Logic Problem (Type 1)")
    logger.info("=" * 70)
    
    initial_state: AgentState = {
        "question": "All birds can fly. Penguins are birds. Are penguins able to fly?",
        "premises": ["All birds can fly", "Penguins are birds"],
        "task_type": "logic",
        "intermediate_answer": {
            "context_rag": "",
            "context_code": "",
            "generated_code": "",
            "code_output": "",
            "code_error": False,
            "error_message": "",
            "reasoning": "",
            "final_output": "",
        },
        "final_answer": {
            "answer": "",
            "explanation": "",
            "fol": "",
            "cot": [],
            "premises": [],
            "confidence": 0.0,
        },
        "error": "",
        "collection_name": "logic_regulations",
        "context": "",
    }
    
    try:
        result = graph.invoke(initial_state)
        logger.info(f"Logic Result: {result['final_answer']}")
    except Exception as e:
        logger.error(f"Logic test failed: {e}")
    
    # 4. Test Physics Problem (Type 2)
    logger.info("=" * 70)
    logger.info("TEST 2: Physics Problem (Type 2)")
    logger.info("=" * 70)
    
    initial_state: AgentState = {
        "question": "R1=30Ω, R2=60Ω parallel. Find R_eq.",
        "premises": [],
        "task_type": "physics",
        "intermediate_answer": {
            "context_rag": "",
            "context_code": "",
            "generated_code": "",
            "code_output": "",
            "code_error": False,
            "error_message": "",
            "reasoning": "",
            "final_output": "",
        },
        "final_answer": {
            "answer": "",
            "explanation": "",
            "fol": "",
            "cot": [],
            "premises": [],
            "confidence": 0.0,
        },
        "error": "",
        "collection_name": "physics_examples",
        "context": "",
    }
    
    try:
        result = graph.invoke(initial_state)
        logger.info(f"Physics Result: {result['final_answer']}")
    except Exception as e:
        logger.error(f"Physics test failed: {e}")
    
    # 5. Cleanup
    logger.info("Shutting down supervisor...")
    supervisor.shutdown()
    logger.info("Test completed!")


if __name__ == "__main__":
    test_pipeline()
