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
    
    # 3. Test Logic Problem (Type 1 - Expected: Yes)
    logger.info("=" * 70)
    logger.info("TEST 1: Logic Problem (Expected: Yes)")
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
        logger.info(f"Logic Result (Expected: Yes): {result['final_answer']}")
    except Exception as e:
        logger.error(f"Logic test failed: {e}")
    
    # 3b. Test Logic Problem (Type 1 - Expected: No)
    logger.info("=" * 70)
    logger.info("TEST 2: Logic Problem (Expected: No)")
    logger.info("=" * 70)
    
    initial_state_no: AgentState = {
        "question": "Felix is a cat. All cats have whiskers. Does Felix not have whiskers?",
        "premises": ["All cats have whiskers", "Felix is a cat"],
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
        result = graph.invoke(initial_state_no)
        logger.info(f"Logic Result (Expected: No): {result['final_answer']}")
    except Exception as e:
        logger.error(f"Logic test failed: {e}")
        
    # 3c. Test Logic Problem (Type 1 - Expected: Unknown)
    logger.info("=" * 70)
    logger.info("TEST 3: Logic Problem (Expected: Unknown)")
    logger.info("=" * 70)
    
    initial_state_unk: AgentState = {
        "question": "Buddy is a dog. Some dogs are friendly. Is Buddy friendly?",
        "premises": ["Some dogs are friendly", "Buddy is a dog"],
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
        result = graph.invoke(initial_state_unk)
        logger.info(f"Logic Result (Expected: Unknown): {result['final_answer']}")
    except Exception as e:
        logger.error(f"Logic test failed: {e}")
    
    # 4. Test Physics Problem (Type 2 - Expected: 20Ω)
    logger.info("=" * 70)
    logger.info("TEST 4: Physics Problem (Expected: 20Ω)")
    logger.info("=" * 70)
    
    initial_state_phys: AgentState = {
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
        result = graph.invoke(initial_state_phys)
        logger.info(f"Physics Result (Expected: 20Ω): {result['final_answer']}")
    except Exception as e:
        logger.error(f"Physics test failed: {e}")
        
    # 4b. Test Physics Problem (Type 2 - Expected: 490J)
    logger.info("=" * 70)
    logger.info("TEST 5: Physics Problem (Expected: 490J)")
    logger.info("=" * 70)
    
    initial_state_gpe: AgentState = {
        "question": "Calculate the gravitational potential energy of a 5kg mass at a height of 10m. (g = 9.8)",
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
        result = graph.invoke(initial_state_gpe)
        logger.info(f"Physics Result (Expected: 490J): {result['final_answer']}")
    except Exception as e:
        logger.error(f"Physics test failed: {e}")
    
    # 5. Cleanup
    logger.info("Shutting down supervisor...")
    supervisor.shutdown()
    logger.info("Test completed!")


if __name__ == "__main__":
    test_pipeline()
