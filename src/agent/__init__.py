"""Agent runtime public helpers.

Keep imports lazy so lightweight API modules can import ``src.agent.llm``
without compiling the LangGraph pipeline or loading graph-only dependencies.
"""


def __getattr__(name: str):
    if name in {"run_pipeline", "get_graph", "build_graph"}:
        from src.agent import graph

        return getattr(graph, name)
    raise AttributeError(f"module 'src.agent' has no attribute {name!r}")

__all__ = ["run_pipeline", "get_graph", "build_graph"]
