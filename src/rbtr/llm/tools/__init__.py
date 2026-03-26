"""LLM tools — tool modules for the pydantic-ai agent.

Tool submodules are NOT imported eagerly. Registration happens
inside `agent.get_agent()` to defer heavy deps (`duckdb`,
`pyarrow`) until the first LLM task.
"""
