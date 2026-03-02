"""LLM tools — expose the code index and git history to the agent.

Each tool is registered on the shared ``agent`` instance via
``@agent.tool``.  Submodules are imported for their side effects
(decorator registration).

Import tools directly from their submodule::

    from rbtr.llm.tools.file import grep
    from rbtr.llm.tools.draft import add_draft_comment
"""

# Side-effect imports: each submodule registers @agent.tool decorators.
import rbtr.llm.tools.discussion
import rbtr.llm.tools.draft
import rbtr.llm.tools.file
import rbtr.llm.tools.git
import rbtr.llm.tools.index
import rbtr.llm.tools.notes  # noqa: F401
