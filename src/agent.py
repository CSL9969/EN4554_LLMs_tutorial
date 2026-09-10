"""Agentic RAG + calculator using the openai module's native tool-calling API.
"""

import ast
import json
import operator

from openai import OpenAI

from rag_retriever import VectorStore

MODEL_NAME = "Qwen/Qwen3.5-2B"
BASE_URL = "http://localhost:8000/v1"
DOCS_DIR = "/home/chamindu/EN4554_LLMs_tutorial/docs"
MAX_ITERATIONS = 5

SYSTEM_PROMPT = """
    You are a helpful assistant. Use the search_documents tool to look things up
    in the knowledge base, where it contains knowledge about attention mechanism and llms
    and the calculator tool for any arithmetic.
    Only call a tool when it's actually needed.
"""

# --- Tool schemas (the JSON-Schema format the OpenAI API / vLLM expects) -------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Searches the knowledge base and returns relevant passages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluates a mathematical expression, e.g. '3 * (4 + 5)'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The arithmetic expression to evaluate."},
                },
                "required": ["expression"],
            },
        },
    },
]

# --- Tool implementations -----------------------------------------------------

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    """Recursively evaluates an AST node, allowing only basic arithmetic."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Safely evaluates an arithmetic expression (+, -, *, /, **, %, parentheses)."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as exc:  # noqa: BLE001
        return f"Error evaluating expression: {exc}"


def search_documents(vectorstore: VectorStore, query: str, k: int = 3) -> str:
    """Retrieves the top-k chunks for a query and formats them as one string."""
    try:
        results = vectorstore.retrieve(query, k=k)
    except Exception as exc:  # noqa: BLE001
        return f"Error searching documents: {exc}"

    if not results:
        return "No relevant documents found."

    formatted = []
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[{i}] (source: {source})\n{doc.page_content.strip()}")
    return "\n\n".join(formatted)


# --- Agent ---------------------------------------------------------------------

class Agent:
    """An agent that uses OpenAI-style native tool calling instead of manual parsing."""

    def __init__(self, vectorstore: VectorStore, model_name: str = MODEL_NAME, base_url: str = BASE_URL):
        """Initializes the Agent class.

        Args:
            vectorstore (VectorStore): A built VectorStore instance for RAG search.
            model_name (str): The model name as registered on the vLLM server.
            base_url (str): The vLLM OpenAI-compatible server URL.
        """
        self.vectorstore = vectorstore
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")

    def _execute_tool(self, name: str, arguments: dict) -> str:
        """Dispatches to the right tool implementation by name."""
        if name == "search_documents":
            return search_documents(self.vectorstore, arguments.get("query", ""))
        if name == "calculator":
            return calculator(arguments.get("expression", ""))
        return f"Error: unknown tool '{name}'."

    def run(self, question: str, verbose: bool = True) -> str:
        """Runs the tool-calling loop until the model returns a plain text answer.

        Args:
            question (str): The user's question.
            verbose (bool): Whether to print each step as it happens.

        Returns:
            str: The final answer.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        for _ in range(MAX_ITERATIONS):
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=TOOLS,
                max_tokens=512,
            )
            message = completion.choices[0].message

            if not message.tool_calls:
                return message.content or ""

            messages.append(message.model_dump())

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments or "{}")

                if verbose:
                    print(f"Tool call: {name}({arguments})")

                result = self._execute_tool(name, arguments)

                if verbose:
                    print(f"Tool result: {result}\n")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        return "Agent stopped: reached max iterations without a final answer."


if __name__ == "__main__":
    store = VectorStore()
    store.build_from_docs(DOCS_DIR)

    agent = Agent(vectorstore=store)

    while True:
        question = input("\nAsk something (or 'exit'): ")
        if question.strip().lower() in {"exit", "quit"}:
            break
        answer = agent.run(question)
        print(f"\nFinal Answer: {answer}")