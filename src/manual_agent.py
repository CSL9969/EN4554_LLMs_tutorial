"""A manually-implemented ReAct-style agent with two tools: RAG search and a calculator.
"""

import ast
import operator
import re

from openai import OpenAI

from rag_retriever import VectorStore

MODEL_NAME = "Qwen/Qwen3.5-2B"
BASE_URL = "http://localhost:8000/v1"
DOCS_DIR = "/home/chamindu/EN4554_LLMs_tutorial/docs"
MAX_ITERATIONS = 5

SYSTEM_PROMPT = """You are an assistant that can use tools to answer questions.

Available tools:
- search_documents(query): Searches a knowledge base and returns relevant passages.
The knowledge base contains knowledge about Attention, LLMs and Generative AI.
- calculator(expression): Evaluates a mathematical expression and returns the result.

To use a tool, respond in EXACTLY this format and then stop:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <input to the tool>

Once you receive an Observation with the tool's result, continue reasoning in the
same format, using more tool calls if needed. When you have enough information,
respond with:
Thought: <your reasoning>
Final Answer: <the answer to the user's question>

Rules:
- Only ever emit ONE Action per turn, then stop and wait for its Observation.
- Never write your own "Observation:" line — that is provided to you.
- If a question needs no tools, just reason and give a Final Answer directly.
"""


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
    """Safely evaluates a arithmetic expression (+, -, *, /, **, %, parentheses)."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
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

ACTION_RE = re.compile(r"Action:\s*(.+)")
ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.+)", re.DOTALL)
FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


class Agent:
    """A manual ReAct loop supporting a search_documents tool and a calculator tool."""

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

    def _call_llm(self, messages: list) -> str:
        """Calls the LLM, stopping before it can hallucinate an Observation."""
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=512,
            stop=["Observation:"],
        )
        return completion.choices[0].message.content or ""

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Dispatches to the right tool implementation."""
        tool_name = tool_name.strip().lower()
        tool_input = tool_input.strip()

        if tool_name == "search_documents":
            return search_documents(self.vectorstore, tool_input)
        if tool_name == "calculator":
            return calculator(tool_input)
        return f"Error: unknown tool '{tool_name}'."

    def run(self, question: str, verbose: bool = True) -> str:
        """Runs the ReAct loop until a Final Answer is produced or iterations run out.

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
            reply = self._call_llm(messages)
            if verbose:
                print(reply.strip())

            final_match = FINAL_ANSWER_RE.search(reply)
            if final_match:
                return final_match.group(1).strip()

            action_match = ACTION_RE.search(reply)
            input_match = ACTION_INPUT_RE.search(reply)

            if not action_match or not input_match:
                # Model didn't follow the format — treat its reply as the answer.
                return reply.strip()

            tool_name = action_match.group(1).strip()
            tool_input = input_match.group(1).strip().splitlines()[0]  # first line only

            observation = self._execute_tool(tool_name, tool_input)
            if verbose:
                print(f"Observation: {observation}\n")

            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return "Agent stopped: reached max iterations without a Final Answer."


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