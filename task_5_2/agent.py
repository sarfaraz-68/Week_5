import os
import json
import ast
import operator

from google import genai


def search(query):
    query = query.lower().strip()

    if "france" in query and "population" in query:
        return "France has a population of about 68 million."

    if "germany" in query and "population" in query:
        return "Germany has a population of about 84 million."

    return "NO_INFORMATION_FOUND"


def calculator(expression):
    try:
        expression = expression.strip()

        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }

        def calculate(node):
            if isinstance(node, ast.Expression):
                return calculate(node.body)

            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError()

            if isinstance(node, ast.BinOp):
                left = calculate(node.left)
                right = calculate(node.right)

                operation = allowed_operators.get(type(node.op))

                if operation is None:
                    raise ValueError()

                return operation(left, right)

            raise ValueError()

        tree = ast.parse(expression, mode="eval")
        result = calculate(tree)

        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return str(result)

    except Exception:
        return "INVALID_CALCULATION"


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()

        if not content.strip():
            return "FILE_EMPTY"

        return content

    except Exception:
        return "FILE_COULD_NOT_BE_READ"


def run_tool(tool_name, tool_input):

    if tool_name == "search":
        return search(tool_input)

    if tool_name == "calculator":
        return calculator(tool_input)

    if tool_name == "read_file":
        return read_file(tool_input)

    return "UNKNOWN_TOOL"


gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def save_run(goal, log, final_answer=None, stopped=False):

    with open(
        "agent_run.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(f"Goal: {goal}\n\n")

        for item in log:

            file.write(
                f"Step {item['step']}\n"
            )

            file.write(
                f"Reason: {item['reason']}\n"
            )

            file.write(
                f"Act: {item['tool']}('{item['input']}')\n"
            )

            file.write(
                f"Observe: {item['observe']}\n\n"
            )

        if final_answer:

            file.write(
                f"Final Answer: {final_answer}\n"
            )

        if stopped:

            file.write(
                "Agent stopped: Maximum step limit reached.\n"
            )


def agent(goal):

    history = []
    log = []

    max_steps = 5

    for step in range(max_steps):

        prompt = f"""
You are a STRICT tool-based AI agent.

User goal:
{goal}

You have ONLY these tools:

1. search(query)
2. calculator(expression)
3. read_file(path)

IMPORTANT STRICT RULES:

- You MUST NOT use your own general knowledge to answer the user's question.
- You MUST NOT invent facts.
- You MUST NOT guess.
- Every factual answer MUST come from a tool result.
- If search returns NO_INFORMATION_FOUND, the requested information is NOT available.
- If the required information is not available from the tools, return a final refusal.
- Do not repeatedly search for the same unavailable information.
- Use calculator only when the required numbers have already been obtained from tool results.
- You may combine information from multiple tool results.
- You may use previous successful tool results.
- Never treat your own knowledge as a tool result.

Available previous tool results:

{json.dumps(history, indent=2)}

DECISION RULE:

If more information is required:

Return ONLY JSON:

{{
    "type": "tool",
    "tool": "search",
    "input": "France population"
}}

OR:

{{
    "type": "tool",
    "tool": "calculator",
    "input": "68000000 + 84000000"
}}

OR:

{{
    "type": "tool",
    "tool": "read_file",
    "input": "example.txt"
}}

If the goal can be answered ONLY using information returned by the tools:

Return:

{{
    "type": "final",
    "answer": "..."
}}

If the requested information is NOT available from the tools:

Return:

{{
    "type": "refuse",
    "answer": "I don't have that information available through my tools."
}}

Do NOT provide any factual answer unless the supporting information exists in the tool results.

Return ONLY valid JSON.
"""


        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        raw_response = response.text.strip()

        try:
            decision = json.loads(raw_response)

        except json.JSONDecodeError:

            print("\nGemini returned an invalid response.")

            print("\nRaw response:")
            print(raw_response)

            return None

        print(f"\nStep {step + 1}")

        print("Reason:", decision)

        decision_type = decision.get("type")

        if decision_type == "final":

            answer = decision.get(
                "answer",
                "No answer was provided."
            )

            print("\nFinal Answer:")
            print(answer)

            save_run(
                goal,
                log,
                final_answer=answer
            )

            print("\nSaved: agent_run.txt")

            return answer

        if decision_type == "refuse":

            answer = decision.get(
                "answer",
                "I don't have that information available through my tools."
            )

            print("\nFinal Answer:")
            print(answer)

            save_run(
                goal,
                log,
                final_answer=answer
            )

            print("\nSaved: agent_run.txt")

            return answer

        if decision_type != "tool":

            print("\nInvalid decision type.")

            return None

        tool_name = decision.get("tool")
        tool_input = decision.get("input")

        if tool_name not in {
            "search",
            "calculator",
            "read_file"
        }:

            print("\nUnknown tool requested.")

            return None

        if not isinstance(tool_input, str):

            print("\nInvalid tool input.")

            return None

        print(
            "Act:",
            f"{tool_name}('{tool_input}')"
        )

        result = run_tool(
            tool_name,
            tool_input
        )

        print("Observe:", result)

        log.append(
            {
                "step": step + 1,
                "reason": decision,
                "tool": tool_name,
                "input": tool_input,
                "observe": result
            }
        )

        history.append(
            {
                "tool": tool_name,
                "input": tool_input,
                "result": result
            }
        )

        if result == "NO_INFORMATION_FOUND":

            history.append(
                {
                    "system_note":
                    "This information is unavailable. "
                    "Do not answer from general knowledge."
                }
            )

    print("\nAgent stopped:")
    print("Maximum step limit reached.")

    save_run(
        goal,
        log,
        stopped=True
    )

    return None


if __name__ == "__main__":

    goal = input(
        "Enter your goal: "
    )

    if not goal.strip():
    
        print(
            "Please enter a goal."
        )

    else:

        agent(goal)

