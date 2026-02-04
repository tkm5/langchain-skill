# ReActエージェント リファレンス

## 目次
1. [ReActとは](#reactとは)
2. [プリビルトエージェント](#プリビルトエージェント)
3. [カスタム実装](#カスタム実装)
4. [高度なパターン](#高度なパターン)

## ReActとは

ReAct（Reasoning and Acting）は，LLMが「思考→行動→観察」のサイクルを反復するエージェント手法：

1. Thought: 情報収集の必要性を判断
2. Action: ツールを選択・実行
3. Observation: 結果を観察し次のステップを決定

## プリビルトエージェント

### create_react_agent

LangGraphの組み込みReActエージェント：

```python
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

# モデル
model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

# ツール
@tool
def search(query: str) -> str:
    """Web検索を実行"""
    return "検索結果"

@tool
def calculator(expression: str) -> str:
    """数式を計算"""
    return str(eval(expression))

tools = [search, calculator]

# システムプロンプト
system_prompt = """あなたは親切なアシスタントです。
ユーザーの質問に答えるために、必要に応じてツールを使用してください。"""

# エージェント作成
agent = create_react_agent(
    model,
    tools,
    state_modifier=system_prompt
)

# 実行
result = agent.invoke({"messages": [HumanMessage(content="東京の人口は？")]})
```

### Checkpointerとの統合

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
agent = create_react_agent(model, tools, checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user-123"}}
result = agent.invoke({"messages": [...]}, config)
```

## カスタム実装

### 基本構造

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def call_model(state: MessagesState):
    """LLMを呼び出してツール使用を判断"""
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def call_tool(state: MessagesState):
    """ツールを実行"""
    last_message = state["messages"][-1]
    results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # ツールを検索して実行
        tool_fn = tool_map[tool_name]
        result = tool_fn.invoke(tool_args)

        results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))

    return {"messages": results}

def should_continue(state: MessagesState) -> str:
    """ツール呼び出しが必要か判断"""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "call_tool"
    return "end"

# グラフ構築
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("call_tool", call_tool)

builder.add_edge(START, "call_model")
builder.add_conditional_edges(
    "call_model",
    should_continue,
    {"call_tool": "call_tool", "end": END}
)
builder.add_edge("call_tool", "call_model")

graph = builder.compile()
```

### ツールマップの作成

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Web検索を実行"""
    return "検索結果"

@tool
def calculator(expression: str) -> str:
    """数式を計算"""
    return str(eval(expression))

# ツール名から関数へのマッピング
tools = [search, calculator]
tool_map = {tool.name: tool for tool in tools}

# モデルにツールをバインド
model_with_tools = model.bind_tools(tools)
```

## 高度なパターン

### 最大ステップ数の制限

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    step_count: int

def call_model(state: AgentState):
    response = model_with_tools.invoke(state["messages"])
    return {
        "messages": [response],
        "step_count": state.get("step_count", 0) + 1
    }

def should_continue(state: AgentState) -> str:
    # 最大10ステップで終了
    if state.get("step_count", 0) >= 10:
        return "end"

    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "call_tool"
    return "end"
```

### 並列ツール実行

```python
import asyncio

async def call_tool(state: MessagesState):
    last_message = state["messages"][-1]

    async def execute_tool(tool_call):
        tool_fn = tool_map[tool_call["name"]]
        result = await tool_fn.ainvoke(tool_call["args"])
        return ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )

    # 並列実行
    results = await asyncio.gather(*[
        execute_tool(tc) for tc in last_message.tool_calls
    ])

    return {"messages": list(results)}
```

### 複数エージェントの並列実行

```python
from langgraph.types import Send

def route_to_agents(state):
    """複数のエージェントを並列起動"""
    sends = []

    if state.get("run_agent_a"):
        sends.append(Send("agent_a", {"messages": state["messages"]}))

    if state.get("run_agent_b"):
        sends.append(Send("agent_b", {"messages": state["messages"]}))

    return sends

builder.add_conditional_edges(
    "router",
    route_to_agents,
    ["agent_a", "agent_b"]
)
```

### 人間介入（Human-in-the-Loop）

```python
from langgraph.prebuilt import create_react_agent

# interrupt_beforeで人間の確認を要求
agent = create_react_agent(
    model,
    tools,
    interrupt_before=["call_tool"]  # ツール実行前に停止
)

# 実行
config = {"configurable": {"thread_id": "user-123"}}
result = agent.invoke({"messages": [...]}, config)

# 人間が確認後に再開
result = agent.invoke(None, config)  # Noneで継続
```

### エラーハンドリング

```python
def call_tool(state: MessagesState):
    last_message = state["messages"][-1]
    results = []

    for tool_call in last_message.tool_calls:
        try:
            tool_fn = tool_map.get(tool_call["name"])
            if not tool_fn:
                raise ValueError(f"Unknown tool: {tool_call['name']}")

            result = tool_fn.invoke(tool_call["args"])
            results.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            ))
        except Exception as e:
            results.append(ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=tool_call["id"]
            ))

    return {"messages": results}
```

### コンテキスト付きエージェント

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str  # RAGなどで取得したコンテキスト
    company_info: dict  # 会社情報などの追加データ

def call_model(state: AgentState):
    # コンテキストをシステムプロンプトに追加
    system_message = SystemMessage(content=f"""
    あなたは経営コンサルタントです。

    ## コンテキスト
    {state.get("context", "")}

    ## 会社情報
    {state.get("company_info", {})}
    """)

    messages = [system_message] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}
```
