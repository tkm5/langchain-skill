---
name: langgraph-guide
description: |
  LangGraph/LangChainを使ったAIエージェント実装のガイド．StateGraph，Checkpointer，Memory Store，ReActエージェントの実装方法を提供．
  使用タイミング：(1) LangGraphでグラフベースのエージェントを構築する時，(2) PostgresSaver/AsyncPostgresSaverでメモリを永続化する時，(3) Gemini/Google GenAIとLangChainを統合する時，(4) ReActエージェントを実装する時，(5) 短期・長期メモリを実装する時，(6) サブグラフでモジュール化する時，(7) Human-in-the-Loopを実装する時，(8) ストリーミングを実装する時，(9) Map-Reduceで並列処理する時，(10) エラーハンドリングとリトライを実装する時
---

# LangGraph/LangChain 実装ガイド

## バージョン情報（2025年1月時点）

- langgraph: 0.2.60+
- langgraph-checkpoint-postgres: 2.0.0+
- langchain-google-genai: 4.1.2（google-genai SDKベース）
- langchain-core: 0.3.0+

## クイックスタート

### 依存関係

```bash
pip install langgraph langchain-google-genai langgraph-checkpoint-postgres psycopg[binary]
```

### 基本的なStateGraph

```python
from langgraph.graph import StateGraph, MessagesState, START, END

def call_model(state: MessagesState):
    # ノード処理
    return {"messages": response}

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

graph = builder.compile()
```

## 主要コンポーネント

### 1. State定義

TypedDictまたはPydantic BaseModelで状態を定義：

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages
import operator

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # メッセージ履歴
    custom_field: Annotated[list, operator.add]  # リスト連結
```

### 2. Nodes（ノード）

現在のStateを受け取り，更新されたStateを返す関数：

```python
def my_node(state: AgentState, config: RunnableConfig) -> dict:
    # 処理
    return {"messages": new_message}
```

### 3. Edges（エッジ）

- 固定エッジ: `builder.add_edge("node_a", "node_b")`
- 条件付きエッジ: `builder.add_conditional_edges("node_a", route_function, {"case1": "node_b", "case2": "node_c"})`

## メモリ機能

詳細は [references/memory.md](references/memory.md) を参照．

### 短期メモリ（Checkpointer）

スレッドレベルの会話状態を永続化：

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:pass@localhost:5432/db"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 初回のみ
    graph = builder.compile(checkpointer=checkpointer)
```

### 長期メモリ（Store）

ユーザー固有のデータをセッション間で保存：

```python
from langgraph.store.postgres import PostgresStore

store = PostgresStore.from_conn_string(DB_URI)
store.setup()
graph = builder.compile(store=store)
```

## Gemini統合

詳細は [references/gemini.md](references/gemini.md) を参照．

### ChatGoogleGenerativeAI

```python
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview",
    api_key=os.getenv("GEMINI_API_KEY"),
)
```

### ツールバインディング

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """検索を実行"""
    return results

model_with_tools = model.bind_tools([search])
```

## ReActエージェント

詳細は [references/react.md](references/react.md) を参照．

### 基本構造

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools, state_modifier=system_prompt)
```

### カスタム実装

Thought → Action → Observation のサイクル：

```python
def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "call_tool"
    return "end"

builder.add_conditional_edges("call_model", should_continue, {"call_tool": "tools", "end": END})
```

## サブグラフ

詳細は [references/subgraph.md](references/subgraph.md) を参照．

複雑なワークフローをモジュール化:

```python
# サブグラフを親グラフのノードとして追加
parent_builder.add_node("subgraph", compiled_subgraph)

# 状態変換が必要な場合はラッパー関数を使用
def call_subgraph(state: ParentState) -> dict:
    child_input = transform_to_child(state)
    result = subgraph.invoke(child_input)
    return transform_to_parent(result)
```

## Human-in-the-Loop

詳細は [references/human-in-the-loop.md](references/human-in-the-loop.md) を参照．

`interrupt`関数で人間の介入を待機:

```python
from langgraph.types import interrupt, Command

def approval_node(state: AgentState):
    result = interrupt({
        "question": "この操作を承認しますか？",
        "action": state["pending_action"]
    })
    return {"approved": result["decision"] == "approve"}

# 再開時
graph.invoke(Command(resume={"decision": "approve"}), config)
```

## ストリーミング

詳細は [references/streaming.md](references/streaming.md) を参照．

リアルタイムでデータを取得:

```python
# LLMトークンをストリーミング
for chunk in graph.stream(input, stream_mode="messages"):
    print(chunk.content, end="")

# サブグラフからもストリーミング
for chunk in graph.stream(input, subgraphs=True):
    print(chunk)
```

## Map-Reduceパターン

詳細は [references/map-reduce.md](references/map-reduce.md) を参照．

`Send` APIで動的に並列タスクを生成:

```python
from langgraph.types import Send

def distribute_tasks(state: MainState) -> list[Send]:
    return [
        Send("worker", {"task": task})
        for task in state["tasks"]
    ]

builder.add_conditional_edges(START, distribute_tasks, ["worker"])
```

## エラーハンドリング

詳細は [references/error-handling.md](references/error-handling.md) を参照．

リトライポリシーとフォールバック:

```python
from langgraph.pregel import RetryPolicy

builder.add_node(
    "api_call",
    call_api,
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        backoff_multiplier=2.0
    )
)
```

## 重要な注意点

1. Checkpointer設定: `autocommit=True`と`row_factory=dict_row`が必要
2. setup()呼び出し: Checkpointer/Store初回使用時に必須
3. config_schema非推奨: v0.6.0から`context_schema`を使用
4. google-genai SDK: langchain-google-genai 4.0.0以降は統合SDKを使用
5. interrupt使用時: Checkpointerは必須（中断状態を保存するため）
6. Map-Reduce: Reducerは軽量に保ち，重い処理はノードで行う

## リファレンス一覧

| トピック | ファイル | 説明 |
|----------|----------|------|
| Gemini統合 | [references/gemini.md](references/gemini.md) | ChatGoogleGenerativeAI，ツールバインディング，埋め込み |
| ReActエージェント | [references/react.md](references/react.md) | プリビルト/カスタムReActエージェント |
| メモリ機能 | [references/memory.md](references/memory.md) | Checkpointer，Store，TTL，セマンティック検索 |
| サブグラフ | [references/subgraph.md](references/subgraph.md) | ネストグラフ，状態変換，モジュール化 |
| Human-in-the-Loop | [references/human-in-the-loop.md](references/human-in-the-loop.md) | interrupt，Command，承認フロー |
| ストリーミング | [references/streaming.md](references/streaming.md) | トークンストリーム，カスタムイベント |
| Map-Reduce | [references/map-reduce.md](references/map-reduce.md) | Send API，並列処理，Deferred Execution |
| エラーハンドリング | [references/error-handling.md](references/error-handling.md) | RetryPolicy，フォールバック，検証 |
