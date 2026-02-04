# ストリーミング リファレンス

## 目次
1. [概要](#概要)
2. [ストリーミングモード](#ストリーミングモード)
3. [LLMトークンのストリーミング](#llmトークンのストリーミング)
4. [カスタムストリーミング](#カスタムストリーミング)
5. [サブグラフからのストリーミング](#サブグラフからのストリーミング)
6. [非同期ストリーミング](#非同期ストリーミング)
7. [ストリーミングの制御](#ストリーミングの制御)
8. [実装例](#実装例)

---

## 概要

LangGraphのストリーミングは，グラフ実行中にリアルタイムでデータを取得する機能．3つの主要なカテゴリのデータをストリーミング可能:

1. ワークフロー進捗: 各ノード実行後の状態更新
2. LLMトークン: LLMが生成するトークン
3. カスタム更新: ユーザー定義のイベントやデータ

---

## ストリーミングモード

### 利用可能なモード

| モード | 説明 | ユースケース |
|--------|------|-------------|
| `values` | 各ステップ後の完全な状態 | デバッグ，状態追跡 |
| `updates` | 各ステップでの状態変更のみ | 効率的な差分取得 |
| `messages` | LLMトークン | チャットUI |
| `custom` | カスタムイベント | 進捗表示，ログ |
| `events` | 全イベント（低レベル） | 詳細なトレース |

### valuesモード

```python
# 各ステップ後の完全な状態を取得
for chunk in graph.stream(
    {"messages": [HumanMessage(content="こんにちは")]},
    stream_mode="values"
):
    print(chunk)
    # 出力: {"messages": [...], "other_state": ...}
```

### updatesモード

```python
# 変更された部分のみ取得
for chunk in graph.stream(
    {"messages": [HumanMessage(content="質問")]},
    stream_mode="updates"
):
    print(chunk)
    # 出力: {"node_name": {"messages": [新しいメッセージのみ]}}
```

### 複数モードの同時使用

```python
# 複数のモードを同時に使用
for chunk in graph.stream(
    {"messages": [...]},
    stream_mode=["values", "messages"]
):
    if chunk[0] == "values":
        print("状態:", chunk[1])
    elif chunk[0] == "messages":
        print("トークン:", chunk[1])
```

---

## LLMトークンのストリーミング

### messagesモード

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END

model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

graph = builder.compile()

# LLMトークンをストリーミング
for chunk in graph.stream(
    {"messages": [HumanMessage(content="長い説明をしてください")]},
    stream_mode="messages"
):
    # AIMessageChunkが返される
    if hasattr(chunk, 'content'):
        print(chunk.content, end="", flush=True)
```

### astream_eventsを使用（詳細な制御）

```python
async for event in graph.astream_events(
    {"messages": [HumanMessage(content="質問")]},
    version="v2"
):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        print(chunk.content, end="", flush=True)
```

---

## カスタムストリーミング

### StreamWriterの使用

```python
from langgraph.types import StreamWriter

def processing_node(state: AgentState, writer: StreamWriter):
    """カスタムイベントをストリーム"""

    # 進捗を通知
    writer({"event": "started", "message": "処理を開始します"})

    # 段階的な処理
    for i, step in enumerate(processing_steps):
        result = process_step(step)
        writer({
            "event": "progress",
            "step": i + 1,
            "total": len(processing_steps),
            "message": f"ステップ {i + 1} 完了"
        })

    writer({"event": "completed", "message": "処理が完了しました"})

    return {"result": final_result}

# グラフ定義時にwriterパラメータを持つノードを追加
builder.add_node("process", processing_node)
```

### customモードでの受信

```python
for chunk in graph.stream(
    {"input": data},
    stream_mode="custom"
):
    if chunk.get("event") == "progress":
        progress = chunk["step"] / chunk["total"] * 100
        print(f"進捗: {progress:.1f}%")
```

### 複数ストリームの組み合わせ

```python
# LLMトークンとカスタムイベントを同時に取得
for mode, chunk in graph.stream(
    {"messages": [...]},
    stream_mode=["messages", "custom"]
):
    if mode == "messages":
        print(chunk.content, end="")
    elif mode == "custom":
        if chunk.get("event") == "tool_started":
            print(f"\n[ツール実行中: {chunk['tool']}]")
```

---

## サブグラフからのストリーミング

### subgraphs=True

```python
# サブグラフからもストリーミングを有効化
for chunk in parent_graph.stream(
    {"input": data},
    subgraphs=True
):
    print(chunk)
```

### サブグラフのLLMトークン

```python
# サブグラフのLLMトークンを取得
for chunk in parent_graph.stream(
    {"messages": [...]},
    stream_mode="messages",
    subgraphs=True
):
    # 親グラフとサブグラフ両方からのトークンが含まれる
    if hasattr(chunk, 'content'):
        print(chunk.content, end="")
```

### StreamWriterでサブグラフのトークンを変換

```python
def subgraph_wrapper_node(state: ParentState, writer: StreamWriter):
    """サブグラフのトークンをカスタムイベントとして転送"""

    # サブグラフをストリーミング実行
    for chunk in subgraph.stream(
        {"query": state["input"]},
        stream_mode="messages"
    ):
        # カスタムイベントとして転送
        writer({
            "event": "subgraph_token",
            "content": chunk.content if hasattr(chunk, 'content') else str(chunk)
        })

    return {"output": final_result}
```

---

## 非同期ストリーミング

### astream

```python
async def stream_response():
    async for chunk in graph.astream(
        {"messages": [HumanMessage(content="質問")]},
        stream_mode="messages"
    ):
        yield chunk.content
```

### astream_events

```python
async def detailed_stream():
    async for event in graph.astream_events(
        {"messages": [...]},
        version="v2"
    ):
        event_type = event["event"]

        if event_type == "on_chain_start":
            print(f"チェーン開始: {event['name']}")

        elif event_type == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            print(content, end="")

        elif event_type == "on_tool_start":
            print(f"\nツール開始: {event['name']}")

        elif event_type == "on_chain_end":
            print(f"\nチェーン終了: {event['name']}")
```

### 非同期Checkpointerとの組み合わせ

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def run_with_streaming():
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "thread-1"}}

        async for chunk in graph.astream(
            {"messages": [...]},
            config,
            stream_mode="messages"
        ):
            print(chunk.content, end="")
```

---

## ストリーミングの制御

### ストリーミングを無効化

```python
# モデルレベルでストリーミングを無効化
model = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview",
    streaming=False  # ストリーミング無効
)
```

### 特定ノードのみストリーミング

```python
def node_with_streaming(state: AgentState, writer: StreamWriter):
    # このノードはストリーミングを使用
    writer({"status": "processing"})
    result = model.invoke(state["messages"])
    return {"messages": [result]}

def node_without_streaming(state: AgentState):
    # このノードはストリーミングなし
    result = process_silently(state)
    return {"result": result}
```

### ストリーム出力のバッファリング

```python
import asyncio

async def buffered_stream():
    buffer = []
    buffer_size = 5

    async for chunk in graph.astream(
        {"messages": [...]},
        stream_mode="messages"
    ):
        buffer.append(chunk.content)

        if len(buffer) >= buffer_size:
            # バッファがいっぱいになったら出力
            print("".join(buffer), end="", flush=True)
            buffer = []

    # 残りを出力
    if buffer:
        print("".join(buffer))
```

---

## 実装例

### FastAPIでのストリーミング

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(message: str):
    async def generate():
        async for chunk in graph.astream(
            {"messages": [HumanMessage(content=message)]},
            stream_mode="messages"
        ):
            if hasattr(chunk, 'content') and chunk.content:
                yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

### 進捗付きの長時間処理

```python
from langgraph.types import StreamWriter

def long_running_task(state: AgentState, writer: StreamWriter):
    total_steps = 10

    for i in range(total_steps):
        # 各ステップの処理
        result = process_step(i, state)

        # 進捗を通知
        writer({
            "type": "progress",
            "current": i + 1,
            "total": total_steps,
            "percentage": (i + 1) / total_steps * 100,
            "message": f"ステップ {i + 1}/{total_steps} を処理中..."
        })

    return {"result": final_result}

# クライアント側
for mode, chunk in graph.stream(
    {"input": data},
    stream_mode=["updates", "custom"]
):
    if mode == "custom" and chunk.get("type") == "progress":
        print(f"\r進捗: {chunk['percentage']:.0f}%", end="")
    elif mode == "updates":
        print(f"\n状態更新: {chunk}")
```

### マルチエージェントのストリーミング

```python
def multi_agent_coordinator(state: CoordinatorState, writer: StreamWriter):
    """複数エージェントを調整し，各エージェントの出力をストリーム"""

    for agent_name, agent_graph in agents.items():
        writer({
            "event": "agent_started",
            "agent": agent_name
        })

        # 各エージェントの出力をストリーム
        for chunk in agent_graph.stream(
            {"query": state["query"]},
            stream_mode="messages"
        ):
            writer({
                "event": "agent_output",
                "agent": agent_name,
                "content": chunk.content if hasattr(chunk, 'content') else ""
            })

        writer({
            "event": "agent_completed",
            "agent": agent_name
        })

    return {"responses": collected_responses}
```

---

## トラブルシューティング

### 問題: サブグラフからトークンがストリームされない

```python
# 解決策: subgraphs=True を必ず指定
for chunk in graph.stream(..., subgraphs=True):
    ...
```

### 問題: 非同期ストリーミングで出力がない

```python
# 解決策: astream を使用し，適切に await
async for chunk in graph.astream(...):  # stream ではなく astream
    print(chunk)
```

### 問題: カスタムイベントが受信できない

```python
# 解決策: stream_mode に "custom" を含める
for chunk in graph.stream(..., stream_mode=["messages", "custom"]):
    ...
```

---

## 参考リンク

- [LangGraph Streaming Concepts](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/concepts/streaming.md)
- [How to stream LLM tokens](https://langchain-ai.lang.chat/langgraphjs/how-tos/stream-tokens/)
- [How to stream custom data](https://langchain-ai.lang.chat/langgraphjs/how-tos/streaming-content/)
