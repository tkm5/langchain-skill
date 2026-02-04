# Map-Reduce パターン リファレンス

## 目次
1. [概要](#概要)
2. [Send API](#send-api)
3. [基本的なMap-Reduce](#基本的なmap-reduce)
4. [Deferred Execution（遅延実行）](#deferred-execution遅延実行)
5. [並行処理の制御](#並行処理の制御)
6. [エラーハンドリング](#エラーハンドリング)
7. [実装例](#実装例)
8. [ベストプラクティス](#ベストプラクティス)

---

## 概要

Map-Reduceは，タスクを複数のサブタスクに分割し，並列処理した後に結果を集約するパターン．

### 主な課題と解決策

1. サブタスクの数がグラフ設計時に未知
2. 各サブタスクが異なる入力状態を必要とする

LangGraphは`Send` APIでこれらを解決．

---

## Send API

### 基本構文

```python
from langgraph.types import Send

def route_to_workers(state: AgentState) -> list[Send]:
    """動的に複数のノードインスタンスを起動"""
    sends = []

    for task in state["tasks"]:
        sends.append(Send(
            "worker",  # 送信先ノード名
            {"task": task, "context": state["context"]}  # 送信する状態
        ))

    return sends
```

### Sendの特徴

- 実行時にタスク数を決定可能
- 各タスクに異なる状態を送信可能
- 送信先は同じノードでも異なるノードでも可
- 結果は自動的に集約される

---

## 基本的なMap-Reduce

### 完全な実装例

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import operator

class MainState(TypedDict):
    topics: list[str]  # 処理するトピックのリスト
    results: Annotated[list[str], operator.add]  # 結果を集約

class WorkerState(TypedDict):
    topic: str
    result: str

# Mapフェーズ: トピックを分散
def distribute_topics(state: MainState) -> list[Send]:
    """各トピックをワーカーに送信"""
    return [
        Send("process_topic", {"topic": topic})
        for topic in state["topics"]
    ]

# 並列処理ノード
def process_topic(state: WorkerState) -> dict:
    """各トピックを処理"""
    result = f"{state['topic']}の分析結果"
    return {"result": result}

# Reduceフェーズ: 結果を集約
def aggregate_results(state: MainState) -> dict:
    """全結果を集約して最終出力を生成"""
    combined = "\n".join(state["results"])
    return {"final_output": f"集約結果:\n{combined}"}

# グラフ構築
builder = StateGraph(MainState)

# ノード追加
builder.add_node("process_topic", process_topic)
builder.add_node("aggregate", aggregate_results)

# エッジ定義
builder.add_conditional_edges(
    START,
    distribute_topics,
    ["process_topic"]  # 動的に複数インスタンスが生成される
)
builder.add_edge("process_topic", "aggregate")
builder.add_edge("aggregate", END)

graph = builder.compile()

# 実行
result = graph.invoke({
    "topics": ["AI", "機械学習", "深層学習"],
    "results": []
})
```

### 結果の集約（Reducer）

```python
import operator
from typing import Annotated

class MainState(TypedDict):
    # operator.addで自動的にリストを結合
    results: Annotated[list[dict], operator.add]

# カスタムReducer
def merge_results(existing: list, new: list) -> list:
    """重複を除去してマージ"""
    existing_ids = {r["id"] for r in existing}
    return existing + [r for r in new if r["id"] not in existing_ids]

class StateWithCustomReducer(TypedDict):
    results: Annotated[list[dict], merge_results]
```

---

## Deferred Execution（遅延実行）

並列ブランチの完了を待機するパターン．

### deferを使用した同期

```python
from langgraph.types import Send
from langgraph.constants import DEFER

def route_to_parallel_agents(state: MainState) -> list[Send]:
    """複数エージェントを並列起動し，全完了を待機"""
    sends = []

    # 各エージェントにタスクを送信
    sends.append(Send("agent_a", {"query": state["query"]}))
    sends.append(Send("agent_b", {"query": state["query"]}))

    # DEFERを追加して全エージェントの完了を待機
    sends.append(Send(DEFER, None))

    return sends

# 全エージェントの結果を受け取るノード
def combine_agent_results(state: MainState) -> dict:
    """全エージェントの結果を統合"""
    # この時点で agent_a と agent_b の結果が state に含まれる
    return {
        "combined_response": merge_responses(
            state["agent_a_result"],
            state["agent_b_result"]
        )
    }
```

### 非対称な完了時間への対処

```python
def route_with_defer(state: MainState) -> list[Send]:
    """処理時間が異なるタスクを並列実行"""
    sends = [
        Send("fast_task", state),    # 速く完了
        Send("medium_task", state),  # 中程度
        Send("slow_task", state),    # 遅い
        Send(DEFER, None)            # 全て完了まで待機
    ]
    return sends
```

---

## 並行処理の制御

### max_concurrency

```python
# 同時実行数を制限
result = graph.invoke(
    {"topics": large_topic_list},
    config={"max_concurrency": 5}  # 最大5並列
)
```

### バッチ処理

```python
def batch_distribute(state: MainState) -> list[Send]:
    """バッチサイズで分割して送信"""
    batch_size = 10
    items = state["items"]

    sends = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        sends.append(Send("process_batch", {"batch": batch, "batch_id": i}))

    return sends
```

---

## エラーハンドリング

### Superstep内のエラー

```python
# 並列ノードの1つが失敗すると，superstep全体が失敗
# Checkpointerを使用している場合，成功したノードの結果は保存される

def safe_worker(state: WorkerState) -> dict:
    """エラーハンドリング付きワーカー"""
    try:
        result = risky_operation(state["task"])
        return {"result": result, "status": "success"}
    except Exception as e:
        # エラーを状態として返す（グラフは継続）
        return {"result": None, "status": "error", "error": str(e)}
```

### リトライ付きワーカー

```python
from langgraph.graph import RetryPolicy

# ノードにリトライポリシーを適用
builder.add_node(
    "worker",
    worker_function,
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        backoff_multiplier=2.0
    )
)
```

### フォールバックパス

```python
def route_with_fallback(state: MainState) -> str:
    """エラー時にフォールバックパスへ"""
    if any(r["status"] == "error" for r in state["results"]):
        return "error_handler"
    return "aggregate"

builder.add_conditional_edges(
    "worker",
    route_with_fallback,
    {"error_handler": "error_handler", "aggregate": "aggregate"}
)
```

---

## 実装例

### ドキュメント並列分析

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import operator

class AnalysisState(TypedDict):
    documents: list[str]
    analyses: Annotated[list[dict], operator.add]
    summary: str

class DocumentState(TypedDict):
    document: str
    doc_id: int

def distribute_documents(state: AnalysisState) -> list[Send]:
    """ドキュメントを分析ノードに配布"""
    return [
        Send("analyze_document", {"document": doc, "doc_id": i})
        for i, doc in enumerate(state["documents"])
    ]

def analyze_document(state: DocumentState) -> dict:
    """個別ドキュメントを分析"""
    analysis = model.invoke([
        SystemMessage(content="以下のドキュメントを分析してください"),
        HumanMessage(content=state["document"])
    ])

    return {
        "analyses": [{
            "doc_id": state["doc_id"],
            "analysis": analysis.content
        }]
    }

def generate_summary(state: AnalysisState) -> dict:
    """全分析結果からサマリー生成"""
    all_analyses = "\n".join([
        f"Doc {a['doc_id']}: {a['analysis']}"
        for a in sorted(state["analyses"], key=lambda x: x["doc_id"])
    ])

    summary = model.invoke([
        SystemMessage(content="以下の分析結果を要約してください"),
        HumanMessage(content=all_analyses)
    ])

    return {"summary": summary.content}

# グラフ構築
builder = StateGraph(AnalysisState)
builder.add_node("analyze_document", analyze_document)
builder.add_node("summarize", generate_summary)

builder.add_conditional_edges(START, distribute_documents, ["analyze_document"])
builder.add_edge("analyze_document", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()
```

### マルチエージェント並列実行

```python
class MultiAgentState(TypedDict):
    query: str
    agent_responses: Annotated[list[dict], operator.add]
    final_answer: str

def route_to_agents(state: MultiAgentState) -> list[Send]:
    """複数の専門エージェントに並列でクエリを送信"""
    agents = ["researcher", "analyst", "critic"]

    return [
        Send(agent, {"query": state["query"], "agent_name": agent})
        for agent in agents
    ]

def researcher_agent(state: dict) -> dict:
    """調査エージェント"""
    response = research_model.invoke(state["query"])
    return {"agent_responses": [{"agent": "researcher", "response": response}]}

def analyst_agent(state: dict) -> dict:
    """分析エージェント"""
    response = analysis_model.invoke(state["query"])
    return {"agent_responses": [{"agent": "analyst", "response": response}]}

def critic_agent(state: dict) -> dict:
    """批評エージェント"""
    response = critic_model.invoke(state["query"])
    return {"agent_responses": [{"agent": "critic", "response": response}]}

def synthesize_responses(state: MultiAgentState) -> dict:
    """全エージェントの回答を統合"""
    combined = "\n".join([
        f"{r['agent']}: {r['response']}"
        for r in state["agent_responses"]
    ])

    final = synthesizer_model.invoke(f"統合してください:\n{combined}")
    return {"final_answer": final}

# グラフ構築
builder = StateGraph(MultiAgentState)
builder.add_node("researcher", researcher_agent)
builder.add_node("analyst", analyst_agent)
builder.add_node("critic", critic_agent)
builder.add_node("synthesize", synthesize_responses)

builder.add_conditional_edges(START, route_to_agents, ["researcher", "analyst", "critic"])
builder.add_edge("researcher", "synthesize")
builder.add_edge("analyst", "synthesize")
builder.add_edge("critic", "synthesize")
builder.add_edge("synthesize", END)

graph = builder.compile()
```

---

## ベストプラクティス

### 1. 軽量なReducer関数

```python
# Good: シンプルで効率的
results: Annotated[list, operator.add]

# Avoid: 複雑な処理をReducerで行わない
def heavy_reducer(existing, new):
    # 重い処理はノードで行う
    return existing + new
```

### 2. 適切な並行数の設定

```python
# リソースに応じて調整
config = {
    "max_concurrency": min(len(tasks), 10)  # 最大10並列
}
```

### 3. 状態の最小化

```python
# Good: 必要な情報のみ送信
Send("worker", {"task_id": task["id"], "input": task["input"]})

# Bad: 不要な情報を含む
Send("worker", state)  # 全状態を送信
```

### 4. べき等性の確保

```python
def idempotent_worker(state: WorkerState) -> dict:
    """同じ入力で同じ結果を返す"""
    # 副作用を避けるか，べき等な操作を使用
    result = pure_function(state["input"])
    return {"result": result}
```

### 5. 進捗の可視化

```python
from langgraph.types import StreamWriter

def worker_with_progress(state: WorkerState, writer: StreamWriter) -> dict:
    writer({
        "event": "task_started",
        "task_id": state["task_id"],
        "total": state["total_tasks"]
    })

    result = process(state)

    writer({
        "event": "task_completed",
        "task_id": state["task_id"]
    })

    return {"result": result}
```

---

## 参考リンク

- [How to create map-reduce branches](https://langchain-ai.github.io/langgraphjs/how-tos/map-reduce/)
- [Send API for parallel execution](https://dev.to/sreeni5018/leveraging-langgraphs-send-api-for-dynamic-and-parallel-workflow-execution-4pgd)
- [Parallel Nodes with Deferred Execution](https://medium.com/@gmurro/parallel-nodes-in-langgraph-managing-concurrent-branches-with-the-deferred-execution-d7e94d03ef78)
