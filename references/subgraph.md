# サブグラフ リファレンス

## 目次
1. [概要](#概要)
2. [基本的なサブグラフ](#基本的なサブグラフ)
3. [状態変換](#状態変換)
4. [サブグラフからのストリーミング](#サブグラフからのストリーミング)
5. [サブグラフの状態管理](#サブグラフの状態管理)
6. [ベストプラクティス](#ベストプラクティス)

---

## 概要

サブグラフは，別のグラフのノードとして使用されるグラフ．複雑なワークフローをモジュール化し，再利用性を高める．

### ユースケース

- マルチエージェントシステム: 異なる専門エージェントを個別のグラフとして実装
- 複雑なワークフロー: タスクを論理的な単位に分割
- 再利用可能なコンポーネント: 共通処理をサブグラフ化

---

## 基本的なサブグラフ

### 方法1: コンパイル済みサブグラフをノードとして追加

親グラフとサブグラフが少なくとも1つのキーを共有する場合:

```python
from langgraph.graph import StateGraph, MessagesState, START, END

# サブグラフの定義
def subgraph_node(state: MessagesState):
    # サブグラフの処理
    return {"messages": [AIMessage(content="サブグラフからの応答")]}

subgraph_builder = StateGraph(MessagesState)
subgraph_builder.add_node("process", subgraph_node)
subgraph_builder.add_edge(START, "process")
subgraph_builder.add_edge("process", END)
subgraph = subgraph_builder.compile()

# 親グラフにサブグラフを追加
parent_builder = StateGraph(MessagesState)
parent_builder.add_node("subgraph", subgraph)  # コンパイル済みグラフを直接追加
parent_builder.add_edge(START, "subgraph")
parent_builder.add_edge("subgraph", END)

parent_graph = parent_builder.compile()
```

### 方法2: 関数内でサブグラフを呼び出し

異なる状態スキーマを持つ場合，状態変換が必要:

```python
def call_subgraph(state: ParentState) -> dict:
    # 親の状態をサブグラフの状態に変換
    subgraph_input = {"query": state["user_input"]}

    # サブグラフを呼び出し
    result = subgraph.invoke(subgraph_input)

    # 結果を親の状態形式に戻す
    return {"response": result["output"]}

parent_builder.add_node("call_subgraph", call_subgraph)
```

---

## 状態変換

親グラフとサブグラフの状態スキーマが異なる場合の対処法．

### 独立した状態スキーマ

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 親グラフの状態
class ParentState(TypedDict):
    user_input: str
    final_response: str

# サブグラフの状態
class ChildState(TypedDict):
    query: str
    analysis: str
    result: str

# サブグラフ
def analyze(state: ChildState):
    return {"analysis": f"分析: {state['query']}"}

def generate_result(state: ChildState):
    return {"result": f"結果: {state['analysis']}"}

child_builder = StateGraph(ChildState)
child_builder.add_node("analyze", analyze)
child_builder.add_node("generate", generate_result)
child_builder.add_edge(START, "analyze")
child_builder.add_edge("analyze", "generate")
child_builder.add_edge("generate", END)
child_graph = child_builder.compile()

# 状態変換を行うラッパー関数
def call_child_graph(state: ParentState) -> dict:
    # 入力変換: ParentState -> ChildState
    child_input = {"query": state["user_input"]}

    # サブグラフ実行
    child_result = child_graph.invoke(child_input)

    # 出力変換: ChildState -> ParentState
    return {"final_response": child_result["result"]}

# 親グラフ
parent_builder = StateGraph(ParentState)
parent_builder.add_node("process", call_child_graph)
parent_builder.add_edge(START, "process")
parent_builder.add_edge("process", END)
parent_graph = parent_builder.compile()
```

### 孫グラフ（ネストされたサブグラフ）

```python
# 孫グラフの状態
class GrandchildState(TypedDict):
    detail: str
    output: str

# 子グラフの状態
class ChildState(TypedDict):
    query: str
    result: str

def call_grandchild(state: ChildState) -> dict:
    # 孫グラフの入力に変換
    grandchild_input = {"detail": state["query"]}
    grandchild_result = grandchild_graph.invoke(grandchild_input)

    # 結果を子グラフの状態に変換
    return {"result": grandchild_result["output"]}
```

---

## サブグラフからのストリーミング

### subgraphs=True でストリーミング有効化

```python
# サブグラフを含むグラフをストリーミング
for chunk in parent_graph.stream(
    {"user_input": "質問内容"},
    subgraphs=True  # サブグラフからもストリーミング
):
    print(chunk)
```

### 非同期ストリーミング

```python
async for chunk in parent_graph.astream(
    {"user_input": "質問内容"},
    subgraphs=True
):
    print(chunk)
```

### StreamWriterでカスタムストリーミング

```python
from langgraph.types import StreamWriter

def subgraph_node(state: ChildState, writer: StreamWriter):
    # カスタムイベントをストリーム
    writer({"event": "processing", "data": "分析中..."})

    # 処理
    result = process(state)

    writer({"event": "completed", "data": result})
    return {"result": result}
```

---

## サブグラフの状態管理

### サブグラフの状態を表示・更新

```python
# サブグラフの状態を取得
config = {"configurable": {"thread_id": "thread-1"}}

# グラフ実行を中断した後
state = parent_graph.get_state(config)

# サブグラフの状態にアクセス
for task in state.tasks:
    if hasattr(task, 'state'):
        print(f"サブグラフ状態: {task.state}")
```

### 状態の重複を避ける

```python
from typing import Annotated
import operator

class ParentState(TypedDict):
    messages: Annotated[list, add_messages]  # add_messagesで重複防止

# サブグラフからの返り値が親の状態にマージされる際，
# reducerが重複を適切に処理
```

---

## ベストプラクティス

### 1. 明確な責務分離

```python
# Good: 単一責任のサブグラフ
research_graph = create_research_subgraph()  # 調査専門
analysis_graph = create_analysis_subgraph()  # 分析専門
writing_graph = create_writing_subgraph()    # 執筆専門

# Bad: 複数責任を持つ巨大なグラフ
do_everything_graph = create_monolith_graph()
```

### 2. 適切な状態変換

```python
# Good: 明示的な変換関数
def transform_to_child_state(parent: ParentState) -> ChildState:
    return {
        "query": parent["user_input"],
        "context": extract_context(parent["history"])
    }

def transform_to_parent_state(child: ChildState) -> dict:
    return {"response": child["result"]}
```

### 3. エラーハンドリング

```python
def call_subgraph_safely(state: ParentState) -> dict:
    try:
        result = subgraph.invoke(transform_input(state))
        return transform_output(result)
    except Exception as e:
        # エラー時のフォールバック
        return {"error": str(e), "response": "処理中にエラーが発生しました"}
```

### 4. Checkpointerの共有

```python
from langgraph.checkpoint.postgres import PostgresSaver

# 親子で同じCheckpointerを使用
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()

    # サブグラフもコンパイル時にcheckpointerを設定可能
    # ただし，通常は親グラフのcheckpointerが使用される
    parent_graph = parent_builder.compile(checkpointer=checkpointer)
```

### 5. 可視化の注意点

```python
# 関数内でサブグラフを呼び出す場合，
# 親グラフの可視化にはサブグラフが表示されない

# 可視化したい場合は，コンパイル済みグラフを直接追加
parent_builder.add_node("child", child_graph)  # 可視化可能

# vs

parent_builder.add_node("child", call_child_graph)  # 可視化不可
```

---

## 参考リンク

- [Subgraphs - LangChain Docs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)
- [How to transform subgraph state](https://langchain-ai.github.io/langgraphjs/how-tos/subgraph-transform-state/)
- [How to view and update state in subgraphs](https://langchain-ai.github.io/langgraphjs/how-tos/subgraphs-manage-state/)
