# Human-in-the-Loop リファレンス

## 目次
1. [概要](#概要)
2. [interrupt関数](#interrupt関数)
3. [Commandによる再開](#commandによる再開)
4. [デザインパターン](#デザインパターン)
5. [interrupt_beforeとinterrupt_after](#interrupt_beforeとinterrupt_after)
6. [実装例](#実装例)
7. [ベストプラクティス](#ベストプラクティス)

---

## 概要

Human-in-the-Loop（HITL）は，AIエージェントの実行を特定のポイントで一時停止し，人間の介入を待つ仕組み．LangGraph 0.2.31以降では`interrupt`関数が推奨される方法．

### 主なユースケース

- 承認/拒否: 重要な操作（API呼び出し，データ変更）の前に人間の承認を得る
- 状態の編集: グラフの状態を確認・修正する
- ツール呼び出しのレビュー: LLMが要求したツール呼び出しを実行前に確認
- 追加情報の収集: 処理に必要な情報を人間から取得

---

## interrupt関数

### 基本的な使い方

```python
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver

def human_review_node(state: MessagesState):
    """人間のレビューを要求するノード"""

    # interruptで実行を一時停止
    # 引数は人間に表示する情報
    response = interrupt({
        "question": "この操作を承認しますか？",
        "proposed_action": state["messages"][-1].content,
        "options": ["approve", "reject", "edit"]
    })

    # 人間からの応答を処理
    if response["decision"] == "approve":
        return {"messages": [AIMessage(content="承認されました")]}
    elif response["decision"] == "reject":
        return {"messages": [AIMessage(content="拒否されました")]}
    else:
        return {"messages": [AIMessage(content=f"編集内容: {response['edit']}")]}

# グラフ構築
builder = StateGraph(MessagesState)
builder.add_node("process", process_node)
builder.add_node("human_review", human_review_node)
builder.add_node("execute", execute_node)

builder.add_edge(START, "process")
builder.add_edge("process", "human_review")
builder.add_edge("human_review", "execute")
builder.add_edge("execute", END)

# Checkpointerは必須（中断状態を保存するため）
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

### interruptの動作

1. `interrupt()`が呼ばれると，グラフ実行が停止
2. スレッドは「interrupted」状態としてマーク
3. 引数で渡した情報が永続化レイヤーに保存
4. `Command(resume=...)`で再開するまで待機

---

## Commandによる再開

### 基本的な再開

```python
from langgraph.types import Command

# 最初の実行（interruptで停止）
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke({"messages": [HumanMessage(content="処理を開始")]}, config)

# 人間がレビューした後，Commandで再開
graph.invoke(
    Command(resume={"decision": "approve"}),
    config
)
```

### 状態の更新を伴う再開

```python
# 状態を更新しながら再開
graph.invoke(
    Command(
        resume={"decision": "edit", "edit": "修正された内容"},
        update={"additional_info": "追加情報"}  # 状態を更新
    ),
    config
)
```

### 別のノードにジャンプ

```python
# 特定のノードにジャンプして再開
graph.invoke(
    Command(
        resume={"decision": "reject"},
        goto="error_handler"  # error_handlerノードにジャンプ
    ),
    config
)
```

---

## デザインパターン

### パターン1: 承認/拒否

```python
def approval_node(state: AgentState):
    """重要なアクションの前に承認を求める"""

    # 実行予定のアクションを表示
    action = state["pending_action"]

    approval = interrupt({
        "type": "approval_required",
        "action": action,
        "message": f"次のアクションを実行しますか？: {action['name']}"
    })

    if approval["approved"]:
        return {"status": "approved"}
    else:
        return {"status": "rejected", "reason": approval.get("reason", "")}

def route_after_approval(state: AgentState) -> str:
    if state["status"] == "approved":
        return "execute_action"
    return "handle_rejection"
```

### パターン2: 状態の編集

```python
def edit_state_node(state: AgentState):
    """状態を人間に編集させる"""

    edited_state = interrupt({
        "type": "edit_request",
        "current_state": {
            "draft": state["draft"],
            "metadata": state["metadata"]
        },
        "instructions": "必要に応じて内容を編集してください"
    })

    return {
        "draft": edited_state.get("draft", state["draft"]),
        "metadata": edited_state.get("metadata", state["metadata"]),
        "human_edited": True
    }
```

### パターン3: ツール呼び出しのレビュー

```python
def review_tool_calls_node(state: MessagesState):
    """LLMが要求したツール呼び出しを確認"""

    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return {"messages": []}

    review_result = interrupt({
        "type": "tool_review",
        "tool_calls": [
            {
                "name": tc["name"],
                "args": tc["args"],
                "id": tc["id"]
            }
            for tc in last_message.tool_calls
        ],
        "message": "以下のツール呼び出しを確認してください"
    })

    approved_calls = review_result.get("approved", [])
    rejected_calls = review_result.get("rejected", [])

    # 承認されたツールのみ実行
    results = []
    for tc in last_message.tool_calls:
        if tc["id"] in approved_calls:
            result = execute_tool(tc)
            results.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        elif tc["id"] in rejected_calls:
            results.append(ToolMessage(
                content="ユーザーにより拒否されました",
                tool_call_id=tc["id"]
            ))

    return {"messages": results}
```

### パターン4: 情報の収集

```python
def collect_info_node(state: AgentState):
    """追加情報を人間から収集"""

    user_input = interrupt({
        "type": "info_request",
        "question": state["clarification_needed"],
        "context": state["current_context"]
    })

    return {
        "user_response": user_input["response"],
        "clarification_needed": None  # クリア
    }
```

---

## interrupt_beforeとinterrupt_after

`interrupt`関数の代替として，グラフコンパイル時に設定する方法．

### interrupt_before

```python
from langgraph.prebuilt import create_react_agent

# ツール実行前に自動的に停止
agent = create_react_agent(
    model,
    tools,
    interrupt_before=["tools"]  # toolsノード実行前に停止
)

# 実行
config = {"configurable": {"thread_id": "user-123"}}
result = agent.invoke({"messages": [...]}, config)

# 中断後，Noneで継続
result = agent.invoke(None, config)
```

### interrupt_after

```python
# 特定ノード実行後に停止
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_after=["analysis"]  # analysisノード後に停止
)
```

### 複数ノードの指定

```python
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["critical_action", "external_api"],
    interrupt_after=["review_required"]
)
```

---

## 実装例

### 完全な承認ワークフロー

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class ApprovalState(TypedDict):
    messages: Annotated[list, add_messages]
    pending_action: dict | None
    approval_status: Literal["pending", "approved", "rejected"] | None

def plan_action(state: ApprovalState):
    """アクションを計画"""
    messages = state["messages"]
    # LLMでアクションを生成
    response = model.invoke(messages)

    return {
        "messages": [response],
        "pending_action": extract_action(response),
        "approval_status": "pending"
    }

def request_approval(state: ApprovalState):
    """人間に承認を要求"""
    action = state["pending_action"]

    result = interrupt({
        "action_type": action["type"],
        "description": action["description"],
        "parameters": action["params"],
        "risk_level": action.get("risk", "medium")
    })

    return {"approval_status": result["decision"]}

def execute_action(state: ApprovalState):
    """承認されたアクションを実行"""
    action = state["pending_action"]
    result = perform_action(action)

    return {
        "messages": [AIMessage(content=f"実行完了: {result}")],
        "pending_action": None
    }

def handle_rejection(state: ApprovalState):
    """拒否された場合の処理"""
    return {
        "messages": [AIMessage(content="アクションは拒否されました。別の方法を検討します。")],
        "pending_action": None
    }

def route_by_approval(state: ApprovalState) -> str:
    if state["approval_status"] == "approved":
        return "execute"
    return "rejected"

# グラフ構築
builder = StateGraph(ApprovalState)
builder.add_node("plan", plan_action)
builder.add_node("approval", request_approval)
builder.add_node("execute", execute_action)
builder.add_node("rejected", handle_rejection)

builder.add_edge(START, "plan")
builder.add_edge("plan", "approval")
builder.add_conditional_edges("approval", route_by_approval, {
    "execute": "execute",
    "rejected": "rejected"
})
builder.add_edge("execute", END)
builder.add_edge("rejected", END)

# コンパイル
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
```

### 使用例

```python
# 初回実行
config = {"configurable": {"thread_id": "approval-flow-1"}}
result = graph.invoke(
    {"messages": [HumanMessage(content="重要なファイルを削除してください")]},
    config
)

# この時点でinterruptにより停止
# result には interrupt の情報が含まれる

# 人間が承認
final_result = graph.invoke(
    Command(resume={"decision": "approved"}),
    config
)

# または拒否
final_result = graph.invoke(
    Command(resume={"decision": "rejected"}),
    config
)
```

---

## ベストプラクティス

### 1. Checkpointerは必須

```python
# Good: Checkpointer設定あり
graph = builder.compile(checkpointer=checkpointer)

# Bad: Checkpointerなし（interruptが機能しない）
graph = builder.compile()  # エラー
```

### 2. interruptに十分な情報を含める

```python
# Good: 判断に必要な情報を全て含める
interrupt({
    "type": "approval",
    "action": "delete_file",
    "target": "/important/data.csv",
    "reason": "ユーザーリクエスト",
    "risk_level": "high",
    "reversible": False
})

# Bad: 情報不足
interrupt({"action": "delete"})
```

### 3. タイムアウトの考慮

```python
# 長時間の中断を想定した設計
def request_approval_with_timeout(state: ApprovalState):
    result = interrupt({
        "action": state["pending_action"],
        "timeout_hours": 24,
        "default_on_timeout": "reject"
    })

    # アプリケーション側でタイムアウトを処理
    return {"approval_status": result.get("decision", "timeout")}
```

### 4. 再開時のバリデーション

```python
def validated_approval_node(state: ApprovalState):
    result = interrupt({"action": state["pending_action"]})

    # 入力値を検証
    valid_decisions = ["approved", "rejected", "edit"]
    if result.get("decision") not in valid_decisions:
        # 無効な入力を再度要求するか，デフォルト値を使用
        return {"approval_status": "rejected", "error": "無効な入力"}

    return {"approval_status": result["decision"]}
```

### 5. エラーハンドリング

```python
def safe_interrupt_node(state: AgentState):
    try:
        result = interrupt({"question": "確認してください"})
        return process_result(result)
    except Exception as e:
        # interruptでエラーが発生した場合のフォールバック
        return {"error": str(e), "status": "failed"}
```

---

## 参考リンク

- [Human-in-the-loop Concepts](https://langchain-ai.github.io/langgraphjs/concepts/human_in_the_loop/)
- [Wait for user input using interrupt](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/)
- [interrupt Announcement](https://changelog.langchain.com/announcements/interrupt-simplifying-human-in-the-loop-agents)
