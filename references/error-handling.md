# エラー処理とリトライ リファレンス

## 目次
1. [概要](#概要)
2. [リトライポリシー](#リトライポリシー)
3. [ノードレベルのエラーハンドリング](#ノードレベルのエラーハンドリング)
4. [グラフレベルのエラーハンドリング](#グラフレベルのエラーハンドリング)
5. [フォールバック戦略](#フォールバック戦略)
6. [検証と修復](#検証と修復)
7. [実装例](#実装例)
8. [ベストプラクティス](#ベストプラクティス)

---

## 概要

LangGraphは多層的なエラーハンドリングを提供:

| レベル | 対象 | 手法 |
|--------|------|------|
| ノードレベル | 個別ノードの失敗 | try/except，RetryPolicy |
| グラフレベル | ワークフロー全体 | 条件分岐，フォールバックパス |
| アプリレベル | システム全体 | サーキットブレーカー，レート制限 |

---

## リトライポリシー

### RetryPolicyの設定

```python
from langgraph.pregel import RetryPolicy

# 基本的なリトライポリシー
retry_policy = RetryPolicy(
    max_attempts=3,              # 最大試行回数
    initial_interval=1.0,        # 初回リトライまでの待機秒数
    backoff_multiplier=2.0,      # 指数バックオフ倍率
    max_interval=10.0,           # 最大待機秒数
    retry_on=(ValueError, TimeoutError)  # リトライ対象の例外
)
```

### ノードへの適用

```python
from langgraph.graph import StateGraph

builder = StateGraph(AgentState)

# ノード追加時にリトライポリシーを指定
builder.add_node(
    "api_call",
    call_external_api,
    retry_policy=RetryPolicy(
        max_attempts=5,
        initial_interval=0.5,
        backoff_multiplier=2.0
    )
)
```

### 例外タイプ別のリトライ

```python
from langgraph.pregel import RetryPolicy

# 特定の例外のみリトライ
api_retry = RetryPolicy(
    max_attempts=3,
    retry_on=(
        ConnectionError,
        TimeoutError,
        RateLimitError
    )
)

# 全ての例外をリトライ（デフォルト）
general_retry = RetryPolicy(
    max_attempts=2,
    retry_on=Exception
)
```

### ジッター（ランダム遅延）

```python
import random

class JitteredRetryPolicy(RetryPolicy):
    def get_interval(self, attempt: int) -> float:
        base = super().get_interval(attempt)
        jitter = random.uniform(0, base * 0.1)  # 10%のジッター
        return base + jitter
```

---

## ノードレベルのエラーハンドリング

### try/exceptパターン

```python
def safe_node(state: AgentState) -> dict:
    """エラーをキャッチして状態に記録"""
    try:
        result = risky_operation(state["input"])
        return {
            "result": result,
            "error": None,
            "status": "success"
        }
    except ValueError as e:
        return {
            "result": None,
            "error": {"type": "validation", "message": str(e)},
            "status": "error"
        }
    except TimeoutError as e:
        return {
            "result": None,
            "error": {"type": "timeout", "message": str(e)},
            "status": "error"
        }
    except Exception as e:
        return {
            "result": None,
            "error": {"type": "unknown", "message": str(e)},
            "status": "error"
        }
```

### 型付きエラーオブジェクト

```python
from pydantic import BaseModel
from typing import Literal

class NodeError(BaseModel):
    type: Literal["validation", "api", "timeout", "unknown"]
    message: str
    node_name: str
    recoverable: bool = True

class AgentState(TypedDict):
    messages: list
    result: str | None
    errors: Annotated[list[NodeError], operator.add]

def node_with_typed_errors(state: AgentState) -> dict:
    try:
        result = process(state)
        return {"result": result}
    except ValidationError as e:
        return {
            "errors": [NodeError(
                type="validation",
                message=str(e),
                node_name="process_node",
                recoverable=True
            )]
        }
```

---

## グラフレベルのエラーハンドリング

### 条件分岐によるエラールーティング

```python
def route_by_status(state: AgentState) -> str:
    """状態に基づいてルーティング"""
    if state.get("errors"):
        last_error = state["errors"][-1]
        if last_error.recoverable:
            return "retry_handler"
        return "error_handler"
    return "next_step"

builder.add_conditional_edges(
    "risky_node",
    route_by_status,
    {
        "retry_handler": "retry_handler",
        "error_handler": "error_handler",
        "next_step": "next_step"
    }
)
```

### エラーハンドラーノード

```python
def error_handler_node(state: AgentState) -> dict:
    """エラーを処理してユーザーにフィードバック"""
    errors = state.get("errors", [])

    if not errors:
        return {}

    # エラーメッセージを生成
    error_summary = "\n".join([
        f"- {e.node_name}: {e.message}"
        for e in errors
    ])

    return {
        "messages": [AIMessage(
            content=f"処理中にエラーが発生しました:\n{error_summary}\n別の方法を試みます。"
        )],
        "errors": []  # エラーをクリア
    }
```

### リトライハンドラー

```python
class AgentState(TypedDict):
    messages: list
    retry_count: int
    max_retries: int

def retry_handler_node(state: AgentState) -> dict:
    """リトライ可能なエラーを処理"""
    current_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if current_count >= max_retries:
        return {
            "status": "failed",
            "messages": [AIMessage(content="最大リトライ回数に達しました")]
        }

    return {
        "retry_count": current_count + 1,
        "status": "retrying"
    }

def route_retry(state: AgentState) -> str:
    if state["status"] == "retrying":
        return "original_node"  # 元のノードに戻る
    return "final_error_handler"
```

---

## フォールバック戦略

### 代替パスへのルーティング

```python
def route_with_fallback(state: AgentState) -> str:
    """プライマリが失敗したらフォールバックへ"""
    if state.get("primary_failed"):
        return "fallback_node"
    return "primary_node"

builder.add_conditional_edges(
    START,
    route_with_fallback,
    {"primary_node": "primary", "fallback_node": "fallback"}
)
```

### 段階的フォールバック

```python
class AgentState(TypedDict):
    messages: list
    fallback_level: int  # 0: primary, 1: secondary, 2: tertiary

def primary_node(state: AgentState) -> dict:
    try:
        result = call_primary_api(state)
        return {"result": result}
    except Exception:
        return {"fallback_level": 1}

def secondary_node(state: AgentState) -> dict:
    try:
        result = call_secondary_api(state)
        return {"result": result}
    except Exception:
        return {"fallback_level": 2}

def tertiary_node(state: AgentState) -> dict:
    # 最終フォールバック（失敗しない）
    return {"result": generate_fallback_response(state)}

def route_by_fallback_level(state: AgentState) -> str:
    level = state.get("fallback_level", 0)
    if level == 0:
        return "primary"
    elif level == 1:
        return "secondary"
    else:
        return "tertiary"
```

### LLMモデルのフォールバック

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableWithFallbacks

# プライマリモデル
primary_model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

# フォールバックモデル
fallback_model = ChatGoogleGenerativeAI(model="gemini-3-flash")

# フォールバック付きモデル
model_with_fallback = primary_model.with_fallbacks([fallback_model])

def call_model_node(state: AgentState) -> dict:
    # 自動的にフォールバック
    response = model_with_fallback.invoke(state["messages"])
    return {"messages": [response]}
```

---

## 検証と修復

### 出力検証

```python
from pydantic import BaseModel, ValidationError

class ExpectedOutput(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

def validate_and_repair_node(state: AgentState) -> dict:
    """LLM出力を検証し，必要に応じて修復"""
    raw_output = state["raw_output"]

    try:
        validated = ExpectedOutput.model_validate(raw_output)
        return {"validated_output": validated.model_dump()}
    except ValidationError as e:
        # 修復を試みる
        repair_prompt = f"""
        以下の出力を修正してください:
        {raw_output}

        エラー: {e}

        期待されるフォーマット:
        - answer: 文字列
        - confidence: 0.0-1.0の数値
        - sources: ソースのリスト
        """

        repaired = model.invoke(repair_prompt)
        return {"validated_output": parse_repaired(repaired)}
```

### 再プロンプトによる修復

```python
def llm_with_validation(state: AgentState) -> dict:
    """検証失敗時に再プロンプト"""
    max_attempts = 3

    for attempt in range(max_attempts):
        response = model.invoke(state["messages"])

        # 検証
        is_valid, errors = validate_response(response.content)

        if is_valid:
            return {"messages": [response]}

        # 修正を要求
        state["messages"].append(AIMessage(content=response.content))
        state["messages"].append(HumanMessage(
            content=f"以下のエラーを修正してください:\n{errors}"
        ))

    # 最大試行後も失敗
    return {
        "messages": [AIMessage(content="検証に失敗しました")],
        "status": "validation_failed"
    }
```

---

## 実装例

### 完全なエラーハンドリングフロー

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.pregel import RetryPolicy
import operator

class RobustAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    errors: Annotated[list[dict], operator.add]
    retry_count: int
    status: Literal["processing", "success", "error", "retrying"]

def main_processing_node(state: RobustAgentState) -> dict:
    """メイン処理ノード"""
    try:
        result = call_external_service(state)
        return {
            "messages": [AIMessage(content=result)],
            "status": "success"
        }
    except RateLimitError as e:
        return {
            "errors": [{"type": "rate_limit", "message": str(e), "recoverable": True}],
            "status": "error"
        }
    except AuthenticationError as e:
        return {
            "errors": [{"type": "auth", "message": str(e), "recoverable": False}],
            "status": "error"
        }
    except Exception as e:
        return {
            "errors": [{"type": "unknown", "message": str(e), "recoverable": True}],
            "status": "error"
        }

def error_router(state: RobustAgentState) -> str:
    """エラー状態に基づいてルーティング"""
    if state["status"] == "success":
        return "success"

    errors = state.get("errors", [])
    if not errors:
        return "success"

    last_error = errors[-1]

    if not last_error["recoverable"]:
        return "fatal_error"

    retry_count = state.get("retry_count", 0)
    if retry_count >= 3:
        return "max_retries_exceeded"

    return "retry"

def retry_node(state: RobustAgentState) -> dict:
    """リトライ準備"""
    import time
    retry_count = state.get("retry_count", 0)

    # 指数バックオフ
    wait_time = min(2 ** retry_count, 30)
    time.sleep(wait_time)

    return {
        "retry_count": retry_count + 1,
        "status": "retrying",
        "errors": []  # エラーをクリア
    }

def fatal_error_node(state: RobustAgentState) -> dict:
    """回復不能なエラー"""
    return {
        "messages": [AIMessage(
            content="申し訳ございません。回復不能なエラーが発生しました。"
        )],
        "status": "error"
    }

def max_retries_node(state: RobustAgentState) -> dict:
    """最大リトライ超過"""
    return {
        "messages": [AIMessage(
            content="複数回の試行後も処理を完了できませんでした。"
        )],
        "status": "error"
    }

def success_node(state: RobustAgentState) -> dict:
    """成功時の後処理"""
    return {"status": "success"}

# グラフ構築
builder = StateGraph(RobustAgentState)

# ノード追加（リトライポリシー付き）
builder.add_node(
    "main_process",
    main_processing_node,
    retry_policy=RetryPolicy(max_attempts=2, initial_interval=0.5)
)
builder.add_node("retry", retry_node)
builder.add_node("fatal_error", fatal_error_node)
builder.add_node("max_retries", max_retries_node)
builder.add_node("success", success_node)

# エッジ定義
builder.add_edge(START, "main_process")
builder.add_conditional_edges(
    "main_process",
    error_router,
    {
        "success": "success",
        "retry": "retry",
        "fatal_error": "fatal_error",
        "max_retries_exceeded": "max_retries"
    }
)
builder.add_edge("retry", "main_process")
builder.add_edge("fatal_error", END)
builder.add_edge("max_retries", END)
builder.add_edge("success", END)

graph = builder.compile()
```

### ツール呼び出しのエラーハンドリング

```python
def safe_tool_executor(state: AgentState) -> dict:
    """ツール呼び出しを安全に実行"""
    last_message = state["messages"][-1]
    results = []

    for tool_call in last_message.tool_calls:
        try:
            tool_fn = tool_map.get(tool_call["name"])

            if not tool_fn:
                results.append(ToolMessage(
                    content=f"Unknown tool: {tool_call['name']}",
                    tool_call_id=tool_call["id"]
                ))
                continue

            result = tool_fn.invoke(tool_call["args"])
            results.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            ))

        except TimeoutError:
            results.append(ToolMessage(
                content="Tool execution timed out. Please try again.",
                tool_call_id=tool_call["id"]
            ))

        except Exception as e:
            results.append(ToolMessage(
                content=f"Tool error: {str(e)}. Please try a different approach.",
                tool_call_id=tool_call["id"]
            ))

    return {"messages": results}
```

---

## ベストプラクティス

### 1. 多層防御

```python
# ノードレベル: 個別エラー処理
def node_with_error_handling(state):
    try:
        # 処理
        pass
    except SpecificError:
        # 特定エラーの処理
        pass

# グラフレベル: フォールバックパス
builder.add_conditional_edges("node", error_router, paths)

# アプリレベル: グローバルエラーハンドリング
try:
    result = graph.invoke(input)
except GraphExecutionError as e:
    handle_system_error(e)
```

### 2. エラーの分類

```python
# 回復可能 vs 回復不能
class RecoverableError(Exception):
    """リトライで解決可能"""
    pass

class FatalError(Exception):
    """回復不能，処理を中止"""
    pass
```

### 3. 適切なリトライ設定

```python
# API呼び出し: 短い間隔，多めのリトライ
api_retry = RetryPolicy(
    max_attempts=5,
    initial_interval=0.5,
    backoff_multiplier=2.0,
    max_interval=30.0
)

# LLM呼び出し: 長い間隔，少なめのリトライ
llm_retry = RetryPolicy(
    max_attempts=2,
    initial_interval=2.0,
    backoff_multiplier=2.0
)
```

### 4. ログとモニタリング

```python
import logging

logger = logging.getLogger(__name__)

def monitored_node(state: AgentState) -> dict:
    logger.info(f"Node started: {state.get('step_name')}")

    try:
        result = process(state)
        logger.info(f"Node completed successfully")
        return {"result": result}

    except Exception as e:
        logger.error(f"Node failed: {e}", exc_info=True)
        raise
```

### 5. グレースフルデグラデーション

```python
def node_with_graceful_fallback(state: AgentState) -> dict:
    """失敗しても何かを返す"""
    try:
        return {"result": full_processing(state)}
    except Exception:
        try:
            return {"result": simplified_processing(state)}
        except Exception:
            return {"result": default_response(state)}
```

---

## 参考リンク

- [Error Handling in LangGraph](https://deepwiki.com/langchain-ai/langgraph/3.7-error-handling-and-retry-policies)
- [Retry Policies Guide](https://dev.to/aiengineering/a-beginners-guide-to-handling-errors-in-langgraph-with-retry-policies-h22)
- [Advanced Error Handling Strategies](https://sparkco.ai/blog/advanced-error-handling-strategies-in-langgraph-applications)
