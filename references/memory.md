# LangGraph メモリ機能リファレンス

## 目次
1. [概要](#概要)
2. [短期メモリ（Checkpointer）](#短期メモリcheckpointer)
3. [長期メモリ（Store）](#長期メモリstore)
4. [Store API詳細](#store-api詳細)
5. [Namespace設計](#namespace設計)
6. [セマンティック検索](#セマンティック検索)
7. [TTL（有効期限）管理](#ttl有効期限管理)
8. [非同期実装](#非同期実装)
9. [メモリ管理](#メモリ管理)
10. [ベストプラクティス](#ベストプラクティス)

---

## 概要

LangGraphは2種類のメモリを提供：

| 種類 | スコープ | 用途 | 実装 |
|------|---------|------|------|
| 短期メモリ | スレッド単位 | マルチターン会話の追跡 | Checkpointer |
| 長期メモリ | クロススレッド | ユーザー固有データの永続化 | Store |

**短期メモリ**: `thread_id`でスコープされ，同一スレッド内の会話履歴を保持
**長期メモリ**: カスタム`namespace`でスコープされ，異なるスレッド間で共有可能

---

## 短期メモリ（Checkpointer）

スレッドレベルの会話状態を永続化．マルチターン会話の追跡に使用．

### InMemorySaver（開発用）

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 実行時にthread_idを指定
config = {"configurable": {"thread_id": "thread-123"}}
result = graph.invoke({"messages": [...]}, config)
```

### PostgresSaver（本番環境）

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5432/db?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 初回のみ必須
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "thread-123"}}
    result = graph.invoke({"messages": [...]}, config)
```

### 手動接続設定

```python
import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

# 重要: autocommit=True, row_factory=dict_row が必須
conn = psycopg.connect(
    DB_URI,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(conn)
checkpointer.setup()
```

---

## 長期メモリ（Store）

ユーザー固有のデータを複数セッション・スレッド間で保存．

### 短期 vs 長期メモリの違い

```python
# 短期メモリ: thread_idでスコープ
config = {"configurable": {"thread_id": "thread-123"}}

# 長期メモリ: user_idとnamespaceでスコープ
config = {"configurable": {"user_id": "user-456"}}
namespace = ("memories", "user-456")
```

### InMemoryStore（開発用）

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
graph = builder.compile(store=store)
```

**注意**: InMemoryStoreはプロセス終了時にデータが消失．開発・テスト用途のみ．

### PostgresStore（本番環境）

```python
from langgraph.store.postgres import PostgresStore

DB_URI = "postgresql://postgres:postgres@localhost:5432/db"

store = PostgresStore.from_conn_string(DB_URI)
store.setup()  # 初回のみ必須（pgvectorテーブル作成）
graph = builder.compile(store=store)
```

### CheckpointerとStoreの組み合わせ

```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    store = PostgresStore.from_conn_string(DB_URI)
    checkpointer.setup()
    store.setup()

    graph = builder.compile(
        checkpointer=checkpointer,  # 短期メモリ
        store=store                  # 長期メモリ
    )
```

---

## Store API詳細

### put / aput - データの保存

```python
store.put(
    namespace: tuple[str, ...],  # 階層的パス
    key: str,                     # 一意識別子
    value: dict[str, Any],        # 保存データ（辞書形式）
    index: Literal[False] | list[str] | None = None,  # 検索インデックス
    ttl: float | None = None      # 有効期限（分）
)

# 例
store.put(
    namespace=("users", "user-123"),
    key="preferences",
    value={"theme": "dark", "language": "ja"},
    ttl=60 * 24 * 7  # 7日間
)

# 非同期版
await store.aput(namespace, key, value)
```

### get / aget - データの取得

```python
item = store.get(
    namespace: tuple[str, ...],
    key: str
)

# 例
item = store.get(("users", "user-123"), "preferences")
if item:
    print(item.value)  # {"theme": "dark", "language": "ja"}
    print(item.key)    # "preferences"
    print(item.namespace)  # ("users", "user-123")
    print(item.created_at)
    print(item.updated_at)

# 非同期版
item = await store.aget(namespace, key)
```

### search / asearch - データの検索

```python
items = store.search(
    namespace_prefix: tuple[str, ...],  # 検索対象のnamespace
    query: str | None = None,           # セマンティック検索クエリ
    filter: dict | None = None,         # メタデータフィルタ
    limit: int = 10,                     # 最大結果数
    offset: int = 0                      # ページネーション
)

# キーワード検索なし（namespace内の全件取得）
items = store.search(("users", "user-123"))

# セマンティック検索
items = store.search(
    ("documents",),
    query="機械学習の医療応用",
    limit=5
)

# フィルタ付き検索
items = store.search(
    ("documents",),
    query="AI技術",
    filter={"type": "research_paper", "year": 2025},
    limit=10
)

# 非同期版
items = await store.asearch(namespace_prefix, query=query)
```

### delete / adelete - データの削除

```python
store.delete(
    namespace: tuple[str, ...],
    key: str
)

# 例
store.delete(("users", "user-123"), "preferences")

# 非同期版
await store.adelete(namespace, key)
```

### list_namespaces / alist_namespaces - Namespace一覧

```python
namespaces = store.list_namespaces(
    prefix: tuple[str, ...] | None = None,  # 接頭辞フィルタ
    suffix: tuple[str, ...] | None = None,  # 接尾辞フィルタ
    max_depth: int | None = None,           # 最大深度
    limit: int = 100,
    offset: int = 0
)

# 例: "users"で始まるnamespaceを取得
namespaces = store.list_namespaces(prefix=("users",))
# [("users", "user-123"), ("users", "user-456"), ...]

# 非同期版
namespaces = await store.alist_namespaces(prefix=prefix)
```

### batch / abatch - バッチ操作

```python
from langgraph.store.base import GetOp, PutOp, SearchOp

results = store.batch([
    GetOp(namespace=("users", "user-123"), key="prefs"),
    PutOp(namespace=("users", "user-456"), key="prefs", value={"theme": "light"}),
    SearchOp(namespace_prefix=("documents",), query="AI", limit=5)
])

# 非同期版
results = await store.abatch(operations)
```

---

## Namespace設計

Namespaceはタプル形式の階層構造．データの整理とアクセス制御に使用．

### 基本パターン

```python
# ユーザー単位のメモリ
namespace = ("memories", user_id)

# ユーザー + カテゴリ
namespace = ("memories", user_id, "preferences")
namespace = ("memories", user_id, "history")

# アプリケーション単位
namespace = ("app", app_id, "config")

# ドキュメント管理
namespace = ("documents", user_id, "uploads")
```

### 推奨設計パターン

```python
# パターン1: ユーザー中心
("users", user_id, "profile")
("users", user_id, "preferences")
("users", user_id, "memories")

# パターン2: 機能中心
("memories", user_id)
("documents", user_id)
("settings", user_id)

# パターン3: 階層的分類
("company", company_id, "employees", employee_id, "tasks")
```

### Namespaceの検索

```python
# 特定ユーザーの全メモリを検索
items = store.search(("memories", user_id))

# 全ユーザーのプロファイルを検索（上位階層）
items = store.search(("users",), filter={"type": "profile"})
```

---

## セマンティック検索

意味に基づいた検索を有効化．ベクトル埋め込みを使用．

### InMemoryStoreでの設定

```python
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings

# 埋め込みモデルを初期化
embeddings = init_embeddings("openai:text-embedding-3-small")

# インデックス設定付きでStoreを作成
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,  # text-embedding-3-smallの次元数
        "fields": ["text", "summary"]  # インデックス対象フィールド
    }
)
```

### PostgresStoreでの設定（pgvector使用）

```python
from langgraph.store.postgres import PostgresStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

store = PostgresStore.from_conn_string(
    DB_URI,
    index={
        "embed": embeddings,
        "dims": 1536,
        "fields": ["$"]  # 全フィールドをインデックス
    }
)
store.setup()  # pgvectorテーブル作成
```

### langgraph.jsonでの設定

```json
{
  "store": {
    "index": {
      "embed": "openai:text-embedding-3-small",
      "dims": 1536,
      "fields": ["text", "summary", "metadata.title"]
    }
  }
}
```

### インデックス対象フィールドの指定

```python
# 特定フィールドのみ
fields = ["text", "summary"]

# ネストされたフィールド
fields = ["metadata.title", "metadata.description"]

# 配列要素
fields = ["context[*].content"]

# 全フィールド
fields = ["$"]
```

### 検索時のインデックス制御

```python
# インデックスに含めない
store.put(
    namespace=("cache",),
    key="temp-data",
    value={"data": "..."},
    index=False  # セマンティック検索対象外
)

# 特定フィールドのみインデックス
store.put(
    namespace=("documents",),
    key="doc-1",
    value={"title": "...", "content": "...", "metadata": {...}},
    index=["title", "content"]  # metadataは除外
)
```

### セマンティック検索の実行

```python
# 類似度検索
results = store.search(
    ("documents",),
    query="機械学習の医療応用について",
    limit=5
)

for item in results:
    print(f"Key: {item.key}")
    print(f"Score: {item.score}")  # 類似度スコア
    print(f"Value: {item.value}")
```

---

## TTL（有効期限）管理

データに有効期限を設定して自動削除．

### TTL設定

```python
# 個別アイテムにTTLを設定（分単位）
store.put(
    namespace=("cache",),
    key="temp-result",
    value={"result": "..."},
    ttl=60  # 60分後に削除
)

# 7日間の有効期限
store.put(
    namespace=("sessions",),
    key="session-123",
    value={"user_id": "..."},
    ttl=60 * 24 * 7  # 10080分 = 7日
)
```

### TTL Sweeper（自動削除）

```python
from langgraph.store.postgres import PostgresStore

store = PostgresStore.from_conn_string(DB_URI)
store.setup()

# TTL Sweeperを開始（バックグラウンドで期限切れアイテムを削除）
store.start_ttl_sweeper(
    sweep_interval_minutes=5  # 5分ごとにチェック
)
```

### TTL設定オプション

```python
# TTLConfigで詳細設定
store = PostgresStore.from_conn_string(
    DB_URI,
    ttl_config={
        "default_ttl": 60 * 24,  # デフォルト24時間
        "refresh_on_read": True,  # 読み取り時にTTLリセット
        "sweep_interval_minutes": 10
    }
)
```

---

## 非同期実装

### AsyncPostgresSaver

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    await checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
```

### AsyncPostgresStore

```python
from langgraph.store.postgres.aio import AsyncPostgresStore

async with AsyncPostgresStore.from_conn_string(DB_URI) as store:
    await store.setup()
    graph = builder.compile(store=store)
```

### 両方を組み合わせる

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

async with (
    AsyncPostgresStore.from_conn_string(DB_URI) as store,
    AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
):
    await store.setup()
    await checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer, store=store)
```

### ノードでStoreを使用

```python
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

async def my_node(
    state: AgentState,
    config: RunnableConfig,
    *,
    store: BaseStore  # 自動注入
):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)

    # メモリを取得
    memories = await store.asearch(namespace, limit=10)

    # メモリを保存
    await store.aput(
        namespace,
        key=f"memory-{datetime.now().isoformat()}",
        value={"content": "新しい記憶", "type": "episodic"}
    )

    return {"messages": [...]}
```

---

## メモリ管理

### メッセージのトリム

コンテキストウィンドウ超過時に古いメッセージを削除：

```python
from langgraph.prebuilt import trim_messages

def trim_messages_node(state):
    messages = trim_messages(
        state["messages"],
        max_tokens=4000,
        strategy="last",  # 最新を保持
    )
    return {"messages": messages}
```

### メッセージの削除

```python
from langchain_core.messages import RemoveMessage

def delete_messages_node(state):
    # 特定のメッセージを削除
    return {"messages": [RemoveMessage(id=msg_id)]}
```

### メッセージの要約

```python
async def summarize_messages_node(state, config):
    messages = state["messages"]

    if len(messages) > 10:
        # 古いメッセージを要約
        summary = await model.ainvoke([
            SystemMessage(content="以下の会話を要約してください"),
            *messages[:-5]
        ])

        # 要約 + 最新5件を保持
        return {"messages": [summary] + messages[-5:]}

    return {"messages": messages}
```

### チェックポイント履歴の取得

```python
# スレッドの履歴を取得
history = checkpointer.list(config)

# 特定の状態に戻る
state = checkpointer.get(config, checkpoint_id)
```

---

## ベストプラクティス

### 1. Namespace設計

```python
# Good: 明確な階層構造
("memories", user_id, "preferences")
("memories", user_id, "history")

# Bad: フラットで曖昧
("user_preferences",)
("user_history",)
```

### 2. メモリの種類を分離

```python
# セマンティックメモリ（事実）
await store.aput(
    ("facts", user_id),
    "user-info",
    {"name": "田中", "role": "エンジニア"}
)

# エピソードメモリ（経験）
await store.aput(
    ("episodes", user_id),
    f"episode-{timestamp}",
    {"event": "プロジェクト完了", "context": "..."}
)
```

### 3. TTLの適切な設定

```python
# キャッシュ: 短いTTL
store.put(namespace, key, value, ttl=60)  # 1時間

# セッション: 中程度のTTL
store.put(namespace, key, value, ttl=60 * 24)  # 1日

# 永続データ: TTLなし
store.put(namespace, key, value)  # 無期限
```

### 4. セマンティック検索の最適化

```python
# 検索対象フィールドを限定
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
        "fields": ["summary", "keywords"]  # 全文ではなく要約のみ
    }
)
```

### 5. エラーハンドリング

```python
async def safe_memory_access(store, namespace, key):
    try:
        item = await store.aget(namespace, key)
        return item.value if item else None
    except Exception as e:
        logger.error(f"Memory access failed: {e}")
        return None
```

### 6. クロススレッドメモリの活用

```python
async def node_with_memory(state, config, *, store):
    user_id = config["configurable"]["user_id"]

    # ユーザーの過去の好みを取得（別スレッドで保存されたもの）
    prefs = await store.aget(("users", user_id), "preferences")

    # ユーザーの過去の会話から学習した情報
    memories = await store.asearch(
        ("memories", user_id),
        query=state["messages"][-1].content,
        limit=5
    )

    # コンテキストに追加
    context = {
        "preferences": prefs.value if prefs else {},
        "relevant_memories": [m.value for m in memories]
    }

    return {"context": context}
```
