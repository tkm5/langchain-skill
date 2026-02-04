# Gemini/Google GenAI 統合リファレンス

## 目次
1. [パッケージ情報](#パッケージ情報)
2. [ChatGoogleGenerativeAI](#chatgooglegenerativeai)
3. [ツールバインディング](#ツールバインディング)
4. [埋め込みモデル](#埋め込みモデル)

## パッケージ情報

### インストール

```bash
pip install langchain-google-genai
```

### 重要な変更点（v4.0.0以降）

- 統合された `google-genai` SDKを使用（legacy `google-ai-generativelanguage` は非推奨）
- Gemini Developer APIとVertex AIの両方をサポート

## ChatGoogleGenerativeAI

### 基本的な使用方法

```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
)

response = model.invoke("Hello, world!")
print(response.content)
```

### ストリーミング

```python
for chunk in model.stream("長い回答を生成してください"):
    print(chunk.content, end="", flush=True)
```

### 非同期実行

```python
response = await model.ainvoke("非同期で実行")
```

### 設定オプション

```python
model = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,          # 創造性の度合い
    max_tokens=1000,          # 最大トークン数
    top_p=0.95,               # 核サンプリング
    top_k=40,                 # Top-Kサンプリング
    timeout=30,               # タイムアウト秒数
)
```

## ツールバインディング

### ツールの定義

```python
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """指定した場所の天気を取得"""
    # 実装
    return f"{location}の天気は晴れです"

@tool
def search_database(query: str) -> list[dict]:
    """データベースを検索"""
    # 実装
    return [{"result": "データ"}]
```

### モデルへのバインド

```python
tools = [get_weather, search_database]
model_with_tools = model.bind_tools(tools)

response = model_with_tools.invoke("東京の天気は？")

# ツール呼び出しの確認
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call['name']}")
        print(f"Args: {tool_call['args']}")
```

### Pydanticスキーマでのツール定義

```python
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    """検索クエリのスキーマ"""
    query: str = Field(description="検索キーワード")
    max_results: int = Field(default=5, description="最大結果数")

class WeatherRequest(BaseModel):
    """天気リクエストのスキーマ"""
    location: str = Field(description="場所名")
    unit: str = Field(default="celsius", description="温度単位")

# スキーマをツールとしてバインド
model_with_tools = model.bind_tools([SearchQuery, WeatherRequest])
```

### 構造化出力

```python
from pydantic import BaseModel

class OutputSchema(BaseModel):
    summary: str
    keywords: list[str]
    sentiment: str

structured_model = model.with_structured_output(OutputSchema)
result = structured_model.invoke("このテキストを分析してください: ...")

print(result.summary)
print(result.keywords)
```

## 埋め込みモデル

### GoogleGenerativeAIEmbeddings

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=os.getenv("GEMINI_API_KEY"),
)

# 単一テキストの埋め込み
vector = embeddings.embed_query("Hello, world!")

# 複数テキストの埋め込み
vectors = embeddings.embed_documents([
    "テキスト1",
    "テキスト2",
    "テキスト3",
])
```

### ベクトルストアとの統合

```python
from langchain_community.vectorstores import FAISS

# ドキュメントからベクトルストアを作成
vectorstore = FAISS.from_documents(documents, embeddings)

# 類似検索
results = vectorstore.similarity_search("検索クエリ", k=5)
```

## LangGraphとの統合例

### ReActエージェント

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

# モデル初期化
model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

# ツール定義
@tool
def search(query: str) -> str:
    """Web検索を実行"""
    return "検索結果"

# ツールをバインド
model_with_tools = model.bind_tools([search])

# ノード関数
def call_model(state: MessagesState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def call_tool(state: MessagesState):
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "search":
            result = search.invoke(tool_call["args"])
            results.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
    return {"messages": results}

# ルーティング
def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "call_tool"
    return "end"

# グラフ構築
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("call_tool", call_tool)
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", should_continue, {"call_tool": "call_tool", "end": END})
builder.add_edge("call_tool", "call_model")

graph = builder.compile()
```
