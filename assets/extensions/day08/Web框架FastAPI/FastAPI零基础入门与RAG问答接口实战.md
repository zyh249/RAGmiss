# FastAPI 零基础入门与 RAG 问答接口实战

## 学习目标

学完本章后，能够完成以下任务：

1. 理解 FastAPI 的作用。
2. 掌握 FastAPI 项目的基本结构。
3. 掌握 GET 接口和 POST 接口的基本写法。
4. 理解 `BaseModel` 在接口参数接收中的作用。
5. 能够编写一个简单的问答接口。
6. 能够在 FastAPI 接口中逐步整合 FAQ、BM25 和 RAG 问答系统。
7. 理解普通 HTTP 接口和 WebSocket 流式接口的区别。

---

# 第一部分 FastAPI 基础入门

## 一、FastAPI 是什么

FastAPI 是一个 Python Web 框架。

简单理解：

```text
FastAPI = 用 Python 写接口的工具
```

在普通 Python 程序中，我们可以写一个函数：

```python
def ask(question):
    return "你问的是：" + question
```

这个函数只能在 Python 代码内部调用。

如果希望前端页面、手机 App 或其他系统也能调用这个函数，就需要把函数封装成接口。

调用过程如下：

```text
前端页面  --->  访问接口  --->  Python 后端函数  --->  返回结果
```

FastAPI 的作用就是：

```text
把 Python 函数变成可以通过网址访问的接口
```

---

## 二、安装 FastAPI

如果使用 `uv` 管理项目依赖，可以执行：

```bash
uv add fastapi uvicorn
```

如果使用 `pip` 安装，可以执行：

```bash
pip install fastapi uvicorn
```

两个包的作用如下：

| 包名 | 作用 |
|---|---|
| `fastapi` | 用来编写接口 |
| `uvicorn` | 用来启动 FastAPI 服务 |

可以这样理解：

```text
FastAPI 负责写接口
Uvicorn 负责运行接口服务
```

---

## 三、第一个 FastAPI 程序

新建文件：

```text
app.py
```

写入代码：

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def hello():
    return {"message": "你好，FastAPI"}
```

启动服务：

```bash
uvicorn app:app --reload


--reload表示自动重启。
你修改代码后，不需要手动停止再启动，uvicorn 会自动重新加载。
```

浏览器访问：

```text
http://127.0.0.1:8000/
```

返回结果：

```json
{
  "message": "你好，FastAPI"
}
```

---

如果不希望每次都在终端中运行FastAPI代码，我们还可以采用如下方式：

```python
import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello():
    return {"message": "你好，FastAPI"}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
```



## 四、代码逐行解释

### 1. 导入 FastAPI

```python
from fastapi import FastAPI
```

表示从 `fastapi` 包中导入 `FastAPI`。

---

### 2. 创建 FastAPI 应用对象

```python
app = FastAPI()
```

表示创建一个 FastAPI 应用对象。

可以理解为：

```text
app = 后端服务对象
```

后面定义的接口，都要绑定到这个 `app` 对象上。

---

### 3. 定义 GET 接口

```python
@app.get("/")
```

表示定义一个 GET 接口，访问路径是 `/`。

也就是说，当用户访问：

```text
http://127.0.0.1:8000/
```

就会执行下面的函数。

---

### 4. 定义接口函数

```python
def hello():
    return {"message": "你好，FastAPI"}
```

表示接口真正要执行的函数。

函数返回的是 Python 字典，FastAPI 会自动把它转换成 JSON 格式。

### 5. uvicorn 启动 FastAPI

什么是uvicorn？

```
uvicorn 是 FastAPI 常用的 ASGI 服务器，用来运行 FastAPI 应用。
```

可以把它理解成：

```
FastAPI = 写接口逻辑
uvicorn = 启动接口服务
```

类似关系：

```
你的代码文件 demo01_fastapi_demo.py
        ↓
FastAPI 创建 app
        ↓
uvicorn 启动 app
        ↓
浏览器 / Postman 才能访问接口
```



## 五、理解接口地址

完整接口地址通常由几部分组成：

```text
http://127.0.0.1:8000/
```

| 部分 | 含义 |
|---|---|
| `http://` | 协议 |
| `127.0.0.1` | 本机地址 |
| `8000` | 服务端口 |
| `/` | 接口路径 |

例如代码如下：

```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

访问地址就是：

```text
http://127.0.0.1:8000/health
```

---

# 第二部分 GET 接口

## 一、GET 接口适合做什么

GET 接口一般用于查询数据。

常见场景包括：

1. 查询服务是否正常。
2. 查询文章列表。
3. 查询用户信息。
4. 查询系统支持的分类。

---

## 二、健康检查接口

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "healthy"}
```

访问地址：

```text
http://127.0.0.1:8000/health
```

返回结果：

```json
{
  "status": "healthy"
}
```

这个接口通常用于检查服务是否正常。

如果接口能够正常返回结果，说明 FastAPI 服务已经启动成功。

---

## 三、带参数的 GET 接口

如果希望通过地址传递参数，可以这样写：

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/user")
def get_user(name: str):
    return {"username": name}
```

访问地址：

```text
http://127.0.0.1:8000/user?name=张三
```

返回结果：

```json
{
  "username": "张三"
}
```

其中：

```python
def get_user(name: str):
```

表示从 URL 参数中接收 `name`。

---

# 第三部分 POST 接口

## 一、POST 接口适合做什么

POST 接口一般用于提交数据。

常见场景包括：

1. 提交用户问题。
2. 提交登录信息。
3. 新增一条数据。
4. 提交聊天内容。

问答系统通常使用 POST 接口，因为用户需要向后端提交问题。

---

## 二、最简单的 POST 问答接口

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@app.post("/api/query")
def query(request: QueryRequest):
    return {
        "answer": "你问的是：" + request.query
    }
```

请求地址：

```text
http://127.0.0.1:8000/api/query
```

请求方式：

```text
POST
```

请求数据：

```json
{
  "query": "什么是 Python？"
}
```

响应结果：

```json
{
  "answer": "你问的是：什么是 Python？"
}
```

---

## 三、BaseModel 是什么

在代码中有这样一段：

```python
class QueryRequest(BaseModel):
    query: str
```

这段代码表示定义请求数据格式。

它要求前端提交的数据必须包含 `query` 字段：

```json
{
  "query": "用户问题"
}
```

如果前端没有传 `query` 字段，FastAPI 会自动提示参数错误。

所以 `BaseModel` 的作用是：

```text
规定请求数据格式，并自动校验数据
```

---

## 四、为什么问答系统适合使用 BaseModel

如果问答接口只接收一个参数，可以直接接收字符串。

但是实际问答系统通常不止一个字段，例如：

```json
{
  "query": "什么是 Python？",
  "source_filter": "python",
  "session_id": "abc123"
}
```

字段说明如下：

| 字段 | 含义 |
|---|---|
| `query` | 用户问题 |
| `source_filter` | 查询范围过滤条件 |
| `session_id` | 会话 ID，用于区分不同用户或不同对话 |

这种情况下，使用 `BaseModel` 更清晰。

---

# 第四部分 构建简单问答接口

## 一、需求说明

先不接入真实 RAG 系统，只实现一个简单问答接口。

接口功能如下：

1. 接收用户问题。
2. 判断是否为简单问候语。
3. 返回对应答案。
4. 自动生成会话 ID。
5. 返回接口处理耗时。

---

## 二、完整代码

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uuid
import time

app = FastAPI(title="零基础问答系统")


class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None


@app.get("/")
def home():
    return {"message": "问答系统启动成功"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/create_session")
def create_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


@app.post("/api/query")
def query(request: QueryRequest):
    start_time = time.time()

    session_id = request.session_id or str(uuid.uuid4())

    if request.query in ["你好", "您好", "hi", "hello"]:
        answer = "你好，我是智能问答助手。"
    else:
        answer = "你问的是：" + request.query

    return {
        "answer": answer,
        "session_id": session_id,
        "source_filter": request.source_filter,
        "processing_time": time.time() - start_time
    }
```

---

## 三、接口说明

这个版本实现了 4 个接口：

| 接口 | 请求方式 | 作用 |
|---|---|---|
| `/` | GET | 首页测试 |
| `/health` | GET | 健康检查 |
| `/api/create_session` | POST | 创建会话 ID |
| `/api/query` | POST | 提交问题并返回答案 |

---

## 四、访问接口文档

启动服务后，访问：

```text
http://127.0.0.1:8000/docs
```

FastAPI 会自动生成接口文档。

可以在接口文档中直接测试接口，不需要单独编写前端页面。

---

# 第五部分 接入 FAQ 问答

## 一、什么是 FAQ

FAQ 是常见问题库。

例如：

```python
faq_data = {
    "什么是Python": "Python是一门简单易学的编程语言。",
    "什么是FastAPI": "FastAPI是一个用于构建API接口的Python框架。",
    "什么是RAG": "RAG是一种检索增强生成技术。"
}
```

用户问题如果正好匹配 FAQ，就直接返回答案。

---

## 二、FAQ 问答接口代码

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uuid
import time

app = FastAPI(title="FAQ问答系统")


faq_data = {
    "什么是Python": "Python是一门简单易学的编程语言。",
    "什么是FastAPI": "FastAPI是一个用于构建API接口的Python框架。",
    "什么是RAG": "RAG是一种检索增强生成技术。"
}


class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None


@app.post("/api/query")
def query(request: QueryRequest):
    start_time = time.time()

    session_id = request.session_id or str(uuid.uuid4())

    if request.query in faq_data:
        answer = faq_data[request.query]
    else:
        answer = "FAQ中没有找到这个问题。"

    return {
        "answer": answer,
        "session_id": session_id,
        "processing_time": time.time() - start_time
    }
```

---

## 三、FAQ 查询流程

```text
用户提交问题
    ↓
FastAPI 接收请求
    ↓
从 request.query 取出问题
    ↓
判断问题是否在 faq_data 中
    ↓
如果存在，返回 FAQ 答案
    ↓
如果不存在，返回未找到
```

---

## 四、FAQ 方式的问题

FAQ 字典匹配要求用户问题和标准问题完全一致。

例如 FAQ 中保存的问题是：

```text
什么是Python
```

用户输入：

```text
Python是什么
```

虽然意思接近，但是字符串不完全一样，普通 FAQ 字典无法命中。

因此，需要引入 BM25 检索。

---

# 第六部分 接入 BM25 检索

## 一、为什么需要 BM25

BM25 的作用是从问题库中找出和用户问题最相似的问题。

可以理解为：

```text
FAQ 字典：要求问题完全一样
BM25 检索：允许问题表达方式不完全一样
```

例如：

```text
标准问题：什么是Python
用户问题：Python是什么
```

BM25 可以根据关键词相似度，判断这两个问题可能表达同一个意思。

---

## 二、先使用模拟 BM25

为了先理解流程，可以先写一个模拟 BM25 函数。

```python
def bm25_search(query: str):
    if "Python" in query or "python" in query:
        return "Python是一门简单易学的编程语言。", False

    if "FastAPI" in query or "fastapi" in query:
        return "FastAPI是一个用于构建API接口的Python框架。", False

    return None, True
```

这个函数返回两个值：

```python
return answer, need_rag
```

| 返回值 | 含义 |
|---|---|
| `answer` | 找到的答案 |
| `need_rag` | 是否需要进入 RAG |

如果 FAQ 找到了答案：

```python
return "答案", False
```

如果 FAQ 没有找到答案：

```python
return None, True
```

---

## 三、整合 BM25 到 FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uuid
import time

app = FastAPI(title="FAQ + BM25 问答系统")


class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None


def bm25_search(query: str):
    if "Python" in query or "python" in query:
        return "Python是一门简单易学的编程语言。", False

    if "FastAPI" in query or "fastapi" in query:
        return "FastAPI是一个用于构建API接口的Python框架。", False

    return None, True


@app.post("/api/query")
def query(request: QueryRequest):
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    answer, need_rag = bm25_search(request.query)

    if need_rag:
        answer = "FAQ没有找到答案，后续需要进入RAG系统。"

    return {
        "answer": answer,
        "need_rag": need_rag,
        "session_id": session_id,
        "processing_time": time.time() - start_time
    }
```

---

## 四、BM25 查询流程

```text
用户提交问题
    ↓
FastAPI 接收请求
    ↓
调用 bm25_search
    ↓
如果找到 FAQ 答案，直接返回
    ↓
如果没有找到 FAQ 答案，标记 need_rag=True
```

---

# 第七部分 接入 RAG 问答

## 一、RAG 在系统中的作用

当前系统流程是：

```text
用户问题
    ↓
先查 FAQ / BM25
    ↓
如果找到答案，直接返回
    ↓
如果找不到，再进入 RAG
```

这样设计的原因是：

```text
FAQ / BM25 适合回答高频标准问题
RAG 适合回答复杂知识库问题
```

FAQ 和 BM25 通常速度较快。

RAG 需要检索知识库，并调用大模型生成答案，通常耗时更长。

---

## 二、先使用模拟 RAG

为了先理解接口流程，可以先写一个模拟 RAG 函数。

```python
def rag_query(query: str, source_filter: Optional[str] = None):
    return f"这是RAG系统根据知识库生成的答案，用户问题是：{query}"
```

这个函数暂时不连接真实向量数据库，只模拟 RAG 返回结果。

---

## 三、FAQ + BM25 + RAG 完整整合版

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uuid
import time

app = FastAPI(title="FAQ + BM25 + RAG 问答系统")


class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None


def check_greeting(query: str):
    greetings = {
        "你好": "你好，我是智能问答助手。",
        "您好": "您好，很高兴为你服务。",
        "hi": "Hi，我是智能问答助手。",
        "hello": "Hello，我是智能问答助手。"
    }
    return greetings.get(query.strip())


def bm25_search(query: str):
    if "Python" in query or "python" in query:
        return "Python是一门简单易学的编程语言。", False

    if "FastAPI" in query or "fastapi" in query:
        return "FastAPI是一个用于构建API接口的Python框架。", False

    return None, True


def rag_query(query: str, source_filter: Optional[str] = None):
    return f"这是RAG系统根据知识库生成的答案，用户问题是：{query}"


@app.get("/")
def home():
    return {"message": "FAQ + RAG 问答系统启动成功"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/create_session")
def create_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


@app.post("/api/query")
def query(request: QueryRequest):
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    greeting_answer = check_greeting(request.query)
    if greeting_answer:
        return {
            "answer": greeting_answer,
            "source": "greeting",
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }

    faq_answer, need_rag = bm25_search(request.query)

    if not need_rag:
        return {
            "answer": faq_answer,
            "source": "faq_bm25",
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }

    rag_answer = rag_query(request.query, request.source_filter)

    return {
        "answer": rag_answer,
        "source": "rag",
        "is_streaming": False,
        "session_id": session_id,
        "processing_time": time.time() - start_time
    }
```

---

## 四、最终查询流程

```text
用户提交问题
    ↓
FastAPI 接收 POST 请求
    ↓
生成或获取 session_id
    ↓
判断是否为问候语
    ↓
如果是问候语，直接返回固定回复
    ↓
如果不是问候语，进入 BM25 FAQ 检索
    ↓
如果 FAQ 命中，返回 FAQ 答案
    ↓
如果 FAQ 未命中，调用 RAG 系统
    ↓
返回 RAG 答案
```

---

# 第八部分 对接真实 RAG 系统

## 一、真实项目中的 RAG 对象

在真实项目中，通常会提前封装一个问答系统对象，例如：

```python
from new_main import IntegratedQASystem

qa_system = IntegratedQASystem()
```

这个对象中通常包含：

1. FAQ 检索模块。
2. BM25 检索模块。
3. 向量数据库检索模块。
4. 大模型生成模块。
5. 历史对话管理模块。

---

## 二、真实 BM25 调用方式

模拟代码中使用的是：

```python
answer, need_rag = bm25_search(request.query)
```

真实项目中可以替换为：

```python
answer, need_rag = qa_system.bm25_search.search(request.query, threshold=0.85)
```

其中：

| 参数 | 含义 |
|---|---|
| `request.query` | 用户问题 |
| `threshold=0.85` | 相似度阈值 |

可以理解为：

```text
如果 BM25 找到相似度超过 0.85 的 FAQ 问题，就返回 FAQ 答案
如果没有达到 0.85，就进入 RAG
```

---

## 三、真实 RAG 调用方式

模拟代码中使用的是：

```python
rag_answer = rag_query(request.query, request.source_filter)
```

真实项目中可以替换为：

```python
for token, is_complete in qa_system.query(
    request.query,
    source_filter=request.source_filter,
    session_id=session_id
):
    pass
```

这里的 `qa_system.query()` 通常会返回流式结果。

每次返回：

```python
token, is_complete
```

| 返回值 | 含义 |
|---|---|
| `token` | 当前生成的一小段内容 |
| `is_complete` | 是否生成完成 |

---

# 第九部分 WebSocket 流式输出

## 一、为什么需要 WebSocket

普通 HTTP 接口的特点是：

```text
用户提交问题
    ↓
后端处理完成
    ↓
一次性返回完整答案
```

如果 RAG 调用大模型生成答案比较慢，用户需要等待较长时间。

WebSocket 的特点是：

```text
用户提交问题
    ↓
后端一边生成
    ↓
一边返回内容
```

这种效果类似聊天机器人逐字输出答案。

---

## 二、WebSocket 输出数据格式

流式输出时，可以约定返回以下几类消息：

### 1. 开始消息

```json
{
  "type": "start",
  "session_id": "xxx"
}
```

表示本次回答开始。

---

### 2. 内容消息

```json
{
  "type": "token",
  "token": "当前生成的内容",
  "session_id": "xxx"
}
```

表示当前生成的一小段内容。

---

### 3. 结束消息

```json
{
  "type": "end",
  "session_id": "xxx",
  "is_complete": true
}
```

表示本次回答结束。

---

### 4. 错误消息

```json
{
  "type": "error",
  "error": "错误信息"
}
```

表示接口执行过程中出现异常。

---

## 三、WebSocket 基本代码结构

```python
from fastapi import WebSocket


@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_text()
        await websocket.send_json({
            "type": "token",
            "token": "收到的问题是：" + data
        })
```

代码说明：

| 代码 | 含义 |
|---|---|
| `@app.websocket("/api/stream")` | 定义 WebSocket 接口 |
| `await websocket.accept()` | 接受连接 |
| `await websocket.receive_text()` | 接收前端发送的文本 |
| `await websocket.send_json()` | 向前端发送 JSON 数据 |

---

# 第十部分 本章小结

FastAPI 的核心作用是：

```text
把 Python 函数封装成接口，让前端或其他系统可以访问
```

本章学习路线如下：

```text
FastAPI 基础
    ↓
GET 接口
    ↓
POST 接口
    ↓
BaseModel 请求模型
    ↓
简单问答接口
    ↓
FAQ 问答
    ↓
BM25 检索
    ↓
RAG 问答
    ↓
WebSocket 流式输出
```

在 RAG 问答系统中，FastAPI 负责接收和返回数据：

```text
前端问题  --->  FastAPI 接口  --->  FAQ / BM25 / RAG  --->  返回答案
```

各模块分工如下：

| 模块 | 作用 |
|---|---|
| FastAPI | 接收前端请求，返回接口响应 |
| FAQ | 处理标准高频问题 |
| BM25 | 从 FAQ 中查找相似问题 |
| RAG | 检索知识库并调用大模型生成答案 |
| WebSocket | 实现流式输出效果 |

学习 FastAPI 时，应该先掌握以下 4 个核心内容：

```python
app = FastAPI()
@app.get()
@app.post()
BaseModel
```

掌握这些内容后，再整合 FAQ、BM25、RAG 和 WebSocket，完整问答系统的代码就会更容易理解。
