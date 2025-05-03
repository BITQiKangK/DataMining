# MBTI分析API

这是一个基于FastAPI的后端服务，可以分析文本并确定作者最可能的MBTI人格类型。该服务使用检索增强生成(RAG)和大型语言模型(LLM)来进行分析。

## 功能

- 接收文本输入
- 使用向量数据库存储MBTI参考知识
- 根据输入文本检索相关MBTI知识
- 使用RAG技术和OpenAI API分析文本
- 返回推测的MBTI类型和详细解释

## 技术实现

- **向量数据库**: 使用Chroma存储MBTI参考知识的向量嵌入
- **文本分割**: 使用LangChain的CharacterTextSplitter将参考文档分割成适当大小的块
- **语义检索**: 根据用户输入文本，从向量数据库中检索最相关的MBTI参考知识
- **RAG技术**: 将检索到的知识与用户输入结合，使用OpenAI API进行增强分析

## 安装

1. 克隆此仓库
2. 安装依赖：
   ```
   pip install -r requirements.txt
   ```
3. 复制`.env-example`文件为`.env`并添加你的OpenAI API密钥：
   ```
   cp .env-example .env
   ```
   然后编辑`.env`文件添加你的API密钥

## 运行服务

```
python app.py
```

服务将在 http://localhost:8000 上运行，启动时会自动将MBTIReference.md文件加载到向量数据库中。

## API端点

### GET /
检查API是否正常运行

### POST /analyze
分析文本并返回MBTI类型

#### 请求格式
```json
{
  "text": "要分析的文本内容"
}
```

#### 响应格式
```json
{
  "mbti": "INTJ",
  "explanation": "详细的分析解释..."
}
```

## 示例使用

```python
import requests

url = "http://localhost:8000/analyze"
data = {
    "text": "我喜欢独自思考问题，经常会想到一些创新的解决方案。我重视逻辑和效率，不太喜欢即兴的社交活动。"
}

response = requests.post(url, json=data)
print(response.json())
``` 