from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
import uvicorn

# 加载LangChain相关模块
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 设置OpenAI API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

class MBTIResult(BaseModel):
    mbti: str
    explanation: str

# 初始化嵌入模型和向量数据库
embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = None
llm = ChatOpenAI(temperature=0.2, api_key=OPENAI_API_KEY)

async def initialize_vectorstore():
    """初始化向量数据库，将MBTI参考知识加载并存入"""
    global vectorstore
    
    # 加载MBTI参考文档
    with open("MBTIReference.md", "r", encoding="utf-8") as f:
        mbti_text = f.read()
    
    # 文本分割
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(mbti_text)
    
    # 创建文档对象
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # 创建向量存储
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    
    print("向量数据库初始化完成，已加载MBTI参考知识")

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化向量数据库"""
    await initialize_vectorstore()

async def analyze_text_with_llm(text: str):
    """使用RAG和LLM分析文本以确定MBTI类型"""
    global vectorstore
    
    if vectorstore is None:
        await initialize_vectorstore()
    
    # 从向量数据库检索相关内容
    query = f"分析此人可能的MBTI类型: {text}"
    retrieval_results = vectorstore.similarity_search(query, k=3)  # 检索最相关的3段文本
    
    # 提取检索到的内容
    retrieved_contexts = [doc.page_content for doc in retrieval_results]
    context = "\n\n".join(retrieved_contexts)
    
    # 构建带有RAG内容的提示
    prompt_template = PromptTemplate(
        input_variables=["context", "text"],
        template="""
        你是一位MBTI性格分析专家。请根据以下参考知识和提供的文本，分析文本作者最可能的MBTI人格类型。

        ### MBTI参考知识:
        {context}

        ### 待分析文本:
        {text}

        请首先分析文本中的语言模式和行为倾向，然后用以下格式回答:
        1. 能量维度 (E/I): [分析]
        2. 信息获取 (S/N): [分析]
        3. 决策方式 (T/F): [分析]
        4. 生活方式 (J/P): [分析]
        5. 综合MBTI类型: [填入四字母类型，如INTJ]
        6. 解释: [简洁解释为什么这个人属于这种类型]
        """
    )
    
    # 创建提示
    prompt = prompt_template.format(
        context=context,
        text=text
    )
    
    try:
        # 调用LLM
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一位MBTI分析专家，能够根据文本准确分析出人的性格类型。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        # 解析回复
        reply = response.choices[0].message.content
        
        # 提取MBTI类型
        mbti_type = ""
        for line in reply.split('\n'):
            if "ISFJ" in line or "ISFP" in line or "ISTJ" in line or "ISTP" in line or \
               "INFJ" in line or "INFP" in line or "INTJ" in line or "INTP" in line or \
               "ESFJ" in line or "ESFP" in line or "ESTJ" in line or "ESTP" in line or \
               "ENFJ" in line or "ENFP" in line or "ENTJ" in line or "ENTP" in line:
                for mbti in ["ISFJ", "ISFP", "ISTJ", "ISTP", "INFJ", "INFP", "INTJ", "INTP",
                             "ESFJ", "ESFP", "ESTJ", "ESTP", "ENFJ", "ENFP", "ENTJ", "ENTP"]:
                    if mbti in line:
                        mbti_type = mbti
                        break
                if mbti_type:
                    break
        
        # 如果没有找到明确的类型，尝试识别四个维度
        if not mbti_type:
            # 在"综合MBTI类型"行中查找
            for line in reply.split('\n'):
                if "综合MBTI类型" in line or "最终类型" in line:
                    for mbti in ["ISFJ", "ISFP", "ISTJ", "ISTP", "INFJ", "INFP", "INTJ", "INTP",
                                "ESFJ", "ESFP", "ESTJ", "ESTP", "ENFJ", "ENFP", "ENTJ", "ENTP"]:
                        if mbti in line:
                            mbti_type = mbti
                            break
            
            # 如果仍未找到，则尝试分析四个维度
            if not mbti_type:
                i_or_e = "I" if "内向" in reply else "E"
                n_or_s = "N" if "直觉" in reply else "S"
                t_or_f = "T" if "思考" in reply else "F"
                j_or_p = "J" if "判断" in reply else "P"
                mbti_type = i_or_e + n_or_s + t_or_f + j_or_p
        
        return {
            "mbti": mbti_type,
            "explanation": reply
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM分析错误: {str(e)}")

@app.post("/analyze", response_model=MBTIResult)
async def analyze_mbti(input_data: TextInput):
    if not input_data.text:
        raise HTTPException(status_code=400, detail="文本不能为空")
    
    result = await analyze_text_with_llm(input_data.text)
    return result

@app.get("/")
async def root():
    return {"message": "MBTI分析API正在运行", "status": "向量数据库已加载"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 