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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.messages import SystemMessage, HumanMessage

openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:11434/v1"

# 初始化全局向量数据库变量
vectorstore = None

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

# 添加向量数据库初始化函数
async def initialize_vectorstore():
    """初始化向量数据库，从已有的Chroma数据库加载"""
    global vectorstore
    
    try:
        # 设置持久化目录路径
        persist_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rag", "chroma_db")
        
        # 使用HuggingFace嵌入模型，与原始数据库创建时使用的模型保持一致
        model_name = "/home/qikangkang/LLMModels/all-roberta-large-v1"  # 本地模型路径
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        
        # 从已有的持久化目录加载Chroma数据库
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        
        print(f"向量数据库已成功从{persist_directory}加载")
    except Exception as e:
        print(f"加载向量数据库时出错: {str(e)}")
        raise e

async def initialize_chat_model():
    global chat_model
    chat_model = ChatOpenAI(
        model="llama3.1:8b",
        api_key=openai_api_key,
        base_url=openai_api_base
    )

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化向量数据库"""
    await initialize_vectorstore()
    await initialize_chat_model()

async def analyze_text_with_llm(text: str):
    """使用RAG和LLM分析文本以确定MBTI类型"""
    global vectorstore
    global chat_model
    
    if vectorstore is None:
        await initialize_vectorstore()

    if chat_model is None:
        await initialize_chat_model()
    
    # 从向量数据库检索相关内容
    query = f"Analyze the most likely MBTI personality type of the text author based on the following reference knowledge and the provided text: {text}"
    retrieval_results = vectorstore.similarity_search(query, k=3)  # 检索最相关的3段文本
    
    # 提取检索到的内容
    retrieved_contexts = [doc.page_content for doc in retrieval_results]
    context = "\n\n".join(retrieved_contexts)
    
    # 构建带有RAG内容的提示
    prompt_template = PromptTemplate(
        input_variables=["context", "text"],
        template="""
        You are a MBTI personality analysis expert. Please analyze the most likely MBTI personality type of the text author based on the following reference knowledge and the provided text.

        ### MBTI Reference Knowledge:
        {context}

        ### Text to Analyze:
        {text}

        Please first analyze the language patterns and behavioral tendencies in the text, and then answer in the following format:
        1. Energy Dimension (E/I): [Analysis]
        2. Information Acquisition (S/N): [Analysis]
        3. Decision-Making Style (T/F): [Analysis]
        4. Lifestyle (J/P): [Analysis]
        5. Comprehensive MBTI Type: [Fill in the four-letter type, e.g., INTJ]
        6. Explanation: [Brief explanation of why this person belongs to this type]
        """
    )
    
    # 创建提示
    prompt = prompt_template.format(
        context=context,
        text=text
    )
    
    try:
        # 调用LLM
        messages = [
            SystemMessage(content="You are a MBTI analysis expert, able to accurately analyze the personality type of a person based on text."),
            HumanMessage(content=prompt)
        ]
        response = chat_model.invoke(messages)
        
        # 解析回复
        reply = response.content
        
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
                if "Comprehensive MBTI Type" in line or "Final Type" in line:
                    for mbti in ["ISFJ", "ISFP", "ISTJ", "ISTP", "INFJ", "INFP", "INTJ", "INTP",
                                "ESFJ", "ESFP", "ESTJ", "ESTP", "ENFJ", "ENFP", "ENTJ", "ENTP"]:
                        if mbti in line:
                            mbti_type = mbti
                            break
            
            # 如果仍未找到，则尝试分析四个维度
            if not mbti_type:
                i_or_e = "I" if "Introverted" in reply else "E"
                n_or_s = "N" if "Intuitive" in reply else "S"
                t_or_f = "T" if "Thinking" in reply else "F"
                j_or_p = "J" if "Judging" in reply else "P"
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
    uvicorn.run(app, host="0.0.0.0", port=2233) 