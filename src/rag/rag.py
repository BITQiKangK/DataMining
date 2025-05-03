import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# 不再需要这些变量，因为我们将使用本地HuggingFace模型
# openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:8000/v1"

def load_and_process_document(file_path):
    """
    加载并处理文档
    
    Args:
        file_path: 文档路径
        
    Returns:
        处理后的文档列表
    """
    # 直接读取文本文件
    with open(file_path, "r", encoding="utf-8") as f:
        text_content = f.read()
    
    # 文本分割器
    text_splitter = CharacterTextSplitter(
        separator="\n\n",  # 使用两个换行符作为分隔符
        chunk_size=1000,   # 每块文本大约1000个字符
        chunk_overlap=200, # 块之间重叠200个字符，确保上下文连贯性
        length_function=len
    )
    
    # 分割文本
    chunks = text_splitter.split_text(text_content)
    
    # 创建文档对象
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    print(f"文档已加载和处理，共分割成{len(documents)}个块")
    return documents

def create_vector_store(documents, embedding_model, persist_directory):
    """
    创建向量存储
    
    Args:
        documents: 文档列表
        embedding_model: 嵌入模型
        persist_directory: 向量数据库持久化目录
        
    Returns:
        向量存储对象
    """
    # 创建向量存储
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    print(f"向量存储已创建并保存到{persist_directory}")
    return vectorstore

def main():
    """
    主函数，执行整个处理流程
    """
    # 文件路径
    file_path = os.path.join(os.path.dirname(__file__), "MBTIReference.md")
    
    # 持久化目录
    persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
    
    # 初始化嵌入模型（使用HuggingFace模型替代OpenAI）
    model_name = "/home/qikangkang/LLMModels/all-roberta-large-v1"  # 本地模型路径
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    
    # 加载和处理文档
    documents = load_and_process_document(file_path)
    
    # 创建向量存储
    vectorstore = create_vector_store(documents, embedding_model, persist_directory)
    
    # 测试查询
    query = "INFJ"
    results = vectorstore.similarity_search(query, k=2)
    
    print("\n查询测试:")
    print(f"查询: {query}")
    print("检索结果:")
    for i, doc in enumerate(results):
        print(f"结果 {i+1}:")
        print(doc.page_content[:200] + "...\n")
    
    print("向量化和存储过程完成！")

if __name__ == "__main__":
    main()
