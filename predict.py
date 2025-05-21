from joblib import load
from preprocess import prep_data
import time
import os
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.messages import SystemMessage, HumanMessage

###############################################################################
#            FUNCTION TO COMBINE THE RESULTS OF THE 4 MODELS                  #
###############################################################################


def trace_back(combined):
    type_list = [
        {"0": "I", "1": "E"},
        {"0": "N", "1": "S"},
        {"0": "F", "1": "T"},
        {"0": "P", "1": "J"},
    ]
    result = []
    for num in combined:
        s = ""
        for i in range(len(num)):
            s += type_list[i][num[i]]
        result.append(s)
    return result


def combine_classes(y_pred1, y_pred2, y_pred3, y_pred4):
    combined = []
    for i in range(len(y_pred1)):
        combined.append(
            str(y_pred1[i]) + str(y_pred2[i]) + str(y_pred3[i]) + str(y_pred4[i])
        )
    result = trace_back(combined)
    return result[0]


###############################################################################
#                           MODEL PREDICTIONS                                 #
###############################################################################


def predict(s, method="ml"):
    """
    预测MBTI类型
    
    参数:
        s (str): 要分析的文本
        method (str): 预测方法，'ml'表示机器学习模型，'llm'表示使用LLM
    
    返回:
        str: 预测的MBTI类型
    """
    if method == "llm":
        return predict_llm(s)
    else:
        return predict_ml(s)


def predict_ml(s):
    """使用机器学习模型进行预测"""
    X = prep_data(s)

    # loading the 4 models
    EorI_model = load(os.path.join("models", "clf_is_Extrovert.joblib"))
    SorN_model = load(os.path.join("models", "clf_is_Sensing.joblib"))
    TorF_model = load(os.path.join("models", "clf_is_Thinking.joblib"))
    JorP_model = load(os.path.join("models", "clf_is_Judging.joblib"))

    # predicting
    EorI_pred = EorI_model.predict(X)
    SorN_pred = SorN_model.predict(X)
    TorF_pred = TorF_model.predict(X)
    JorP_pred = JorP_model.predict(X)

    # combining the predictions from the 4 models
    result = combine_classes(EorI_pred, SorN_pred, TorF_pred, JorP_pred)

    return result


###############################################################################
#                           LLM PREDICTIONS                                  #
###############################################################################

# MBTI类型列表
mbti_type_list = ["INFJ", "INTJ", "ENFJ", "ENTJ", "ISFJ", "ISTJ", "ESFJ", "ESTJ", 
                  "INFP", "INTP", "ENFP", "ENTP", "ISFP", "ISTP", "ESFP", "ESTP"]

def clean_text(text):
    """清理文本"""
    # Replace triple pipes with period to separate thoughts
    text = text.replace('|||', '\n')
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove punctuation and numbers except periods
    text = re.sub(r'[^a-z.\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def initialize_vectorstore():
    """初始化向量数据库，从已有的Chroma数据库加载"""
    try:
        # 初始化embedding模型
        model_name = os.getenv("EMBEDDING_MODEL_PATH", "/home/qikangkang/LLMModels/all-roberta-large-v1")
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        
        persist_directory = os.path.join("rag", "chroma_db")
        
        # 从已有的持久化目录加载Chroma数据库
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        
        print(f"向量数据库已成功从{persist_directory}加载")
    except Exception as e:
        print(f"加载向量数据库时出错: {str(e)}")
        raise e
    return vectorstore

def get_prompt(text, vectorstore):
    """生成提示"""
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

        Example Answer:
        Energy Dimension (E/I): [Analysis]
        Information Acquisition (S/N): [Analysis]
        Decision-Making Style (T/F): [Analysis]
        Lifestyle (J/P): [Analysis]
        Comprehensive MBTI Type: [INTJ]
        """
    )
    
    # 创建提示
    prompt = prompt_template.format(
        context=context,
        text=text
    )
    return prompt

def parse_llm_response(reply):
    """解析LLM回复并提取MBTI类型"""
    # 提取MBTI类型
    mbti_type = ""
    for line in reply.split('\n'):
        if "Comprehensive MBTI Type" in line:
            for type_candidate in mbti_type_list:
                if type_candidate in line:
                    mbti_type = type_candidate
                    break
    
    # 如果没有找到类型，返回默认类型
    if not mbti_type:
        mbti_type = "INFJ"  # 设置一个默认值
        
    return mbti_type

def predict_llm(text):
    """使用RAG+LLM分析文本以确定MBTI类型"""
    # 清理文本
    clean_text_content = clean_text(text)
    
    try:
        # 初始化向量数据库
        vectorstore = initialize_vectorstore()
        
        # 获取提示
        prompt = get_prompt(clean_text_content, vectorstore)
        
        # 设置LLM模型
        openai_api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        openai_api_base = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:11434/v1")
        llm_model = os.getenv("LLM_MODEL", "gemma3:27b")
        
        chat_model = ChatOpenAI(
            model=llm_model,
            api_key=openai_api_key,
            base_url=openai_api_base
        )
        
        # 调用LLM
        messages = [
            SystemMessage(content="You are a MBTI analysis expert, able to accurately analyze the personality type of a person based on text."),
            HumanMessage(content=prompt)
        ]
        response = chat_model.invoke(messages)
        
        # 解析回复
        mbti_type = parse_llm_response(response.content)
        
        return mbti_type
    except Exception as e:
        print(f"使用LLM预测时出错: {str(e)}")
        # 出错时回退到ML模型
        return predict_ml(text)

###############################################################################
#                                   MAIN                                      #
###############################################################################

if __name__ == "__main__":
    t = time.time()
    string = "I just wanna to go home!!!!!! :sadpanda: https://www.youtube.com/watch?v=TQP20LTI84A"
    print(string)
    print(f"ML预测结果: {predict_ml(string)}")
    print(f"LLM预测结果: {predict_llm(string)}")
    print(f"预测时间: {time.time() - t} 秒")
