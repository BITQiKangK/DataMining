import pandas as pd
import numpy as np
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from sklearn.preprocessing import LabelEncoder
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.messages import SystemMessage, HumanMessage
from sklearn.metrics import accuracy_score, f1_score
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def initialize_vectorstore():
    """初始化向量数据库，从已有的Chroma数据库加载"""
    
    try:
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

def clean_text(text):
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

def get_prompt(text: str):
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

def analyze_text_with_llm(text: str):
    """使用RAG和LLM分析单个文本以确定MBTI类型"""
    
    prompt = get_prompt(text)
    
    # 调用LLM
    messages = [
        SystemMessage(content="You are a MBTI analysis expert, able to accurately analyze the personality type of a person based on text."),
        HumanMessage(content=prompt)
    ]
    response = chat_model.invoke(messages)

    # 解析回复
    result = parse_llm_response(response.content)
    if result['mbti'] == "":
        result['mbti'] = "ERROR"
    return result

mbti_type_list = ["INFJ", "INTJ", "ENFJ", "ENTJ", "ISFJ", "ISTJ", "ESFJ", "ESTJ", "INFP", "INTP", "ENFP", "ENTP", "ISFP", "ISTP", "ESFP", "ESTP"]

def parse_llm_response(reply):
    """解析LLM回复并提取MBTI类型"""
    # 提取MBTI类型
    mbti_type = ""
    for line in reply.split('\n'):
        if "Comprehensive MBTI Type" in line:
            for mbti_type in mbti_type_list:
                if mbti_type in line:
                    mbti_type = mbti_type
                    break
    return {
        "mbti": mbti_type,
        "explanation": reply
    }

if __name__ == "__main__":
    orig_df = pd.read_csv('dataset/mbti.csv')
    df = orig_df.copy()

    # Reapply cleaning
    df['clean_posts'] = df['posts'].apply(clean_text)

    model_name = "/home/qikangkang/LLMModels/all-roberta-large-v1"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    if (os.path.exists('dataset/text_vector.npy')):
        X = np.load('dataset/text_vector.npy')
    else:
        X = embedding_model.embed_documents(df['clean_posts'])
        np.save('dataset/text_vector.npy', X)

    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory="rag/chroma_db")
    
    # Encode MBTI types to integers
    le = LabelEncoder()
    y = le.fit_transform(df['type'])

    vectorstore = initialize_vectorstore()

    openai_api_key = "EMPTY"
    openai_api_base = "http://127.0.0.1:11434/v1"
    chat_model = ChatOpenAI(
        model="gemma3:27b",
        api_key=openai_api_key,
        base_url=openai_api_base
    )

    # 从每种MBTI类型中抽取5个样本
    sampled_df = pd.DataFrame()
    for mbti_type in mbti_type_list:
        type_df = df[df['type'] == mbti_type]
        if len(type_df) > 5:
            sampled_type_df = type_df.sample(5, random_state=42)
        else:
            sampled_type_df = type_df  # 如果样本数小于5，则全部使用
        sampled_df = pd.concat([sampled_df, sampled_type_df])
    
    # 重置索引
    sampled_df = sampled_df.reset_index(drop=True)
    
    print(f"已从每种MBTI类型中抽取样本，总共选择了{len(sampled_df)}个样本进行预测")

    predictions = []
    explanations = []
    
    print(f"开始预测抽样后的{len(sampled_df)}个样本的MBTI类型")
    
    for i in tqdm(range(len(sampled_df))):
        text = sampled_df['clean_posts'][i]
        for j in range(3):
            result = analyze_text_with_llm(text)
            if result['mbti'] != "ERROR":
                predictions.append(result['mbti'])
                explanations.append(result['explanation'])
                break
        if result['mbti'] == "ERROR":
            predictions.append("ERROR")
            explanations.append("ERROR")

    # 将预测结果添加到DataFrame
    sampled_df['predicted_type'] = predictions
    sampled_df['explanation'] = explanations

    # 计算准确率
    valid_predictions = sampled_df[sampled_df['predicted_type'] != "ERROR"]
    accuracy = accuracy_score(valid_predictions['type'], valid_predictions['predicted_type'])
    f1 = f1_score(valid_predictions['type'], valid_predictions['predicted_type'], average='weighted')

    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"F1分数 (F1-Score): {f1:.4f}")

    # 生成混淆矩阵
    true_labels = valid_predictions['type']
    pred_labels = valid_predictions['predicted_type']
    
    cm = confusion_matrix(true_labels, pred_labels, labels=mbti_type_list)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=mbti_type_list, yticklabels=mbti_type_list)
    plt.xlabel('预测类型')
    plt.ylabel('真实类型')
    plt.title('MBTI类型预测混淆矩阵')
    plt.savefig('confusion_matrix.png')
    plt.close()

    sampled_df.to_csv('dataset/mbti_sampled_predicted.csv', index=False)