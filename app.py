from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
import uvicorn

# 加载环境变量
load_dotenv()

# 设置OpenAI API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

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

# MBTI参考知识库 - 用于RAG
mbti_knowledge = {
    "I vs E": "内向型(I)的人倾向于从独处中获取能量，而外向型(E)的人从社交互动中获取能量。",
    "N vs S": "直觉型(N)的人关注可能性和未来，而感觉型(S)的人关注现实和事实。",
    "T vs F": "思考型(T)的人在决策时重视逻辑和客观分析，而情感型(F)的人重视和谐与个人价值。",
    "J vs P": "判断型(J)的人喜欢计划和结构，而感知型(P)的人更灵活和适应性强。",
    "INTJ": "建筑师 - 独立思考、有远见、高度自信",
    "INTP": "逻辑学家 - 创新、好奇、理性",
    "ENTJ": "指挥官 - 果断、有领导力、计划性强",
    "ENTP": "辩论家 - 机智、好奇、活跃",
    "INFJ": "提倡者 - 理想主义、有洞察力、有原则",
    "INFP": "调停者 - 理想主义、富有同情心、创意",
    "ENFJ": "主人公 - 热情、有说服力、利他主义",
    "ENFP": "活动家 - 热情、创意、社交能力强",
    "ISTJ": "物流师 - 实际、实事求是、可靠",
    "ISFJ": "守卫者 - 保护者、忠诚、传统",
    "ESTJ": "总经理 - 有组织、有力量、实用",
    "ESFJ": "执政官 - 关心他人、尽责、传统",
    "ISTP": "鉴赏家 - 冷静、灵活、观察者",
    "ISFP": "探险家 - 艺术性、灵活、魅力",
    "ESTP": "企业家 - 聪明、精力充沛、善于察言观色",
    "ESFP": "表演者 - 自发、精力充沛、热情"
}

async def analyze_text_with_llm(text: str):
    """使用RAG和LLM分析文本以确定MBTI类型"""
    
    # 构建包含RAG知识的提示
    prompt = f"""
    分析以下文本，确定作者最可能的MBTI人格类型。参考以下MBTI知识:
    
    内向(I) vs 外向(E):
    {mbti_knowledge["I vs E"]}
    
    直觉(N) vs 感觉(S):
    {mbti_knowledge["N vs S"]}
    
    思考(T) vs 情感(F):
    {mbti_knowledge["T vs F"]}
    
    判断(J) vs 感知(P):
    {mbti_knowledge["J vs P"]}
    
    请逐一分析每个维度，然后给出最终的MBTI类型和简短解释。
    
    文本: {text}
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一位MBTI分析专家，能够根据文本准确分析出人的性格类型。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        # 解析回复以提取MBTI类型和解释
        reply = response.choices[0].message.content
        
        # 简单解析，实际应用中可能需要更复杂的解析
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
    return {"message": "MBTI分析API正在运行"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 