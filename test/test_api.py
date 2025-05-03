import requests
import json

def test_api():
    base_url = "http://localhost:2233"
    
    # 测试根端点
    print("测试API是否运行...")
    response = requests.get(base_url)
    if response.status_code == 200:
        print("API正常运行")
    else:
        print(f"API测试失败，状态码: {response.status_code}")
        return
    
    # 测试分析端点
    print("\n测试分析功能...")
    
    test_texts = [
        "I like to think about problems by myself, and I often come up with innovative solutions. I value logic and efficiency, and I don't like spontaneous social activities.",
        "I always feel the emotions of others, like to work in a team, and helping others solve problems makes me feel satisfied.",
        "I like to have a structured plan and structure, completing tasks on time is very important to me. I don't like sudden situations."
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n测试文本 {i+1}:")
        print(text)
        
        data = {"text": text}
        try:
            response = requests.post(f"{base_url}/analyze", json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"检测到的MBTI类型: {result['mbti']}")
                print(f"解释摘要: {result['explanation'][:100]}...")
            else:
                print(f"请求失败，状态码: {response.status_code}")
                print(f"错误信息: {response.text}")
        except Exception as e:
            print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    test_api() 