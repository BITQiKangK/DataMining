import requests
import json

def test_api():
    base_url = "http://localhost:8000"
    
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
        "我喜欢独自思考问题，经常会想到一些创新的解决方案。我重视逻辑和效率，不太喜欢即兴的社交活动。",
        "我总是能感受到别人的情绪，喜欢在团队中工作，帮助他人解决问题让我感到满足。",
        "我喜欢有条理的计划和结构，按时完成任务对我来说非常重要。我不太喜欢突发状况。"
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