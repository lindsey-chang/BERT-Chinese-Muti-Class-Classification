import requests

# API接口的URL
url = 'http://localhost:5000//api/nlp'

# 要发送的数据
data = {
    'text': '开左车门'
}

# 发送POST请求
response = requests.post(url, json=data)

# 获取响应数据
result = response.json()

# 输出标签和分数
print(result)