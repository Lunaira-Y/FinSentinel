import requests

# ⚠️ 注意：这里填入一个刚刚在主爬虫里爬取失败（被拦截）的东方财富新闻详情页 URL
TARGET_URL = "https://guba.eastmoney.com/news,600030,1680290131.html"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://guba.eastmoney.com/"
}

print("🕵️ 正在向东方财富详情页发送侦察请求...")
response = requests.get(TARGET_URL, headers=headers)

print(f"状态码: {response.status_code}")
print("--- 服务器真实返回的前 800 个字符 ---")
# 直接打印原始文本，不经过任何 BeautifulSoup 过滤，寻找 JS 挑战或重定向的蛛丝马迹
print(response.text[:800])
print("--------------------------------------")
