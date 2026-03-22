from langserve import RemoteRunnable

client = RemoteRunnable("http://127.0.0.1:8088/chain")
print(client.invoke({
    "language": "法语",
    "text": "我要吃粑粑"
}))
