# %%
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
import os

from langserve import add_routes

model = ChatOpenAI(
    model='Qwen/Qwen3-Omni-30B-A3B-Instruct',
    base_url=os.environ.get('OPENAI_BASE_URL'),
    api_key=os.environ.get('OPENAI_API_KEY')
)

# 定义提示词模板
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages([
    ('system', '请将下面的内容翻译为{language}'),
    ('user', "{text}")
])

from langchain_core.output_parsers import StrOutputParser

# 定义结果解析器
parser = StrOutputParser()

# 定义链
chain = prompt_template | model | parser

print(chain.invoke({
    "text": "你好，请问你要去哪里?",
    "language": "日语"
}))

# 创建服务
app = FastAPI(
    title="LangChain-LangServer",
    description="LangChain-LangServer",
    version="0.1.0",
)

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088)
