import os

import bs4
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 模型
model = ChatOpenAI(
    model='Qwen/Qwen3-Omni-30B-A3B-Instruct',
    base_url=os.environ.get('OPENAI_BASE_URL'),
    api_key=os.environ.get('OPENAI_API_KEY')
)
# 实例化一个向量数据库
vector_store = Chroma(
    collection_name='t_agent_blog',
    embedding_function=OpenAIEmbeddings(
        model='Qwen/Qwen3-Embedding-8B',
        base_url=os.environ.get('OPENAI_BASE_URL'),
        api_key=os.environ.get('OPENAI_API_KEY')
    ),
    persist_directory='./chroma_qwen'
)


def create_dense_db():
    loader = WebBaseLoader(web_paths=[
        "https://lilianweng.github.io/posts/2023-06-23-agent/"
    ],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_=('post-header', 'post-title', 'post-content'))
        )
    )
    docs = loader.load()
    # 文本切割
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",  # GPT-4 使用的编码器，最适合英文
        chunk_size=1000,  # 每个 chunk 的 token 数
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    print(f"分块数量：{len(splits)}")
    ids = ['id' + str(i + 1) for i in range(len(splits))]
    vector_store.add_documents(splits, ids=ids)


# create_dense_db()


### 问题上下文化 ###
# 系统提示词：用于将带有聊天历史的问题转化为独立问题
contextualize_q_system_prompt = (
    "给定聊天历史和最新的用户问题（可能引用聊天历史中的上下文），"
    "将其重新表述为一个独立的问题（不需要聊天历史也能理解）。"
    "不要回答问题，只需在需要时重新表述问题，否则保持原样。"
)

# 创建聊天提示模板
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),  # 系统角色提示
        MessagesPlaceholder("chat_history"),  # 聊天历史占位符
        ("human", "{input}"),  # 用户输入占位符
    ]
)

# 创建一个向量数据库的检索器
retriever = vector_store.as_retriever(search_kwargs={'k': 2})

# 创建一个上下文感知的检索器 (检索外部知识以及历史上下文)
history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)

### 回答问题 ###
# 系统提示词：定义助手的行为和回答规范
system_prompt = (
    "你是一个问答任务助手。"
    "使用以下检索到的上下文来回答问题。"
    "如果不知道答案，就说你不知道。"
    "回答最多三句话，保持简洁。"
    "\n\n"
    "{context}"  # 从向量数据库中检索出来的doc
)
# 创建问答提示模板
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),  # 系统角色提示
        MessagesPlaceholder("chat_history"),  # 聊天历史占位符
        ("human", "{input}"),  # 用户输入占位符
    ]
)

# 创建文档处理链
question_chain = create_stuff_documents_chain(model, qa_prompt)

# 最终的RAG链
rag_chain = create_retrieval_chain(history_aware_retriever, question_chain)

store = {}


def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


final_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

res = final_chain.invoke(
    {
        "input": "What is Task Decomposition?"
    },
    config={"configurable": {"session_id": "user123"}}
)

res = final_chain.invoke(
    {
        "input": "刚才我提的什么问题？"
    },
    config={"configurable": {"session_id": "user123"}}
)
print(res["answer"])
