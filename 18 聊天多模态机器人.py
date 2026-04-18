import os

import gradio as gr
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import create_async_engine

# 模型
model = ChatOpenAI(
    model='Qwen/Qwen3-Omni-30B-A3B-Instruct',
    base_url=os.environ.get('OPENAI_BASE_URL'),
    api_key=os.environ.get('OPENAI_API_KEY')
)

# 提示词
prompt = ChatPromptTemplate.from_messages([
    # 系统提示
    ('system', '你是一个乐于助人的助手。尽你所能回答所有的问题。提供的聊天历史包含与你对话用户的相关信息'),
    # 聊天历史
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    # 用户输入
    ('human', '{input}')
])

# 聊天历史存储
# 使用异步调用
async_engine = create_async_engine("sqlite+aiosqlite:///data/chat_history.db")


def get_session_history_sql(session_id):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=async_engine
    )


# 聊天历史摘要: 保留最后两条历史
def summarize_context(current_input):
    session_id = current_input.get('config', {}).get("configurable", {}).get('session_id')
    if not session_id:
        raise ValueError('session_id is required')
    chat_history = get_session_history_sql(session_id)
    stored_messages = chat_history.messages
    if len(stored_messages) <= 2:
        return {"original_messages": stored_messages, "summary": None}  # 不满足摘要条件时返回原始消息
    print("开始摘要历史对话")
    last_two_messages = stored_messages[-2:]
    messages_to_be_summarized = stored_messages[:-2]

    summarize_prompt = ChatPromptTemplate.from_messages([
        ('system', '请将下面消息进行剪辑和摘要，并返回结果'),
        MessagesPlaceholder(variable_name='{chat_history}'),
        ('human', '请生成包含上述对话核心内容的摘要，暴露重要的事实和决策'),
    ])

    summarize_chain = summarize_prompt | model
    summary_message = summarize_chain.invoke({
        'chat_history': messages_to_be_summarized,
    })

    return {
        "original_messages": last_two_messages,  # 保留的原始消息
        "summary": summary_message  # 生成的摘要
    }


# 创建带有聊天历史的chain
final_chain = RunnableWithMessageHistory(
    prompt | summarize_context | model,
    get_session_history_sql,
    input_messages_key='input',
    history_messages_key='chat_history'
)


def add_message(chat_history, user_message):
    if user_message:
        chat_history.append({"role": "user", "content": user_message})
    return chat_history, gr.Textbox(value=None, interactive=False)
def execute_chain(chat_history):
    input = chat_history[-1]
    result = final_chain.invoke({'input': input['content'], "config": {"configurable": {"session_id": "user123"}}},
                                            config={"configurable": {"session_id": "user123"}})
    chat_history.append({'role': 'assistant', 'content': result.content})
    return chat_history

# 开发一个聊天机器人的Web界面
with gr.Blocks(title='多模态聊天机器人', theme=gr.themes.Soft()) as block:

    # 聊天历史记录的组件
    chatbot = gr.Chatbot(type='messages', height=500, label='聊天机器人')

    with gr.Row():

        # 文字输入的区域
        with gr.Column(scale=4):
            user_input = gr.Textbox(placeholder='请给机器人发送消息...', label='文字输入', max_lines=5)

            submit_btn = gr.Button('发送', variant="primary")

        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=['microphone'], label='语音输入', type='filepath', format='wav')


    chat_msg = user_input.submit(add_message, [chatbot, user_input], [chatbot, user_input])
    chat_msg.then(execute_chain, chatbot, chatbot)


if __name__ == '__main__':
    block.launch()