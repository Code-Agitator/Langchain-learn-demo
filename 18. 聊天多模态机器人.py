import os

import gradio as gr
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import create_async_engine

from utils.base64_util import audio_to_base64

# 模型
model = ChatOpenAI(
    model='Qwen/Qwen3-Omni-30B-A3B-Instruct',
    base_url=os.environ.get('OPENAI_BASE_URL'),
    api_key=os.environ.get('OPENAI_API_KEY')
)

# 提示词
prompt = ChatPromptTemplate.from_messages([
    ('system', "你是一个乐于助人的助手。尽你所能回答所有问题。摘要：{summary}"),  # 动态注入系统消息
    MessagesPlaceholder(variable_name='chat_history', optional=True),
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


summary_len = 5


# 聊天历史摘要: 保留最后两条历史
async def summarize_context_v2(current_input):
    session_id = current_input.get('config', {}).get("configurable", {}).get('session_id')
    if not session_id:
        raise ValueError('session_id is required')
    chat_history = get_session_history_sql(session_id)
    stored_messages = await chat_history.aget_messages()
    if len(stored_messages) <= summary_len:
        return {"original_messages": stored_messages, "summary": None}
    print("开始摘要历史对话")
    last_two_messages = stored_messages[-summary_len:]
    messages_to_be_summarized = stored_messages[:-summary_len]

    summarize_prompt = ChatPromptTemplate.from_messages([
        ('system', '请将下面消息进行剪辑和摘要，并返回结果'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '请生成包含上述对话核心内容的摘要，暴露重要的事实和决策'),
    ])

    summarize_chain = summarize_prompt | model
    summary_message = await summarize_chain.ainvoke({
        'chat_history': messages_to_be_summarized,
    })
    return {
        "original_messages": last_two_messages,  # 保留的原始消息
        "summary": summary_message
    }


async def invoke_and_save(context):
    session_id = context.get('config', {}).get("configurable", {}).get('session_id')
    if not session_id:
        raise ValueError('session_id is required')
    messages_summarized = context.get('messages_summarized', {})
    summary = messages_summarized.get('summary', '无摘要')
    chat_history = messages_summarized.get('original_messages')
    user_input = context.get('input')
    invoke_chain = prompt | model
    print(prompt.invoke({
        'input': user_input,
        'chat_history': chat_history,
        'summary': summary
    }))
    result = await invoke_chain.ainvoke({
        'input': user_input,
        'chat_history': chat_history,
        'summary': summary
    })
    chat_history_sql = get_session_history_sql(session_id)
    await chat_history_sql.aadd_message(HumanMessage(content=user_input))
    await chat_history_sql.aadd_message(result)
    return result


# 创建带有聊天历史的chain
final_chain = (
        RunnablePassthrough.assign(messages_summarized=summarize_context_v2) |
        RunnablePassthrough.assign(
            input=lambda x: x['input'],
            chat_history=lambda x: x['messages_summarized']['original_messages']
        ) | invoke_and_save

)


def add_message(chat_history, user_message):
    if user_message:
        chat_history.append({"role": "user", "content": user_message})
    return chat_history, gr.Textbox(value=None)


async def execute_chain(chat_history):
    input = chat_history[-1]
    result = await final_chain.ainvoke(
        {'input': input['content'], "config": {"configurable": {"session_id": "user123"}}},
        config={"configurable": {"session_id": "user123"}})
    chat_history.append({'role': 'assistant', 'content': result.content})
    return chat_history


def read_audio(chatbox, audio_messages, user_input):
    audio_base64 = audio_to_base64(audio_messages)
    audio_to_text_chain_prompt = ChatPromptTemplate.from_messages([
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "data:audio/wav;base64,{audio_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": "这段音频的内容是什么？只输出音频内容文本"
                }
            ]
        }
    ])
    audio_to_text_chain = audio_to_text_chain_prompt | model | StrOutputParser()
    text_of_audio = audio_to_text_chain.invoke({"audio_base64": audio_base64})

    return chatbox, gr.Textbox(value=text_of_audio)


if __name__ == '__main__':
    # 开发一个聊天机器人的Web界面
    with gr.Blocks(title='多模态聊天机器人') as block:
        # 聊天历史记录的组件
        chatbot = gr.Chatbot(height=500, label='聊天机器人')

        with gr.Row():
            # 文字输入的区域
            with gr.Column(scale=4):
                user_input = gr.Textbox(placeholder='请给机器人发送消息...', label='文字输入', max_lines=5)

            with gr.Column(scale=1):
                audio_input = gr.Audio(sources=['microphone'], label='语音输入', type='filepath', format='wav')
                audio_input.change(read_audio, [chatbot, audio_input], [chatbot, user_input])

        chat_msg = user_input.submit(add_message, [chatbot, user_input], [chatbot, user_input])
        chat_msg.then(execute_chain, chatbot, chatbot)

        block.launch(theme=gr.themes.Soft())
