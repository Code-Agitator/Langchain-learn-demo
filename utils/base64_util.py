import base64


def audio_to_base64(file_path):
    """将音频文件转换为 base64 字符串"""
    with open(file_path, "rb") as audio_file:
        encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
    return encoded_string