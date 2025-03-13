import gradio as gr
from app_api import chatbot_response


def text2image_gr():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                num = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="参考图片")
                model = gr.Dropdown(choices=["智谱GLM-4V", "Deepseek V3", "Qwen2.5-VL"],
                                    value="智谱GLM-4V", label="模型选择")
                language = gr.Radio(choices=["中文", "English"], value="中文", label="语言选择")

            with gr.Column(scale=5):
                output_images = gr.Gallery(label="参考图片",columns=4, height=200)
                output_text = gr.Textbox(label="建议", interactive=False,lines=5)
        with gr.Row():
            chat_input = gr.Textbox(placeholder="请输入内容...",scale=8,lines=1,label="输入框")
            send_btn = gr.Button("发送",scale=1)
        inputs = [num, model, language, chat_input]
        send_btn.click(fn=chatbot_response, inputs=inputs, outputs=[output_images,output_text])

    return demo

if __name__ == "__main__":
    app = gr.TabbedInterface(
        interface_list=[text2image_gr()],  # 子界面列表
        tab_names=["穿搭小助手"],  # 标签名称
        title='穿搭小助手'
    )

    app.launch(server_name="0.0.0.0", server_port=5001)