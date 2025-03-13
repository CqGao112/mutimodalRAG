import chromadb
import base64
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import torch
from IPython.display import Image, display, Markdown
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI



def load_db():
    # 加载矢量化数据库
    chroma_client = chromadb.PersistentClient(path='/mnt/d/myProjects/datasets/multimodal_vdb/fashionpedia')
    # Instantiate the ChromaDB Image Loader
    image_loader = ImageLoader()
    # Instantiate CLIP embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLIP = OpenCLIPEmbeddingFunction(checkpoint="/mnt/d/myProjects/datasets/fashionpedia/open_clip_pytorch_model.bin",
                                     device=device)
    image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function=CLIP, data_loader=image_loader)
    return image_vdb

# 定义 数据库 查询函数
def query_db(image_vdb, query, results=5):
    results = image_vdb.query(
        query_texts=[query],
        n_results=results,
        include=['uris', 'distances'])
    return results



# Define a function whereby a user query and two images are passed to GPT4o
def format_prompt_inputs(data, user_query,num):
    inputs = {}

    # Add user query to the dictionary
    inputs['user_query'] = user_query

    # Get the first two image paths from the 'uris' list
    for i in range(num):
        image_path = data['uris'][0][i]
        # Encode the first image
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        inputs[f'image_data_{i+1}'] = base64.b64encode(image_data).decode('utf-8')

    return inputs





def chatbot_response(num, model, language, chat_input):
    image_vdb = load_db()
    # Running Retrieval and Generation
    results = query_db(image_vdb,chat_input, results=num)

    prompt_input = format_prompt_inputs(results, chat_input,num)

    template = [{"type": "text", "text": "{user_query}穿搭方面有什么建议"}]
    for i in range(num):
        template.append({"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data_"f"{i+1}"+"}"})


    if language == "中文":
        system_a = "你是一个乐于助人的时尚和造型助理。使用给定的图像上下文直接参考提供的图像部分来回答用户的问题。"
        system_b = " 保持更具对话性的语气，不要列出太多清单。请使用中文回答"
        # " 保持更具对话性的语气，不要列出太多清单。对突出显示、强调和结构使用markdown格式。请使用中文回答"),
    else:
        system_a = "You are a helpful fashion and styling assistant.Use the given image context to directly reference the provided image portion to answer the user's question."
        system_b = " Maintain a more conversational tone and avoid making too many lists. Please answer in English"
    image_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_a+system_b),
            (
                "user",
                template,
            ),
        ]
    )

    llm = ChatOpenAI(
        temperature=0.95,
        model="glm-4v",
        openai_api_key="your api",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )

    # Instantiate the Output Parser
    parser = StrOutputParser()

    # Define the LangChain Chain
    vision_chain = image_prompt | llm | parser

    response = vision_chain.invoke(prompt_input)
    images = results['uris'][0]
    return images, response