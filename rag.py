
import chromadb
import base64
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import torch
from IPython.display import Image, display, Markdown
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# 加载矢量化数据库
chroma_client = chromadb.PersistentClient(path='/mnt/d/myProjects/datasets/multimodal_vdb/fashionpedia')
# Instantiate the ChromaDB Image Loader
image_loader = ImageLoader()
# Instantiate CLIP embeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP = OpenCLIPEmbeddingFunction(checkpoint= "/mnt/d/myProjects/datasets/fashionpedia/open_clip_pytorch_model.bin",device=device)
image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function = CLIP, data_loader = image_loader)



# 定义 数据库 查询函数
def query_db(query, results=5):
    results = image_vdb.query(
        query_texts=[query],
        n_results=results,
        include=['uris', 'distances'])
    return results

llm = ChatOpenAI(
    temperature=0.95,
    model="glm-4v",
    openai_api_key="943c0af9738a4a76902024ed6079762a.xnlNhLsqjSxpdRSq",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# Instantiate the Output Parser
parser = StrOutputParser()
# Define the Prompt
image_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的时尚和造型助理。使用给定的图像上下文直接参考提供的图像部分来回答用户的问题。"
                    " 保持更具对话性的语气，不要列出太多清单。对突出显示、强调和结构使用markdown格式。请使用中文回答"),
        (
            "user",
            [
                {"type": "text", "text": "{user_query}穿搭方面有什么建议"},
                {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data_1}"},
                {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data_2}"},
            ],
        ),
    ]
)

# Define the LangChain Chain
vision_chain = image_prompt | llm | parser

# Define a function whereby a user query and two images are passed to GPT4o
def format_prompt_inputs(data, user_query):
    inputs = {}

    # Add user query to the dictionary
    inputs['user_query'] = user_query

    # Get the first two image paths from the 'uris' list
    image_path_1 = data['uris'][0][0]
    image_path_2 = data['uris'][0][1]

    # Encode the first image
    with open(image_path_1, 'rb') as image_file:
        image_data_1 = image_file.read()
    inputs['image_data_1'] = base64.b64encode(image_data_1).decode('utf-8')

    # Encode the second image
    with open(image_path_2, 'rb') as image_file:
        image_data_2 = image_file.read()
    inputs['image_data_2'] = base64.b64encode(image_data_2).decode('utf-8')

    return inputs

query = input("\n")

# Running Retrieval and Generation
results = query_db(query, results=2)
prompt_input = format_prompt_inputs(results, query)

response = vision_chain.invoke(prompt_input)

display(Markdown("---"))
# Showing Retrieved Pictures
display(Markdown("**Example Picture 1:**"))
display(Image(filename=results['uris'][0][0], width=300))
display(Markdown("**Example Picture 2:**"))
display(Image(filename=results['uris'][0][1], width=300))

display(Markdown("---"))
# Printing LLM Response
display(Markdown(response))