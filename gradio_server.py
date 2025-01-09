import gradio as gr
from my_milvus_client import MyMilvusClient

title = "text-image-search"
description="This demo allows you to search for images by typing a query and selecting the number of results."


milvus_client = MyMilvusClient()

def search_image(query, num_results):
    # 搜索得到 top_k 结果: [image_path1, image_path2, ...]
    results = milvus_client.search_image(query, top_k=num_results)
    # 返回最优结果列表
    if results:
        return results
    return ["No results found."]

# 定义输入文本框
textbox = gr.Textbox(label="Type your query here:", placeholder="dog", lines=1)

# 定义返回图片数量的滑块
slider = gr.Slider(label="Number of results", minimum=3, maximum=12, step=1)

# 定义输出为图片类型
interface = gr.Interface(
    fn=search_image,
    inputs=[textbox, slider],
    outputs=gr.Gallery(label="Search Results"),  # 使用 Gallery 显示多张图片
    title=title,
    description=description
)

interface.launch()
