# 简介
- 使用CLIP和Milvus搭建图片检索系统，可以使用文字检索图片、图片检索图片，或者同时文字+图片检索。

# 使用
step0: 修改config.py配置项

step1: 将图片存入milvus数据库


```shell
python my_milvus_client.py
```

step2: 启动gradio服务
```shell
python gradio_server.py
```

# 效果展示

https://github.com/user-attachments/assets/bed246ba-36c2-4a6a-896d-6e02002e7eba

