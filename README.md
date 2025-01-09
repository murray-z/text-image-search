# 简介
- 使用CLIP和Milvus搭建图片检索系统

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
