# 重排rerank模型如何下载

## 1. 下载方法

rerank的下载方法和大语言模型下载方法和途径是一样的

- ModelScope
- huggingface

具体的使用详见【大语言模型是如何下载.md】
- https://git.imooc.com/coding-920/RAG_full_stack_course_notebooks/src/master/install/%e5%a4%a7%e8%af%ad%e8%a8%80%e6%a8%a1%e5%9e%8b%e5%a6%82%e4%bd%95%e4%b8%8b%e8%bd%bd.md


可以在ModelScope和huggingface中进行模型搜索


## 2. rerank模型下载路径

bge-reranker-base

- https://hf-mirror.com/BAAI/bge-reranker-base
- https://huggingface.co/BAAI/bge-reranker-base
- https://www.modelscope.cn/models/BAAI/bge-reranker-base

``` shell
huggingface-cli download --resume-download BAAI/bge-reranker-base  --local-dir bge-reranker-base


git clone https://www.modelscope.cn/BAAI/bge-reranker-base.git

``` 