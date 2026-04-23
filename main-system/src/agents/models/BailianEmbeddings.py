import dashscope
from http import HTTPStatus


class BailianEmbeddings:
    """阿里云百炼嵌入类"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        dashscope.api_key = api_key

    def embed_documents(self, texts):
        """批量嵌入文档（分批处理，每批不超过10个）"""
        all_embeddings = []
        batch_size = 10  # API限制每批不超过10个

        try:
            # 分批处理
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                resp = dashscope.TextEmbedding.call(
                    model="text-embedding-v4",
                    input=batch
                )

                if resp.status_code == HTTPStatus.OK:
                    batch_embeddings = [item['embedding'] for item in resp.output['embeddings']]
                    all_embeddings.extend(batch_embeddings)
                else:
                    raise Exception(f"API调用失败: {resp}")

            return all_embeddings

        except Exception as e:
            print(f"批量嵌入异常，降级为逐个嵌入: {e}")
            return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str):
        """单个查询嵌入"""
        try:
            resp = dashscope.TextEmbedding.call(
                model="text-embedding-v4",
                input=text
            )

            if resp.status_code == HTTPStatus.OK:
                return resp.output['embeddings'][0]['embedding']
            else:
                raise Exception(f"API调用失败: {resp}")

        except Exception as e:
            print(f"单个嵌入异常: {e}")
            return [0.0] * 1024
