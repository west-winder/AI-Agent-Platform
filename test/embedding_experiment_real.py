from sentence_transformers import SentenceTransformer
import numpy as np


# ==================================================
# 1. 加载 Embedding Model
# ==================================================

print("正在加载 Embedding Model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("模型加载完成！")


# ==================================================
# 2. 准备测试文本
# ==================================================

texts = [
    "用户喜欢Python",
    "用户喜欢使用Python进行后端开发",
    "用户不喜欢Python",
    "今天深圳下雨了",
]


# ==================================================
# 3. 生成 Embedding
# ==================================================

embeddings = model.encode(texts)


# ==================================================
# 4. 查看向量信息
# ==================================================

print("\n" + "=" * 60)
print("Embedding 结果")
print("=" * 60)

for text, embedding in zip(texts, embeddings):

    print(f"\n文本：{text}")

    print(f"向量维度：{len(embedding)}")

    # 只显示前10个数字，避免输出太长
    print("向量前10维：")
    print(embedding[:10])


# ==================================================
# 5. Cosine Similarity
# ==================================================

def cosine_similarity(a, b):
    """
    计算两个向量之间的 Cosine Similarity。
    """

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# ==================================================
# 6. 两两比较
# ==================================================

print("\n" + "=" * 60)
print("Cosine Similarity")
print("=" * 60)

for i in range(len(texts)):

    for j in range(i + 1, len(texts)):

        similarity = cosine_similarity(
            embeddings[i],
            embeddings[j]
        )

        print(
            f"\n[{texts[i]}]"
            f"\nvs"
            f"\n[{texts[j]}]"
            f"\nSimilarity: {similarity:.4f}"
        )