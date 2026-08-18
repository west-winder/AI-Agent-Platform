import math


def cosine_similarity(a, b):
    """
    计算两个向量之间的 Cosine Similarity。

    参数：
        a: 第一个向量
        b: 第二个向量

    返回：
        两个向量的余弦相似度
    """

    # 计算两个向量的点积
    dot_product = sum(x * y for x, y in zip(a, b))

    # 计算向量 A 的长度
    norm_a = math.sqrt(sum(x * x for x in a))

    # 计算向量 B 的长度
    norm_b = math.sqrt(sum(y * y for y in b))

    # 防止出现除以 0
    if norm_a == 0 or norm_b == 0:
        return 0.0

    # Cosine Similarity
    return dot_product / (norm_a * norm_b)


# ==================================================
# 模拟三个向量
# ==================================================

# 假设这是两个非常相似的文本对应的向量
vector_a = [1, 2, 3]

vector_b = [1.1, 2.1, 3.1]

# 假设这是一个与 A 完全相反方向的向量
vector_c = [-1, -2, -3]

# 假设这是一个方向差异很大的向量
vector_d = [3, -2, 1]


# ==================================================
# 计算相似度
# ==================================================

similarity_ab = cosine_similarity(vector_a, vector_b)
similarity_ac = cosine_similarity(vector_a, vector_c)
similarity_ad = cosine_similarity(vector_a, vector_d)


# ==================================================
# 输出结果
# ==================================================

print("=" * 50)
print("Cosine Similarity 实验")
print("=" * 50)

print(f"A vs B: {similarity_ab:.4f}")
print(f"A vs C: {similarity_ac:.4f}")
print(f"A vs D: {similarity_ad:.4f}")

print("=" * 50)