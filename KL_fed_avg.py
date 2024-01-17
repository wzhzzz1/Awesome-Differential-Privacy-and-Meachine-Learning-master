import numpy as np

# 示例向量 A 和 B
vector_A = np.array([1, 2, 3, 4, 5])
vector_B = np.array([2, 3, 4, 5, 6])

# 计算余弦相似度
cosine_similarity = np.dot(vector_A, vector_B) / (np.linalg.norm(vector_A) * np.linalg.norm(vector_B))

print('余弦相似度:', cosine_similarity)