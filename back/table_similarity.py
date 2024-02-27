import math

def calculate_similarity(table1_size, table2_size):
    # 각 차원의 크기 구하기
    width_diff = abs(table1_size[0] - table2_size[0])
    height_diff = abs(table1_size[1] - table2_size[1])
    depth_diff = abs(table1_size[2] - table2_size[2])
    
    # 차원 간의 거리 계산
    distance = math.sqrt(width_diff**2 + height_diff**2 + depth_diff**2)
    
    # 유사도 계산
    similarity = 1 / (1 + distance)
    
    return similarity

# 두 개의 테이블의 크기 예시 (가로, 세로, 높이)
table1_size = (10, 20, 5)
table2_size = (12, 18, 6)

# 유사도 계산
similarity = calculate_similarity(table1_size, table2_size)
print("두 테이블 간의 유사도:", similarity)
