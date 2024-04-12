import numpy as np

# 예제: 100x100 기능적 연결성 매트릭스 생성 (임의의 데이터)
fc_matrix = np.random.rand(100, 100)

# 대각선을 0으로 설정합니다(자기 자신과의 연결 제외).
np.fill_diagonal(fc_matrix, 0)

# 상위 10% 및 20% 연결성 값을 유지하기 위한 임계값 계산
top_10_percentile_threshold = np.percentile(fc_matrix, 90) # 상위 10%의 임계값
top_20_percentile_threshold = np.percentile(fc_matrix, 80) # 상위 20%의 임계값

# 상위 10%의 연결성 값을 유지하고 나머지를 0으로 설정
fc_matrix_10 = np.where(fc_matrix < top_10_percentile_threshold, 0, fc_matrix)

# 상위 20%의 연결성 값을 유지하고 나머지를 0으로 설정
fc_matrix_20 = np.where(fc_matrix < top_20_percentile_threshold, 0, fc_matrix)

# 결과 확인 (예: 상위 10% 및 20% 임계값 출력)
print("상위 10% 임계값:", top_10_percentile_threshold)
print("상위 20% 임계값:", top_20_percentile_threshold)