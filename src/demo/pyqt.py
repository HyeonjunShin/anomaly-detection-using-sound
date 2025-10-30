# import matplotlib
# matplotlib.use('TkAgg')  # GUI 백엔드
# import matplotlib.pyplot as plt

# # x, y 데이터를 지정하여 라인 그래프 생성
# plt.plot([1, 2, 3, 4], [1, 4, 9, 16])

# # 그래프를 화면에 표시


import matplotlib
matplotlib.use("Qt5Agg")
print(matplotlib.get_backend())


import matplotlib.pyplot as plt
import numpy as np

# x축과 y축 좌표 정의
x = [1, 8]
y = [3, 10]

# 꺾은선 그래프 그리기
plt.plot(x, y)

# 그래프에 제목과 축 레이블 추가 (선택 사항)export QT_QPA_PLATFORM=xcb

plt.title("Simple Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# 그래프 표시
plt.show()
