import pickle
import random
from tabulate import tabulate

# Đọc dữ liệu từ file domains_spoof.pkl
with open('process_spoof.pkl', 'rb') as file:
    data = pickle.load(file)

# Lấy dữ liệu từ trường 'test'
test_data = data['test']

# Lấy các đối tượng có nhãn 1 và 0
label_1_objects = [obj for obj in test_data if obj[2] == 1]
label_0_objects = [obj for obj in test_data if obj[2] == 0]

# Chọn ngẫu nhiên 10 đối tượng có nhãn 1
sample_label_1 = random.sample(label_1_objects, 10)

# Chọn ngẫu nhiên 10 đối tượng có nhãn 0
sample_label_0 = random.sample(label_0_objects, 10)

# Kết hợp các đối tượng đã chọn
sampled_objects = sample_label_1 + sample_label_0

# Tạo bảng để in dữ liệu
table_data = []
for obj in sampled_objects:
    table_data.append([obj[0], obj[1], obj[2]])

# In bảng dữ liệu
headers = ['Process', 'Spoof', 'Label']
print(tabulate(table_data, headers, tablefmt='grid'))