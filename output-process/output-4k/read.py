import pickle

# Hàm này sẽ đọc và in nội dung của file 'domain_results.pkl'
def read_and_print_pickle_file(filename):
    try:
        # Mở file để đọc ('rb' là read binary mode)
        with open(filename, 'rb') as file:
            # Deserialize dữ liệu từ file
            data = pickle.load(file, encoding='latin1')
            # In dữ liệu đã đọc
            print(data)
    except FileNotFoundError:
        print(f"File '{filename}' không tìm thấy.")
    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc file: {e}")

# Gọi hàm với tên file thực tế
read_and_print_pickle_file('process_results.pkl')