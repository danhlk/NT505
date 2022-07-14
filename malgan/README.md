Sử dụng Python 3.6, pip==8.1.1<br>
Cài các phụ thuộc trong pip_requirements/, nên cài trong virtualenv để không bị xung đột với bên ngoài.

# Chuẩn bị dữ liệu cho blackbox

## Mặc định của blackbox.<br>
File: extract_features.py, đầu ra là 3 tệp dạng mảng tensor
- feature_vector_directory/benign/benign_feature_set.pk: lưu thông tin đặc trưng của tệp lành tính.
- feature_vector_directory/malware/malware_feature_set.pk: lưu thông tin đặc trưng tệp độc hại.
- feature_vector_mapping.pk: chứa tên từng cột đặc trưng đã trích xuất.

Tham số yêu cầu,
- "-m": đường dẫn thư mục chứa tệp thực thi, mặc định là Data/malware
- "-b": Data/benign
- "-o": thư mục chứa kết quả, mặc định là feature_vector_directory
- "-l": xem thông cụ thể trong quá trình chạy (debug/info)

Trích xuất toàn bộ Import Function và tên Section làm thuộc tính huấn luyện blackbox học. Thể hiện dưới dạng tính năng nhị phân 0 và 1.<br>

## Sử dụng dữ liệu nhị phân có sẵn ở ngoài
- Có thể sử dụng dạng dữ liệu nhị phân khác có sẵn (không nhất thiết phải được trích xuất từ file extract_trên) như [https://github.com/lzylucy/Malware-GAN-attack/blob/master/train_data.libsvm](https://github.com/lzylucy/Malware-GAN-attack/blob/master/train_data.libsvm) hoặc [https://github.com/yanminglai/Malware-GAN/blob/master/mydata.npz](https://github.com/yanminglai/Malware-GAN/blob/master/mydata.npz)<br>
- Cần phải tự phân chia thành hai tệp chứa đặc trưng của từng loại như mặc định.
Tự làm file feature_vector_mapping.pk, nếu như chỉ sinh mẫu vector thì không cần file này.<br>

**Mặc định**
```sh
python3 extract_features.py -l debug 
```

# Huấn luyện MalGAN
## Mặc định
File: main_malgan.py, đầu ra là
- adversarial_feature_vector_directory/adversarial_feature_set_" + str(date.today()) + '_' + str(args.detector) + ".pk (đi kèm với thuật toán mà blackbox sử dụng): chứa mẫu vector đối kháng bypass được blackbox, sẽ được dùng để ánh xạ ngược để lấy tên đặc trưng cụ thể (nếu mục tiêu cao hơn feature-level).
- results_with_epochs.csv: chứa kết quả mỗi lần chạy xong một MalGAN, gồm thông tin cấu hình, thuật toán, auc, acc, pre, recall, f1, recall_after (vector đột biến). 
- File model blackbox: lưu trong thư mục models 'models/' + str(self._bb.type.name) + '_' + str(date.today()) + '.h5'
- File mode GAN: lưu eppch có chỉ số loss thấp nhất, thu mục saved_models

Các tham số:
- "-z": số chiều vector nhiễu, mặc định là 10
- "-s": số lượng batch size, mặc định 32
- "-n": số vòng muốn train MalGAN, mặc định 100
- "-m": đường dẫn file đặc trưng malware, mặc định là malware_feature_set.pk
- "-b": benign_feature_set.pk
- "-o": thư mục chứa kết quả vector đối kháng, mặc định là 
adversarial_feature_vector_directory.
- "--detector": chỉ định thụât toán sử dụng blackbox, là tên các biến được khai báo trong dmalgan/detector.py, estimator chứa danh sách các thuật toán sử dụng cho Voting, Stacking (yêu cầu scikit-learn >= 0.22)
- Lớp ẩn của G và D đều gồm 2 lớp có 256 nút, hàm kích hoạt gồm RELU, ELU, LeakyRELU, tanh, sigmoid, mặc định là LeakyRELU cho lớp ẩn G và D.

Dữ liệu được chia theo tỉ lệ benign là 8/2 cho train và test blackbox, malware là 6/2/2 một phần 2 dùng huấn luyện G và D.

```sh
python main_malgan.py --detector StackingRandomForest -m feature_vector_directory/malware/malware_feature_set.pk -b feature_vector_directory/benign/benign_feature_set.pk -o adversarial_feature_vector_directory -l debug
```

Sau khi chạy xong có thể xem kết quả trong file results_with_epochs.csv 

Đối với các mẫu vector đột biến, sử dụng binary_builder.py để ánh xạ ngược lấy lại thông tin ban đầu của đặc trưng.

```sh
python binary_builder.py -l debug -m <đường dẫn đến một mẫu malware> -a <đường dẫn đến file vector đột biến> -f <đường dẫn file chứa thông tin ban đầu> -l debug
```

Kết quả lưu trong thư mục Mutated_Binaries