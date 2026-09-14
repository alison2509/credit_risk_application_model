1. Bài toán
Dự đoán xác suất một đơn vay sẽ vỡ nợ (target = 1), chỉ dùng thông tin có sẵn tại thời điểm khách hàng nộp hồ sơ. Phân loại nhị phân, dữ liệu mất cân bằng.

Điểm đặc thù của bộ dữ liệu: mô hình không chỉ cần chính xác mà còn phải ổn định theo thời gian. Vì vậy toàn bộ quy trình đều bám theo trục WEEK_NUM.

Nhãn target do Kaggle cung cấp sẵn, không tự xây dựng. Đây là một hạn chế: không biết chính xác ngưỡng quá hạn và cửa sổ quan sát mà Home Credit sử dụng.

2. Dữ liệu
Nguồn: Home Credit Credit Risk Model Stability

Chỉ dùng 4 file depth 0 (mỗi đơn vay một dòng), theo tinh thần "application model":

File	Nội dung
train_base.parquet	Bảng gốc: case_id, date_decision, WEEK_NUM, target
train_static_0_0.parquet	Thông tin hồ sơ, phần 1
train_static_0_1.parquet	Thông tin hồ sơ, phần 2
train_static_cb_0.parquet	Thông tin từ trung tâm tín dụng


Dùng bản .parquet để giữ nguyên kiểu dữ liệu từng cột, nhẹ hơn và đọc được theo từng lô, tránh đầy RAM.

Lấy mẫu: do giới hạn phần cứng, huấn luyện trên mẫu 100.000 đơn thay vì toàn bộ ~1,5 triệu đơn. Mẫu lấy theo tỷ lệ của từng tuần để giữ nguyên phân bố thời gian và tỷ lệ vỡ nợ của dữ liệu gốc.

3. Cách chạy
bash
pip install -r requirements.txt
python 01_Data_prep.py     # đọc, lấy mẫu, ghép bảng -> output/data_flat.parquet
python 02_Model_train      # WOE/IV, huấn luyện, đánh giá, quy đổi điểm
Trước khi chạy, sửa DATA_DIR ở đầu file 01_Data_prep.py cho trỏ đúng thư mục chứa 4 file parquet. Nếu máy hết bộ nhớ, giảm SAMPLE_SIZE xuống 50.000.

Kết quả sinh ra trong output/:

model_coefficients.csv

evaluation_results.csv

scorecard_points.csv

3 biểu đồ PNG

4. Phương pháp
Xử lý dữ liệu: cột ngày đổi thành số ngày chênh lệch so với date_decision; bỏ cột thiếu trên 95% và cột chỉ có một giá trị; chỉ giữ cột dạng số.

Chia dữ liệu: theo thời gian, không ngẫu nhiên. 80% số tuần đầu để huấn luyện, 20% tuần cuối để kiểm tra (out-of-time).

Binning & WOE: mỗi biến chia 5 nhóm theo phân vị, giá trị thiếu là một nhóm riêng, thay giá trị gốc bằng WOE = ln(%khách tốt / %khách xấu). Mốc chia và bảng WOE tính trên tập huấn luyện rồi áp sang tập kiểm tra.

Chọn biến: lấy 20 biến có IV cao nhất.

Mô hình: Logistic Regression. Sau khi đổi sang WOE, các biến đã cùng thang đo log-odds nên không cần chuẩn hóa.

Quy đổi điểm: PDO = 20, 600 điểm ứng với tỷ lệ tốt/xấu 50:1.

5. Kết quả
Chỉ số	Giá trị
auc_train	0.6846
auc_valid	0.7103
gini_valid	0.4205
ks_valid	0.3347
mean_weekly_gini	0.3931


Nhận xét:

Hiệu lực chấp nhận được. AUC 0.71 và KS 0.33 là hợp lý cho một application scorecard chỉ dùng dữ liệu tại thời điểm nộp hồ sơ, với 20 biến và mẫu 100.000 đơn. Trong thực tế ngân hàng, KS > 0.30 thường đủ điều kiện đưa vào vận hành.

Không có dấu hiệu overfit. AUC trên tập kiểm tra còn cao hơn tập huấn luyện (0.7103 so với 0.6846). Mô hình chỉ có 20 biến tuyến tính nên khả năng học thuộc rất thấp.

Mô hình ổn định theo thời gian. Slope dương (+0.0041) nghĩa là đường Gini theo tuần không đi xuống mà còn nhích lên nhẹ. Đây là điểm quan trọng nhất vì chỉ số chính thức của cuộc thi phạt rất nặng nếu độ dốc âm.

Gini theo tuần thấp hơn Gini tổng thể (0.3931 so với 0.4205). Điều này bình thường: khi tính trong phạm vi một tuần, mô hình chỉ phân biệt giữa các hồ sơ cùng thời điểm, khó hơn so với khi được hưởng lợi từ chênh lệch rủi ro giữa các tuần.
