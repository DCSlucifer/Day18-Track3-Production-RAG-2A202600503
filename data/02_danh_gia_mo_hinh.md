# Đánh giá và kiểm thử mô hình học máy

## Test set là gì

Test set là tập dữ liệu dùng để đánh giá hiệu suất của mô hình sau khi huấn luyện. Test set phải hoàn toàn tách biệt với training data và validation set, không được dùng cho việc điều chỉnh tham số. Tỷ lệ test set thường chiếm 10-20% tổng dữ liệu, đảm bảo đánh giá khách quan khả năng tổng quát hóa của mô hình.

## Validation set dùng để làm gì

Validation set được dùng để điều chỉnh tham số và chọn mô hình tốt nhất trong quá trình phát triển. Khác với test set chỉ dùng một lần ở cuối, validation set có thể được dùng nhiều lần để so sánh các phiên bản mô hình. Kỹ thuật k-fold cross-validation là cách phổ biến để tận dụng dữ liệu khi validation set nhỏ.

## Overfitting là gì

Overfitting xảy ra khi mô hình học quá kỹ dữ liệu huấn luyện và hoạt động kém trên dữ liệu mới. Dấu hiệu là độ chính xác trên training data rất cao nhưng trên test set thấp. Cách phòng tránh overfitting bao gồm regularization, dropout, early stopping và tăng cường dữ liệu (data augmentation).

## Underfitting là gì

Underfitting xảy ra khi mô hình quá đơn giản và không học được đầy đủ quy luật từ dữ liệu. Mô hình underfitting có hiệu suất thấp trên cả training data lẫn test set. Khắc phục bằng cách tăng độ phức tạp của mô hình, thêm đặc trưng (feature engineering), hoặc huấn luyện lâu hơn.

## Accuracy là gì

Accuracy là tỷ lệ dự đoán đúng trên tổng số dự đoán của mô hình. Công thức tính accuracy là số dự đoán đúng chia cho tổng số mẫu. Tuy nhiên, accuracy không phù hợp với dữ liệu mất cân bằng vì có thể đạt giá trị cao chỉ nhờ dự đoán nhãn đa số.

## Precision là gì

Precision cho biết trong các kết quả được mô hình dự đoán là đúng, có bao nhiêu kết quả thật sự đúng. Công thức là true positives chia cho tổng số positive predictions. Precision quan trọng khi chi phí của dự đoán dương sai (false positive) cao, ví dụ trong chẩn đoán bệnh hoặc lọc thư rác.

## Recall là gì

Recall cho biết mô hình tìm được bao nhiêu trường hợp đúng trong tổng số trường hợp đúng thực tế. Công thức là true positives chia cho tổng số actual positives. Recall quan trọng khi chi phí bỏ sót cao, ví dụ trong phát hiện gian lận hoặc tầm soát ung thư.

## F1-score là gì

F1-score là chỉ số kết hợp giữa precision và recall, thường dùng khi dữ liệu bị mất cân bằng. F1-score được tính bằng trung bình điều hòa của precision và recall, theo công thức 2 × (precision × recall) / (precision + recall). Giá trị F1-score nằm trong khoảng từ 0 đến 1, càng gần 1 càng tốt.
