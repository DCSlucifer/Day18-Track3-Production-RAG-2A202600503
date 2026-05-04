# Failure Analysis - Lab 18: Production RAG

**Sinh viên:** Võ Thành Danh - 2A202600503  
**Ngày cập nhật:** 04/05/2026  
**Nguồn số liệu:** `reports/ragas_report.json` sau lần chạy `python main.py` gần nhất

---

## 1. Tổng Quan Kết Quả

| Metric | Naive Baseline | Production Pipeline | Delta |
|--------|----------------|---------------------|-------|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 0.9333 | 0.9625 | +0.0292 |
| Context Precision | 0.9625 | 0.9750 | +0.0125 |
| Context Recall | 0.9625 | 0.9750 | +0.0125 |

**Nhận định chính:** Production pipeline đạt kết quả tốt và cải thiện so với naive baseline ở answer relevancy, context precision và context recall. Bottom failure chủ yếu đến từ một câu placeholder trong test set, không phải từ lỗi pipeline.

---

## 2. Bottom Failure Quan Trọng Nhất

### Failure #1: Placeholder trong test set

| Trường | Nội dung |
|--------|----------|
| Question | hihi cả nhà tự tạo testset bằng cơm nhé |
| Ground truth | Không kịp tạo luôn |
| Production answer | Đoạn context về "Đánh giá và kiểm thử mô hình học máy", bắt đầu từ section "Test set là gì". |

**Metric chi tiết**

| Faithfulness | Answer Relevancy | Context Precision | Context Recall | Avg Score | Worst Metric |
|--------------|------------------|-------------------|----------------|-----------|--------------|
| 1.0000 | 0.2500 | 0.5000 | 0.5000 | 0.5625 | answer_relevancy |

**Error Tree walkthrough**

1. **Output đúng không?** Không. Câu hỏi và ground truth là placeholder, không phải một câu hỏi RAG hợp lệ.
2. **Context đúng không?** Một phần. Hệ thống retrieve context về test set/evaluation vì query có từ "testset", nhưng ground truth không chứa tri thức thật để đối chiếu.
3. **Query/retrieval OK không?** Retrieval hoạt động hợp lý theo keyword "testset", nhưng đầu vào test không hợp lệ.
4. **Root cause:** Test set có câu placeholder, làm RAGAS chấm thấp answer relevancy và kéo aggregate xuống.
5. **Suggested fix:** Thay câu này bằng một câu hợp lệ, ví dụ "Hybrid search là gì?" hoặc "Test set dùng để làm gì?", rồi chạy lại `python main.py`.

---

## 3. Các Câu Còn Lại

Phần lớn các câu còn lại đạt điểm rất cao. Ví dụ các câu "AI là gì?", "Machine learning là gì?", "Deep learning là gì?", "Dataset là gì?", "Prompt là gì?", "F1-score là gì?", "LLM là gì", "Vector database là gì?" đều có context và answer khớp tốt với corpus.

Trong `reports/ragas_report.json`, các failure sau placeholder có `avg_score=1.0000`; nghĩa là bản chất không phải failure thực sự. Hàm `failure_analysis()` vẫn liệt kê vì nó luôn lấy bottom-N, kể cả khi các câu còn lại đã đạt điểm tối đa.

---

## 4. Vì Sao Naive Baseline Vẫn Cao?

Naive baseline cao vì:

1. **Corpus nhỏ và sạch:** Chỉ có vài tài liệu Markdown, mỗi câu hỏi định nghĩa có câu trả lời rõ trong corpus.
2. **Câu hỏi gần với heading:** Nhiều câu hỏi có dạng "X là gì?", trùng trực tiếp với heading trong tài liệu.
3. **Baseline copy context:** `naive_baseline.py` lấy context đầu tiên làm answer, nên faithfulness thường cao.

Tuy vậy production vẫn cải thiện:

- Answer relevancy: 0.9333 -> 0.9625
- Context precision: 0.9625 -> 0.9750
- Context recall: 0.9625 -> 0.9750

---

## 5. Case Study Cho Presentation

**Case chọn:** Placeholder trong test set.

**Lý do chọn:** Đây là case duy nhất làm giảm điểm rõ rệt và không phản ánh lỗi kỹ thuật của pipeline.

**Error Tree**

1. **Output sai?** Có, nhưng vì input test không hợp lệ.
2. **Context đúng?** Hệ thống chọn context liên quan đến "test set", hợp lý theo keyword.
3. **Query rewrite/search có lỗi?** Không có bằng chứng lỗi search.
4. **Fix ở đâu?** Fix ở test data, không phải ở pipeline.

---

## 6. Hướng Tối Ưu

1. Thay câu placeholder bằng câu hỏi thật trước khi nộp.
2. Chạy lại `python main.py` để regenerate `ragas_report.json`, `naive_baseline_report.json` và `latency_breakdown.json`.
3. Nếu muốn chứng minh production vượt baseline rõ hơn, thêm challenge set đúng domain AI/RAG nhưng ít trùng heading hơn.
4. Giữ nguyên fallback behavior để pipeline chạy ổn định khi không bật OpenAI hoặc real RAGAS.

---

## 7. Kết Luận

Production pipeline hiện đạt kết quả tốt với faithfulness 1.0000, answer relevancy 0.9625, context precision 0.9750 và context recall 0.9750. Điểm yếu chính trong failure analysis hiện tại là dữ liệu test có một câu placeholder. Sau khi thay câu đó bằng câu hỏi hợp lệ, báo cáo sẽ chuyên nghiệp hơn và nhiều khả năng các metric aggregate còn tăng thêm.
