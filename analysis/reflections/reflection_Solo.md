# Individual Reflection - Lab 18 Production RAG

**Họ và tên:** Võ Thành Danh  
**MSSV:** 2A202600503  
**Hình thức thực hiện:** Cá nhân  
**Ngày hoàn thành:** 04/05/2026

---

## 1. Tổng Quan Đóng Góp

Trong Lab 18, tôi thực hiện toàn bộ pipeline Production RAG theo hình thức cá nhân. Phạm vi công việc bao gồm triển khai các module M1-M5, tích hợp pipeline end-to-end, chạy đánh giá RAGAS, tạo report, phân tích failure cases, bổ sung latency breakdown và hoàn thiện reflection cá nhân.

Kết quả cuối cùng:

- Unit tests: 40/40 passed.
- Production faithfulness: 1.0000.
- Production answer relevancy: 0.9625.
- Production context precision: 0.9750.
- Production context recall: 0.9750.
- Enrichment pipeline đã được tích hợp.
- Latency breakdown đã được ghi vào `reports/latency_breakdown.json`.

---

## 2. Đóng Góp Kỹ Thuật Cụ Thể

### 2.1. Module 1 - Advanced Chunking

Tôi triển khai và kiểm thử các chiến lược chunking nâng cao:

- **Semantic chunking:** tách câu và nhóm các câu gần nghĩa, có fallback lexical similarity để chạy ổn định khi không tải được model embedding.
- **Hierarchical chunking:** tạo parent chunk và child chunk, retrieve trên child nhưng trả về parent để tăng ngữ cảnh.
- **Structure-aware chunking:** parse markdown headers để giữ nguyên cấu trúc section.
- **Compare strategies:** thống kê số chunk, độ dài trung bình, min/max length cho từng chiến lược.

Trong quá trình kiểm tra pipeline, tôi phát hiện lỗi `parent_id` bị trùng giữa các tài liệu khác nhau. Ví dụ, nhiều document đều có `parent_0`, khiến pipeline retrieve đúng child nhưng attach sai parent context. Tôi đã thêm test regression và sửa parent id bằng cách thêm prefix từ `source`, giúp mỗi parent id là duy nhất theo tài liệu.

### 2.2. Module 2 - Hybrid Search

Tôi triển khai hệ thống search gồm:

- Vietnamese segmentation bằng `underthesea.word_tokenize()` khi thư viện khả dụng.
- BM25 search trên token đã segment.
- Dense search bằng Sentence-Transformers/Qdrant, có fallback in-memory cosine search.
- Reciprocal Rank Fusion để hợp nhất BM25 và dense results.

Điểm tôi học được ở module này là hybrid search đặc biệt hữu ích với tiếng Việt và các câu hỏi định nghĩa. BM25 bắt tốt từ khóa chính xác, còn dense retrieval hỗ trợ các truy vấn gần nghĩa.

### 2.3. Module 3 - Reranking

Tôi triển khai reranker theo hướng production-friendly:

- Ưu tiên cross-encoder reranker nếu model đã có sẵn hoặc được bật download.
- Có fallback lexical reranker để unit tests và pipeline vẫn chạy được trong môi trường offline.
- Bổ sung benchmark latency với các chỉ số `avg_ms`, `min_ms`, `max_ms`.

Reranking giúp giảm nhiễu sau bước hybrid search, đặc biệt khi top-k ban đầu chứa nhiều chunk cùng chủ đề nhưng không trực tiếp trả lời câu hỏi.

### 2.4. Module 4 - RAGAS Evaluation

Tôi triển khai evaluation pipeline gồm:

- Load test set 20 câu hỏi.
- Chạy real RAGAS khi bật OpenAI key.
- Có deterministic fallback để test local không phụ thuộc API.
- Lưu `reports/ragas_report.json` gồm aggregate metrics, per-question metrics và failures.
- Tạo failure analysis theo error tree: faithfulness, answer relevancy, context precision, context recall.

Sau khi cập nhật OpenAI key hợp lệ, tôi chạy full real RAGAS và thu được:

| Metric | Production |
|--------|------------|
| Faithfulness | 1.0000 |
| Answer Relevancy | 0.9625 |
| Context Precision | 0.9750 |
| Context Recall | 0.9750 |

### 2.5. Module 5 - Enrichment Pipeline

Tôi triển khai các kỹ thuật enrichment:

- Summarization fallback bằng extractive summary.
- HyQA để sinh câu hỏi giả định mà chunk có thể trả lời.
- Contextual prepend để thêm mô tả tài liệu và chủ đề vào trước chunk.
- Metadata extraction để bổ sung topic, entities, category và language.
- Full enrichment pipeline trên danh sách chunks.

Enrichment được tích hợp trong `pipeline.py` trước bước indexing, đáp ứng phần bonus enrichment pipeline.

### 2.6. End-to-end Pipeline

Tôi tích hợp các module thành pipeline hoàn chỉnh:

```text
Load documents
  -> Hierarchical chunking
  -> Enrichment
  -> Hybrid indexing/search
  -> Reranking
  -> LLM generation
  -> RAGAS evaluation
  -> Failure analysis + latency report
```

Pipeline có thể chạy ở hai chế độ:

- **Offline/deterministic mode:** phục vụ unit tests và kiểm thử nhanh.
- **API-backed mode:** sử dụng OpenAI generation và real RAGAS khi bật các biến môi trường phù hợp.

---

## 3. Kiến Thức Học Được

### 3.1. RAG production cần nhiều hơn vector search

Trước lab này, cách hiểu đơn giản về RAG thường là chunk tài liệu, embedding, search và đưa context vào LLM. Khi triển khai đầy đủ, tôi nhận ra chất lượng RAG phụ thuộc vào nhiều tầng:

- Chunking quyết định đơn vị thông tin được lưu và truy xuất.
- Hybrid search quyết định khả năng tìm đúng tài liệu ở cả keyword và semantic level.
- Reranking quyết định chất lượng top-k cuối cùng.
- Prompt và generation quyết định câu trả lời có đúng format mong muốn hay không.
- Evaluation giúp xác định lỗi nằm ở retrieval hay generation.

### 3.2. Context precision/recall giúp tách lỗi retrieval với lỗi generation

Trong kết quả cuối, production đạt faithfulness 1.0000, answer relevancy 0.9625, context precision 0.9750 và context recall 0.9750. Điều này cho thấy retrieval và generation đều hoạt động tốt trên bộ test hiện tại. Điểm yếu còn lại chủ yếu đến từ chất lượng test data: câu đầu tiên vẫn là placeholder, làm bottom failure không phản ánh lỗi kỹ thuật thật của pipeline.

### 3.3. Evaluation thật có thể chậm và không ổn định

Real RAGAS evaluation mất 313681.00 ms cho 20 câu hỏi. Đây là bài học quan trọng về production: evaluation không nên nằm trong request path. Thay vào đó, nên chạy định kỳ, cache kết quả hoặc chạy bất đồng bộ.

### 3.4. Một lỗi nhỏ về metadata có thể làm sai cả pipeline

Lỗi `parent_id` collision là bài học kỹ thuật đáng nhớ nhất. Retrieval trên child có thể đúng, nhưng nếu mapping child -> parent sai, context đưa vào LLM vẫn sai. Điều này cho thấy metadata trong RAG không phải phần phụ; nó là contract quan trọng giữa các bước chunking, search và generation.

---

## 4. Khó Khăn Và Cách Giải Quyết

### Khó khăn 1: OpenAI key lỗi 401

Ban đầu, OpenAI key trong môi trường bị trả về lỗi 401, làm generation và real RAGAS không chạy được ổn định. Tôi xử lý bằng cách tách rõ hai chế độ:

- Mặc định pipeline có thể chạy deterministic/offline.
- Khi có key hợp lệ, bật API-backed run bằng biến môi trường.

Sau khi key được cập nhật, tôi chạy lại smoke test với `gpt-4o-mini` và nhận phản hồi `OK`, sau đó chạy full real RAGAS.

### Khó khăn 2: Parent context bị attach sai

Kết quả ban đầu cho thấy một số câu hỏi retrieve nhầm parent context dù search có vẻ trả về child đúng. Tôi truy nguyên và phát hiện parent id bị trùng giữa nhiều file. Cách giải quyết:

1. Viết test regression cho trường hợp hai document khác source nhưng cùng `parent_0`.
2. Sửa `_make_parent()` để parent id có prefix theo source.
3. Chạy lại pipeline và xác nhận context precision/context recall tăng lên 0.9750 trên report mới nhất.

### Khó khăn 3: Answer relevancy thấp dù câu trả lời đúng

Một số câu trả lời đúng nhưng dài hơn ground truth, ví dụ câu hỏi "Prompt là gì?" hoặc "Accuracy là gì?". RAGAS chấm answer relevancy thấp do câu trả lời có thêm chi tiết. Tôi phân tích đây là lỗi generation length control, không phải lỗi retrieval. Hướng cải thiện là thêm answer compressor cho câu hỏi định nghĩa.

---

## 5. Tự Đánh Giá

Tôi tự đánh giá mức hoàn thành: **5/5**.

Lý do:

- Hoàn thành đầy đủ M1-M5.
- Pipeline chạy end-to-end.
- Unit tests cuối cùng đạt 40/40.
- RAGAS report mới nhất có faithfulness 1.0000, answer relevancy 0.9625, context precision 0.9750, context recall 0.9750.
- Hoàn thành failure analysis, group report, reflection cá nhân.
- Hoàn thành đủ điều kiện bonus: faithfulness >= 0.85, enrichment integrated và latency breakdown.

---

## 6. Hướng Phát Triển Tiếp Theo

Nếu tiếp tục cải thiện hệ thống, tôi sẽ ưu tiên:

1. **Answer compressor:** Tự động rút gọn câu trả lời cho câu hỏi định nghĩa.
2. **Intent-aware prompt:** Dùng prompt khác nhau cho định nghĩa, so sánh, giải thích quy trình và câu hỏi số liệu.
3. **Adaptive context selection:** Dùng child-only context cho câu hỏi ngắn, parent context cho câu hỏi cần nhiều ngữ cảnh.
4. **Evaluation cache:** Cache RAGAS results để giảm chi phí và thời gian chạy lại evaluation.
5. **Production monitoring:** Lưu lại query, retrieved context, answer và latency để phân tích lỗi thực tế sau triển khai.

---

## 7. Kết Luận Cá Nhân

Lab này giúp tôi hiểu rõ hơn cách xây dựng một Production RAG System có thể kiểm chứng bằng số liệu. Phần quan trọng nhất tôi học được là phải tách bạch lỗi retrieval và lỗi generation. Một pipeline tốt không chỉ cần trả lời nghe hợp lý, mà còn cần context đúng, câu trả lời bám sát context, latency có thể đo được và failure cases được phân tích có hệ thống.
