# RAG, Embedding và Vector Database

## RAG là gì

RAG là kỹ thuật kết hợp truy xuất thông tin với mô hình sinh văn bản để tạo câu trả lời dựa trên nguồn dữ liệu bên ngoài. Pipeline RAG cơ bản gồm các bước: chunking tài liệu, tạo embedding, lưu vào vector database, truy xuất theo câu hỏi và sinh câu trả lời với LLM. RAG giúp mô hình trả lời chính xác hơn, cập nhật thông tin mới và giảm hallucination.

## Embedding là gì

Embedding là cách biểu diễn văn bản, hình ảnh hoặc dữ liệu khác dưới dạng vector số để máy tính có thể xử lý. Các embedding tốt phản ánh ngữ nghĩa: hai văn bản có ý nghĩa gần nhau sẽ có vector gần nhau trong không gian embedding. Các mô hình embedding phổ biến gồm BAAI/bge-m3 (đa ngôn ngữ), text-embedding-3 của OpenAI và Sentence-Transformers.

## Vector database là gì

Vector database là cơ sở dữ liệu chuyên lưu trữ và tìm kiếm các vector embedding. Khác với cơ sở dữ liệu truyền thống tìm theo khóa chính xác, vector database thực hiện tìm kiếm tương đồng (similarity search) sử dụng các thuật toán như HNSW hoặc IVF. Các vector database phổ biến gồm Qdrant, Pinecone, Weaviate, Milvus và pgvector.

## Hybrid search

Hybrid search kết hợp tìm kiếm dựa trên từ khóa (BM25) với tìm kiếm vector dày đặc (dense retrieval) để tận dụng ưu điểm của cả hai. BM25 mạnh ở các truy vấn có thuật ngữ chính xác, trong khi dense retrieval tốt với truy vấn ngữ nghĩa. Reciprocal Rank Fusion là một thuật toán phổ biến để gộp kết quả của hai phương pháp này.

## Reranking

Reranking là bước sàng lọc thứ hai sau truy xuất ban đầu, sử dụng mô hình cross-encoder để tính điểm liên quan chính xác hơn. Cross-encoder nhận cả câu truy vấn và tài liệu cùng lúc, cho điểm tương quan tốt hơn nhưng chậm hơn so với bi-encoder. Pipeline điển hình truy xuất top 20 rồi rerank xuống top 3 để cân bằng tốc độ và chất lượng.

## Chunking

Chunking là quá trình chia tài liệu lớn thành các đoạn nhỏ phù hợp với cửa sổ ngữ cảnh của mô hình embedding. Các chiến lược chunking gồm có: chia theo độ dài cố định, chia theo câu, chia theo ngữ nghĩa (semantic chunking), và chia phân cấp (parent-child). Chiến lược chunking ảnh hưởng đáng kể đến chất lượng RAG.
