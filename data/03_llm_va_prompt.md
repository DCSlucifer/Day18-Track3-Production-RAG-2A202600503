# Mô hình ngôn ngữ lớn và Prompt

## Prompt là gì

Prompt là câu lệnh hoặc nội dung đầu vào mà người dùng cung cấp cho mô hình AI để nhận phản hồi. Một prompt tốt thường bao gồm bối cảnh, yêu cầu cụ thể và định dạng đầu ra mong muốn. Kỹ năng viết prompt hiệu quả được gọi là prompt engineering, và là yếu tố quan trọng để khai thác sức mạnh của các mô hình AI hiện đại.

## LLM là gì

LLM là mô hình ngôn ngữ lớn, được huấn luyện trên lượng dữ liệu văn bản rất lớn để hiểu và tạo ngôn ngữ tự nhiên. Các LLM phổ biến hiện nay gồm GPT của OpenAI, Claude của Anthropic, Gemini của Google và LLaMA của Meta. LLM thường có hàng tỷ đến hàng nghìn tỷ tham số và sử dụng kiến trúc Transformer làm nền tảng.

## Hallucination trong AI

Hallucination là hiện tượng mô hình AI tạo ra thông tin sai hoặc không có căn cứ nhưng nghe có vẻ hợp lý. Đây là một trong những thách thức lớn nhất khi triển khai LLM trong môi trường doanh nghiệp. Các kỹ thuật giảm hallucination gồm có RAG (cung cấp ngữ cảnh từ nguồn tin cậy), grounding, fact-checking sau sinh và fine-tuning với dữ liệu chuyên ngành.

## Few-shot và zero-shot learning

Zero-shot learning là khả năng mô hình thực hiện một nhiệm vụ mới mà không cần ví dụ nào. Few-shot learning là khi mô hình được cung cấp một vài ví dụ trong prompt để hướng dẫn cách trả lời. Cả hai kỹ thuật này đều khai thác kiến thức tổng quát mà LLM đã học được trong giai đoạn pre-training.

## Temperature và sampling

Temperature là tham số điều khiển độ ngẫu nhiên của đầu ra LLM. Giá trị temperature thấp (gần 0) tạo ra đáp án nhất quán và tập trung, phù hợp cho tác vụ đòi hỏi chính xác. Giá trị cao (gần 1 hoặc hơn) tạo ra đáp án đa dạng và sáng tạo hơn, phù hợp cho viết sáng tạo nhưng dễ bị hallucination.
