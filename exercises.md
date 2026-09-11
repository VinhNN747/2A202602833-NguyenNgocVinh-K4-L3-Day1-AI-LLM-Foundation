# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**
Câu trả lời qua 4 lần gọi với temperature:
- 0.0: Việt Nam là quốc gia xuất khẩu hạt tiêu lớn nhất thế giới, chiếm khoảng 40% sản lượng toàn cầu.
- 0.5: Vịnh Hạ Long của Việt Nam đã được UNESCO công nhận là Di sản Thiên nhiên Thế giới hai lần, vào năm 1994 và 2000, nhờ vào vẻ đẹp thiên nhiên hùng vĩ và giá trị địa chất độc đáo.
- 1.0: Phở, món ăn truyền thống nổi tiếng của Việt Nam, đã được đưa vào từ điển Oxford vào năm 2007, công nhận sự phổ biến và tầm ảnh hưởng của món ăn này trên thế giới.
- 1.5: Hang Sơn Đoòng, nằm trong Vườn quốc gia Phong Nha - Kẻ Bàng, Quảng Bình, là hang động lớn nhất thế giới, có kích thước đủ để chứa cả một khu phố với những tòa nhà chọc trời.
**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> temperature càng cao, phản hồi càng đa dạng và sáng tạo hơn, trong khi temperature thấp dẫn đến câu trả lời chính xác và ít biến thể hơn.


### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature thấp, khỏang từ 0.2 đến 0.7 để đảm bảo phản hồi chính xác, nhất quán và chuyên nghiệp, tránh những câu trả lời quá sáng tạo hoặc không phù hợp với ngữ cảnh hỗ trợ khách hàng. 

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Ước tính chi phí:
> GPT-4o: 10.000 người * 3 lần/người * 350 token * $0.03/1K token = $31.50/ngày
> GPT-4o-mini: 10.000 người * 3 lần/người * 350 token * $0.01/1K token = $10.50/ngày
> GPT-4o đắt hơn GPT-4o-mini khoảng 3 lần.
> Trường hợp xứng đáng dùng GPT-4o: Khi cần phản hồi chất lượng cao, phức tạp, hoặc yêu cầu hiểu biết sâu về ngữ cảnh, ví dụ như tư vấn pháp lý hoặc y tế.
> Trường hợp nên dùng GPT-4o-mini: Khi cần phản hồi nhanh, đơn giản, và chi phí là yếu tố quan trọng, ví dụ như chatbot hỗ trợ khách hàng cơ bản hoặc FAQ.
---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với system prompt là giáo viên tiểu học, câu trả lời ngắn gọn, sử dụng từ vựng và các miêu tả đơn giản, dễ hình dung.
> Với system prompt là chuyên gia tài chính, câu trả lời dài hơn, sử dụng thuật ngữ kỹ thuật và giải thích chi tiết hơn về cơ chế hoạt động của blockchain. 
> System prompt định hướng cách model trả lời, ảnh hưởng đến độ dài, từ vựng và mức độ phức tạp của câu trả lời.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Số token theo tiktoken: 243
> Số token theo ước lượng: 100 / 0.75 = 133.33
> Chênh lệch: (243 - 133.33) / 133.33 * 100% ≈ 82.2%
> Tiếng Việt thường tốn nhiều token hơn tiếng Anh cùng độ dài vì tiếng Việt sử dụng nhiều từ ghép, dấu câu và ký tự đặc biệt, dẫn đến việc phân tách token phức tạp hơn. Ngoài ra, tiếng Việt có nhiều từ đa nghĩa và các biến thể ngữ pháp, làm tăng số lượng token cần thiết để biểu diễn cùng một ý nghĩa.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các trường hợp cần xử lý dữ liệu theo thời gian thực, như livestream, cập nhật giá chứng khoán, hoặc giám sát hệ thống. Non-streaming phù hợp hơn khi xử lý dữ liệu không yêu cầu tức thời, như phân tích dữ liệu hàng loạt hoặc báo cáo định kỳ.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff có lợi thế hơn delay cố định vì nó giúp giảm tải cho server một cách hiệu quả hơn. Bằng cách tăng dần thời gian chờ giữa các lần thử lại, nó giảm thiểu khả năng các yêu cầu từ hàng nghìn client cùng xảy ra đồng thời, giúp server có thời gian phục hồi. Nếu tất cả client đều retry với delay cố định giống nhau, có thể dẫn đến tình trạng "thundering herd", nơi server bị quá tải bởi quá nhiều yêu cầu cùng lúc, dẫn đến hiệu suất kém và có thể gây ra tình trạng sập hệ thống.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona tôi chọn là: "Bạn là một trợ lý ảo thông minh hỗ trợ nghiên cứu về AI, thân thiện và luôn cung cấp thông tin chính xác. Hãy trả lời các câu hỏi của người dùng một cách ngắn gọn, rõ ràng và bằng tiếng Việt."
> Lựa chọn từ ngữ quan trọng:
> - "trả lời ngắn gọn": Đảm bảo rằng các câu trả lời không quá dài, giúp người dùng dễ dàng tiếp nhận thông tin mà không bị quá tải.
> - "bằng tiếng Việt": Đảm bảo rằng trợ lý luôn sử dụng ngôn ngữ mà người dùng mong muốn, tạo sự thân thiện và dễ hiểu hơn cho người dùng Việt Nam.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của trợ lý là không có bộ nhớ dài hạn, dẫn đến việc không thể ghi nhớ các cuộc trò chuyện trước đó và cung cấp thông tin liên quan trong các tương tác tiếp theo.
> Cải thiện: Triển khai một cơ chế lưu trữ lịch sử trò chuyện trong cơ sở dữ liệu hoặc tệp tin, cho phép trợ lý truy xuất và tham khảo các cuộc trò chuyện trước đó. Khi người dùng bắt đầu một phiên mới, trợ lý có thể truy vấn lịch sử để cung cấp phản hồi phù hợp dựa trên ngữ cảnh đã biết.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
