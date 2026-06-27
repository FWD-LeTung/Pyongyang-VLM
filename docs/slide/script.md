# Kịch bản slide: Sử dụng VLM-LLM để tìm kiếm người

## Slide 1: Tiêu đề

**Sử dụng VLM-LLM để tìm kiếm người bằng mô tả ngôn ngữ tự nhiên**

Nội dung trình bày:

Bài toán tìm kiếm người không chỉ là nhận diện bằng ảnh, mà là tìm một người trong tập ảnh hoặc video dựa trên mô tả như: "người mặc áo đỏ, đeo ba lô đen, đi giày trắng". Hướng nghiên cứu chính là Text-based Person Retrieval, kết hợp Computer Vision và NLP.

## Slide 2: Động lực bài toán

Ý chính:

- Trong thực tế, người dùng thường không có ảnh mẫu của đối tượng.
- Truy vấn bằng văn bản tự nhiên phù hợp với tình huống nhân chứng, an ninh, tìm kiếm trong video.

Lời nói:

Điểm quan trọng của bài toán là chuyển từ việc tìm bằng ảnh sang tìm bằng mô tả. Điều này làm hệ thống gần hơn với cách con người ghi nhớ và trao đổi thông tin.

## Slide 3: Bài toán nằm ở đâu trong Person Retrieval

So sánh ngắn:

- Image-based Re-ID: query là ảnh người.
- Attribute-based Search: query là nhãn cố định như giới tính, màu áo.
- Text-based Person Retrieval: query là câu mô tả tự nhiên.
- Sketch-based Re-ID: query là hình phác thảo.

Thông điệp:

TBPR linh hoạt hơn attribute-based search, nhưng khó hơn vì phải đối khớp giữa hai miền dữ liệu khác nhau: ảnh và văn bản.

## Slide 4: Thách thức cốt lõi

Ý chính:

- Modality gap: ảnh là dữ liệu trực quan, văn bản là dữ liệu trừu tượng.
- Một câu mô tả không thể bao phủ toàn bộ thông tin trong ảnh.
- Cách con người mô tả có thể mơ hồ: "xanh lá", "xanh ngọc", "áo tối màu".
- Cần học một không gian nhúng chung giữa ảnh và text.

Lời nói:

Cốt lõi của bài toán là đưa embedding ảnh và embedding văn bản vào cùng một không gian, sau đó so sánh bằng cosine similarity hoặc các độ đo tương tự.

## Slide 5: Tiến hóa kỹ thuật

Timeline:

- Trước 2017: tìm kiếm bằng thuộc tính cố định.
- 2017: CUHK-PEDES và GNA-RNN đưa TBPR thành bài toán độc lập.
- 2018-2021: CNN/RNN, attention, BERT, fine-grained matching.
- 2022-2024: CLIP và vision-language pretraining giúp tăng mạnh hiệu quả.
- 2025-2026: MLLM, ChatPR, truy vấn hội thoại, làm giàu chú thích.

Thông điệp:

Lĩnh vực đang chuyển từ "so khớp ảnh-text" sang "hiểu và tương tác đa phương thức".

## Slide 6: Các hướng mô hình nổi bật

Nội dung:

- Dual encoder: image encoder và text encoder tạo embedding riêng.
- Attention/fine-grained alignment: căn chỉnh từ hoặc cụm từ với vùng cơ thể.
- CLIP-based methods: tận dụng tri thức tiền huấn luyện ảnh-văn bản.
- AUL: xử lý độ không chắc chắn và nhiễu trong cặp ảnh-text.
- MARS/attribute-relation methods: chú ý nhiều hơn đến thuộc tính và quan hệ giữa chi tiết.

Thông điệp:

Mô hình tốt không chỉ cần biết "áo đỏ", mà còn phải phân biệt chi tiết nhỏ như phụ kiện, kiểu dáng, vị trí và quan hệ giữa các thuộc tính.

## Slide 7: Dataset và benchmark

Các dataset chính:

- CUHK-PEDES: dataset nền tảng cho TBPR.
- RSTPReid: gần thực tế hơn với ánh sáng, góc nhìn camera.
- ICFG-PEDES: quy mô lớn hơn, có mô tả chi tiết hơn.
- UFine6926/UFineBench: mô tả siêu chi tiết, phục vụ tìm kiếm hạt mịn.
- ChatPedes: hướng dữ liệu hội thoại đa vòng.

Thông điệp:

Dataset đang phát triển theo hướng mô tả dài hơn, chi tiết hơn.

## Slide 8: Vấn đề annotation-induced mismatch

Ý chính:

- Nhiều người có ngoại hình gần giống nhau.
- Annotator có thể viết mô tả giống nhau cho các ID khác nhau.
- Mô hình trả về ảnh hợp mô tả nhưng khác ID vẫn bị tính sai.
- MLLM có thể giúp viết lại mô tả để nhấn mạnh đặc trưng phân biệt.

Lời nói:

Điểm này rất quan trọng vì lỗi không chỉ nằm ở mô hình, mà còn nằm ở cách dataset được chú thích và đánh giá.

## Slide 9: Hạn chế khi đưa vào thực tế

Ý chính:

- Mô tả người dùng mơ hồ.
- Ảnh hoặc video có che khuất, góc nhìn xấu, ánh sáng kém.
- Tìm kiếm trong gallery lớn cần tốc độ cao.
- Mô hình lớn như Transformer hoặc MLLM khó dùng trực tiếp ở thời gian thực.

Thông điệp:

Benchmark tốt chưa đồng nghĩa với hệ thống thực tế tốt. Project cần chọn phạm vi vừa sức.

## Slide 10: Hướng đề tài đề xuất

Tên hướng:

**Nghiên cứu hệ thống tìm kiếm người bằng mô tả văn bản sử dụng VLM-LLM**

Định hướng từ tài liệu research:

- Lấy Text-based Person Retrieval làm bài toán trung tâm.
- Sử dụng VLM/CLIP để học không gian nhúng chung giữa ảnh người và mô tả văn bản.
- Tận dụng LLM/MLLM để xử lý truy vấn mơ hồ, làm giàu mô tả và hỗ trợ tương tác.
- Tập trung vào các vấn đề thực tế: modality gap, mô tả không nhất quán, che khuất, góc nhìn xấu và tốc độ tìm kiếm.

Hướng triển khai nghiên cứu:

1. Khảo sát và so sánh các hướng TBPR truyền thống với các hướng dựa trên CLIP/VLM.
2. Xây dựng mô hình đối khớp ảnh-văn bản dựa trên joint embedding space.
3. Đánh giá trên các benchmark TBPR như CUHK-PEDES, RSTPReid, ICFG-PEDES hoặc UFine6926 nếu có điều kiện.
4. Phân tích các lỗi thường gặp: nhầm người có trang phục giống nhau, mô tả thiếu chi tiết, ảnh bị che khuất.
5. Đề xuất bổ sung LLM/MLLM để viết lại truy vấn, làm rõ mô tả hoặc hỏi lại người dùng trong hướng ChatPR.

## Slide 11: Vai trò của LLM trong project

Ý chính:

- Chuẩn hóa truy vấn người dùng.
- Tách thuộc tính: màu áo, quần, phụ kiện, hành động.
- Viết lại query rõ hơn.
- Hỏi lại khi mô tả mơ hồ: "Người đó có đeo balo không?"
- Làm giàu chú thích bằng cách nhấn mạnh các đặc trưng phân biệt giữa những người có ngoại hình tương tự.
- Hỗ trợ hướng ChatPR, nơi hệ thống không chỉ nhận một câu truy vấn tĩnh mà có thể tương tác đa vòng với người dùng.

Thông điệp:

VLM dùng để hiểu ảnh-text, còn LLM giúp giao tiếp, làm rõ truy vấn và tăng chất lượng mô tả.

## Slide 12: Phạm vi khả thi cho bài tập lớn

Phạm vi nên tập trung:

- Tìm hiểu bài toán Text-based Person Retrieval và vị trí của nó trong Person Retrieval/Re-ID.
- Trình bày cơ chế chung: image encoder, text encoder, joint embedding space và similarity ranking.
- Khảo sát vai trò của CLIP/VLM trong việc giảm chi phí học từ đầu và cải thiện đối khớp ảnh-văn bản.
- Phân tích các hướng mới: AUL, MARS, MLLM-based annotation refinement và ChatPR.
- Đề xuất mô hình thực nghiệm ở mức prototype dựa trên hướng CLIP/VLM, nếu thời gian môn học cho phép.

Mở rộng nghiên cứu nếu còn thời gian:

- Xử lý truy vấn mơ hồ bằng LLM.
- Thử nghiệm mô tả chi tiết hơn theo hướng UFineBench.
- Phân tích bài toán che khuất trong Text-based Occluded Person Re-ID.
- Mô phỏng tương tác ChatPR: hệ thống hỏi lại để làm rõ đặc điểm phân biệt.

## Slide 13: Kết luận

Ý chính:

- TBPR là bài toán quan trọng vì gần với nhu cầu tìm kiếm thực tế.
- Xu hướng hiện nay là kết hợp VLM, CLIP, MLLM và tương tác hội thoại.
- Khó khăn chính nằm ở modality gap, mô tả mơ hồ, che khuất và tốc độ.
- Đề tài nên tập trung vào hướng VLM/CLIP cho đối khớp ảnh-văn bản, sau đó mở rộng bằng LLM/MLLM để xử lý truy vấn mơ hồ và tương tác hội thoại.

Câu chốt:

Hướng tiếp cận phù hợp cho bài tập lớn là nghiên cứu hệ thống tìm kiếm người bằng mô tả văn bản, trong đó VLM đảm nhiệm phần đối khớp ảnh-văn bản và LLM hỗ trợ hiểu, chuẩn hóa, làm rõ truy vấn của người dùng.
