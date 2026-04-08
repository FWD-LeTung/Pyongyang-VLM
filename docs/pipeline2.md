Khi logic xử lý ảnh tĩnh đã chạy mượt, việc chuyển sang video bản chất là thêm một lớp xử lý chuỗi thời gian (time-series) trước khi đưa ảnh vào VLM.

1. Frame Sampling (Lấy mẫu khung hình):

    * Một video 30 FPS (khung hình/giây) chứa rất nhiều dữ liệu dư thừa. Bạn không cần xử lý mọi frame. Hãy thiết lập lấy mẫu khoảng 3-5 FPS là đủ để bắt được các chuyển động thay đổi góc mặt/trang phục.

2. Multi-Object Tracking (Bắt buộc):

    * Sử dụng ByteTrack hoặc DeepSORT gắn phía sau YOLO.

    * Tác dụng: Thay vì hệ thống hiểu là có "100 ông mặc áo đỏ" trong 100 frame liên tiếp, Tracking giúp hệ thống hiểu đây chỉ là "Người mang ID #01 xuất hiện trong 100 frame".

3. Keyframe Selection (Tối ưu hóa VLM):

    * Đây là bí quyết để hệ thống video chạy nhanh: Không đưa cả 100 ảnh crop của ID #01 qua CLIP.

    * Viết logic chọn ra 3-5 khung hình đẹp nhất của ID #01 (dựa trên: bounding box to nhất, độ tự tin của YOLO cao nhất, hoặc ảnh ít bị mờ (motion blur) nhất). Chỉ đẩy những keyframe này qua Image Encoder để lưu vào database.