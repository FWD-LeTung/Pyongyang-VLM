# Giai đoạn 1: Pipeline xử lý ảnh Upload
Mục tiêu: User upload ảnh (ảnh nhóm), hệ thống sẽ phân tích ảnh đó và cho phép tìm kiếm người bằng văn bản

1. Nhận ảnh và phát hiện đối tượng 
    * Khi ảnh được upload, đẩy nó qua một mô hình phát hiện người như YOLO
    * yolo sẽ trả về bounding box các tọa độ của tất cả những người xuất hiện trong ảnh
2. Cắt và tiền xử lý
    * Dựa vào bounding box --> Person Crop
    * Áp dụng CLAHE, HE để cân bằng sáng
3. Trích xuất và lưu trữ 
    * Resize ảnh đã crop về đúng kích thước VLM (224x224 cho CLIP)
    * Đưa qua __Image Encoder__ để lấy vector f_image 
    * Lưu vector và metadata vào FAISS
4. Truy vấn (Text querry):
    * người dùng nhập mô tả -> **CLIP Text Encoder** sinh ra vector f_text
    * FAISS tính khoảng cách cosine -> trả về top **K** ảnh khớp nhất