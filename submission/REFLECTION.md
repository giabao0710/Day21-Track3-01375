# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Sự đảo ngược hoàn toàn giữa loss huấn luyện và năng lực thực tế ở NB4: cấu hình `attn_only` nâng rank lên r=283 có train loss thấp hơn cả bản `correct` (0.0531 < 0.0549) nhưng khi đánh giá trên tập target thực tế lại thua rõ rệt (0.7350 so với 0.8150). Điều này chứng minh trực quan rằng việc tối ưu một chỉ số thay thế (proxy metric) như train loss hoàn toàn có thể dẫn đến kết luận sai lệch nếu không đo lường trực tiếp trên tác vụ nghiệp vụ.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở việc hiểu sâu và kiểm chứng cơ chế loss mask, giải mã token ngược từ `input_ids` và `labels` ở NB1 để đảm bảo ranh giới giữa lượt của user và assistant được phân định tuyệt đối chính xác. Đây đúng là chỗ tôi dự đoán từ đầu, vì nếu dữ liệu và mask sai lệch từ gốc thì toàn bộ thời gian huấn luyện và đánh giá phía sau đều trở nên vô nghĩa.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước lab này, tôi từng tin rằng fine-tuning luôn vượt trội hơn prompt engineering trong mọi trường hợp và chỉ cần tăng rank LoRA lên càng cao thì mô hình sẽ càng thông minh. Giờ tôi hiểu rằng một prompt được thiết kế tối ưu có thể đạt hiệu năng rất cao (0.7650) với chi phí triển khai tức thì, và trong LoRA thì vị trí gắn adapter (toàn bộ text linear layers) mới là đòn bẩy cốt lõi chứ không phải rank.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để hỗ trợ phân tích luồng dữ liệu, viết mã kiểm thử và đối soát các số liệu thực nghiệm. Chỗ AI assistant dễ mắc sai lầm là thường sao chép các thói quen mặc định phổ biến trên mạng (như tự động gán `bf16=True` trên phần cứng Turing/T4 không hỗ trợ phần cứng, hoặc tin tưởng cờ `assistant_only_loss` của thư viện mà không kiểm tra xem template có thẻ `{% generation %}` hay không).

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên tôi làm là đóng băng một tập dữ liệu đánh giá chuẩn (gồm cả dữ liệu nghiệm thu tác vụ chuyên biệt và dữ liệu kiểm tra năng lực tổng quát), sau đó xây dựng một baseline prompt thật kỹ lưỡng để đo mốc hiệu năng chuẩn. Chỉ khi chứng minh được bài toán cần tối ưu độ trễ, giảm token ngữ cảnh hoặc prompt không thể đáp ứng được ngưỡng chất lượng thì mới bắt tay vào quy trình fine-tune có kiểm soát.
