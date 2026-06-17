# Report nhanh: Hệ thống dự báo tăng trưởng lợi nhuận doanh nghiệp niêm yết

## 1. Mục tiêu

Khóa luận xây dựng hệ thống hỗ trợ dự báo xu hướng tăng trưởng lợi nhuận của doanh nghiệp niêm yết tại Việt Nam trong ngắn hạn và trung hạn. Thay vì dự báo trực tiếp một giá trị lợi nhuận cụ thể, hệ thống tiếp cận bài toán dưới dạng phân loại nhị phân nhằm xác định doanh nghiệp có khả năng thuộc nhóm tăng trưởng lợi nhuận mạnh hay không.

Mục tiêu chính là hỗ trợ sàng lọc và phân tích doanh nghiệp, không đưa ra khuyến nghị mua bán cổ phiếu.

## 2. Dữ liệu sử dụng

Dữ liệu được tổ chức theo dạng doanh nghiệp - quý, bao gồm nhiều nhóm thông tin:

- Báo cáo tài chính: doanh thu, lợi nhuận, tài sản, vốn chủ sở hữu, dòng tiền, ROA, ROE.
- Dữ liệu thị trường: giá cổ phiếu, khối lượng giao dịch, lợi suất, biến động giá.
- Chỉ báo kỹ thuật: RSI, MACD, stochastic, CCI và các chỉ báo xu hướng.
- Dữ liệu vĩ mô: GDP, lạm phát, lãi suất, tỷ giá, tăng trưởng công nghiệp.
- Thông tin ngành và dữ liệu tin tức/sự kiện tài chính.

## 3. Phương pháp

Điểm trọng tâm của khóa luận là thiết kế biến mục tiêu thích nghi theo ngành. Thay vì dùng một ngưỡng tăng trưởng cố định cho tất cả doanh nghiệp, hệ thống xác định ngưỡng tăng trưởng mạnh theo phân vị của từng nhóm ngành. Cách làm này giúp phản ánh sự khác biệt về chu kỳ và mức dao động lợi nhuận giữa các ngành.

Ngoài ngưỡng tăng trưởng, hệ thống kết hợp thêm điều kiện ROA, ROE và net profit TTM để giảm nhiễu nhãn, tránh trường hợp doanh nghiệp được gán nhãn tăng trưởng mạnh chỉ vì nền lợi nhuận quá thấp.

Các mô hình được thử nghiệm gồm:

- Random Forest
- XGBoost
- LightGBM
- Weighted Soft Voting
- Ridge Regression
- ARIMA

Mô hình được đánh giá bằng các chỉ số Accuracy, Precision, Recall, F1-score, Balanced Accuracy và AUC.

## 4. Kết quả chính

Kết quả thực nghiệm cho thấy cấu hình biến mục tiêu `s80_q10_p10` cho hiệu quả tốt hơn cấu hình `s70_q20_p20` trên phần lớn chỉ số quan trọng.

Với cấu hình `s80_q10_p10`:

- LightGBM đạt Balanced Accuracy cao nhất: 0.783.
- Weighted Soft Voting đạt Accuracy cao nhất: 0.825.
- Weighted Soft Voting đạt F1-score cao nhất: 0.671.
- Weighted Soft Voting, XGBoost và Random Forest cùng đạt AUC cao nhất: 0.860.

Kết quả này cho thấy các mô hình học máy trên dữ liệu bảng, đặc biệt là nhóm ensemble và boosting, phù hợp hơn so với các mô hình cơ sở tuyến tính hoặc chuỗi thời gian đơn biến trong bài toán này.

## 5. Thành phần chatbot

Bên cạnh mô hình dự báo, khóa luận tích hợp chatbot để hỗ trợ người dùng truy vấn và diễn giải kết quả. Chatbot giúp người dùng hỏi theo mã cổ phiếu, ngành hoặc kết quả dự báo, sau đó trả lời bằng ngôn ngữ tự nhiên dựa trên dữ liệu dự báo và bằng chứng liên quan.

Ví dụ:

- Truy vấn kết quả dự báo của ACB.
- Truy vấn nhóm ngân hàng có mã nào được dự báo tăng trưởng mạnh.
- Diễn giải xác suất dự báo, ngưỡng quyết định và nhãn dự báo.

Chatbot đóng vai trò là lớp giao tiếp hỗ trợ phân tích, không thay thế mô hình dự báo và không đưa ra khuyến nghị đầu tư.

## 6. Đóng góp chính

Khóa luận có ba đóng góp chính:

- Xây dựng quy trình dữ liệu theo doanh nghiệp - quý, kết hợp nhiều nguồn dữ liệu tài chính, thị trường, vĩ mô, ngành và tin tức.
- Thiết kế biến mục tiêu thích nghi theo ngành nhằm phản ánh đặc thù tăng trưởng của từng nhóm ngành.
- Tích hợp chatbot hỗ trợ truy vấn và diễn giải kết quả dự báo theo hướng dễ tiếp cận hơn cho người dùng.

## 7. Hạn chế

Một số hạn chế còn tồn tại:

- Dữ liệu tài chính theo quý còn giới hạn về độ dài lịch sử.
- Hiệu quả dự báo chưa đồng đều giữa các nhóm ngành.
- Dữ liệu tin tức/sự kiện còn hạn chế về độ phủ và chất lượng xử lý ngữ nghĩa.
- Hệ thống hiện mới phân loại doanh nghiệp có hoặc không thuộc nhóm tăng trưởng mạnh, chưa dự báo chính xác giá trị lợi nhuận cụ thể.
- Chatbot chủ yếu hỗ trợ truy vấn và diễn giải kết quả, chưa xử lý được hội thoại phân tích tài chính phức tạp nhiều bước.

## 8. Hướng phát triển

Các hướng phát triển tiếp theo gồm:

- Mở rộng dữ liệu lịch sử và chuẩn hóa thêm dữ liệu tài chính trước năm 2018.
- Xây dựng mô hình riêng cho từng nhóm ngành lớn.
- Bổ sung dữ liệu báo cáo phân tích, tin tức chất lượng cao và đặc trưng chu kỳ ngành.
- Thử nghiệm các mô hình chuỗi thời gian hiện đại khi dữ liệu đủ lớn.
- Mở rộng chatbot theo hướng RAG để truy xuất báo cáo, giải thích dự báo và hỗ trợ phân tích tự nhiên hơn.

## 9. Kết luận ngắn

Khóa luận đã xây dựng được một hệ thống dự báo tăng trưởng lợi nhuận doanh nghiệp dựa trên học máy và biến mục tiêu thích nghi theo ngành. Kết quả thực nghiệm cho thấy các mô hình học máy trên dữ liệu bảng có khả năng khai thác tín hiệu từ dữ liệu tài chính, thị trường, vĩ mô, ngành và tin tức để hỗ trợ nhận diện doanh nghiệp có triển vọng tăng trưởng lợi nhuận mạnh. Thành phần chatbot giúp kết quả dự báo dễ tiếp cận hơn, nhưng hệ thống vẫn giữ vai trò hỗ trợ phân tích và không đưa ra khuyến nghị mua bán cổ phiếu.

