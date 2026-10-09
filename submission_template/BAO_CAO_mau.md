# Báo cáo lab: chọn tracker cho 5 video

**Nhóm:** DK

**Thành viên:**
- Nguyễn Minh Dương — 2A202602920
- Hoàng Trung Khải — 2A202602947

Detector cố định: `yolo26n.pt`, ảnh 640 px, chỉ lớp người. Re-ID cố định: `osnet_x0_25_msmt17.pt`. Thực nghiệm chạy trên GPU NVIDIA GeForce GTX 1650 4 GB, thiết bị `cuda:0`.

## 1. Cấu hình đã chọn

Mỗi video được thử trên cùng 150 frame đầu với ByteTrack và BoT-SORT. Với BoT-SORT, giữ `iou=0.5` để thử riêng `conf=0.15/0.3/0.5`, rồi giữ `conf=0.3` để thử riêng `iou=0.4/0.5/0.7`. Video_4 được thử thêm bốn cấu hình ByteTrack theo cùng cách vì xuất hiện hộp trùng ở BoT-SORT. Các lượt thử và video có ID nằm trong `runs/thu_nghiem/`; bản nộp dùng toàn bộ chuỗi ảnh, không có `--max-frames`.

Quan sát được đối chiếu tại frame 30/90/150 của các lượt thử và các mốc của bản đủ frame. Đây là đánh giá bằng mắt trên những đoạn kiểm tra, không phải phép đo danh tính toàn bộ video_2–video_5.

| Video | Tracker | conf | iou | Quan sát khi xem video | Đã thử nhưng loại |
|---|---|---|---|---|---|
| video_1 (quảng trường, tĩnh, ban ngày) | BoT-SORT | 0.15 | 0.5 | Người gần camera có hộp rõ; nhiều người nhỏ dưới tán cây và cuối quảng trường vẫn bị bỏ sót. Hạ conf giữ thêm một số hộp ở người xa nhưng cũng tăng hộp giả/hộp trùng. Chọn theo HOTA và IDF1 trên bản đủ 600 frame. | ByteTrack 0.3/0.5 có HOTA và IDF1 thấp hơn; BoT-SORT conf=0.5 bỏ nhiều hộp hơn trong đoạn thử. |
| video_2 (phố đêm, tĩnh, rất đông) | BoT-SORT | 0.15 | 0.5 | Các người ở hai vỉa hè và gần camera dễ được giữ hộp hơn nhóm phía xa. Frame 30, ngưỡng 0.15 giữ thêm người áo sáng cạnh vùng đèn phía trên bên trái. Vẫn có người không được phát hiện trong nhóm đông và có track bị ngắt quanh vùng đèn chói. | ByteTrack 0.3/0.5 và BoT-SORT conf=0.5 cho ít hộp ở người xa hơn; không chọn chỉ dựa vào việc có ít ID. |
| video_3 (camera di động, ảnh nhỏ) | BoT-SORT | 0.15 | 0.5 | Hai người áo xám và áo sọc ở tiền cảnh có hộp qua các mốc 30/90/150. Ngưỡng thấp giữ thêm người bị cắt ở mép trái và người nhỏ ở nền; ID vẫn dễ bị ngắt khi người bị che hoặc camera đổi góc. | ByteTrack 0.3/0.5 có ít hộp ở các mốc thử; BoT-SORT conf=0.5 mất hộp người nhỏ ở phía xa trong frame 90. |
| video_4 (trong nhà, camera di chuyển) | ByteTrack | 0.3 | 0.5 | Người áo đỏ giữ ID 4 và người áo trắng giữ ID 6 ở frame 30/90/150. BoT-SORT tạo hai hộp gần trùng cho một người phía trái lối đi ở frame 90; ByteTrack 0.3 tránh được trường hợp này trong đoạn đối chiếu. Những người ở xa, bị che và vùng kính/sàn bóng vẫn cần kiểm tra thêm. | BoT-SORT 0.3/0.5 có hộp trùng; ByteTrack conf=0.15 giữ thêm hộp nhưng có một frame với hộp gần trùng trong đoạn thử. |
| video_5 (trên xe bus, giao lộ đông) | BoT-SORT | 0.15 | 0.5 | Hộp người chủ yếu ở hai vỉa hè; ô tô không được xuất vì đã khóa lớp người. Frame 90, conf=0.15 giữ thêm các người trong nhóm bên trái so với 0.3/0.5. Khi xe tiến lên, người nhỏ, bị cột che hoặc ra mép ảnh vẫn mất hộp/đứt track. | ByteTrack 0.3/0.5 và BoT-SORT conf=0.5 giữ ít hộp của nhóm người bên trái hơn trong đoạn thử. |

Trong các lượt giữ nguyên tracker/conf, các file kết quả khi đổi `iou` 0.4/0.5/0.7 được đối chiếu trực tiếp. Nếu kết quả giống nhau thì giữ 0.5; không quy kết một ngưỡng IoU giúp giữ ID tốt hơn khi chưa có bằng chứng.

### Kiểm tra bản nộp đủ frame

| Video | Frame đã xử lý và frame preview | Số dòng kết quả MOT |
|---|---:|---:|
| video_1 | 600 | 5617 |
| video_2 | 1050 | 13339 |
| video_3 | 837 | 5489 |
| video_4 | 900 | 5729 |
| video_5 | 750 | 3481 |

Tổng cộng **4.137 frame**. Năm file TXT có 10 cột MOT hợp lệ, ID/frame là số nguyên và không lặp cùng ID trong một frame; cả năm preview giải mã được ở đầu/giữa/cuối và có đúng số frame nguồn. Video_5 có hai frame không có track nên không có dòng kết quả cho chúng; preview vẫn đủ 750 frame. Đã chạy **7 unit test: tất cả thành công**; `git diff --check` cũng thành công. Cấu hình và biên bản kiểm tra nằm trong `runs/nop_bai/cau_hinh.json` và `runs/nop_bai/kiem_tra_bai_nop.json`.

## 2. Số liệu video_1

Chấm bằng `scripts/evaluate_practice.py`, chỉ dùng nhãn video_1. Bảng chính dùng **đủ 600 frame**; các số TrackEval dưới đây theo thang 0–100.

```text
Cấu hình bản nộp: BoT-SORT, conf=0.15, iou=0.5
Run: DK_video1
Sequence   HOTA     MOTA     IDF1
video_1    28.594   16.528   27.865
COMBINED   28.594   16.528   27.865
```

Các giá trị được lấy từ output HOTA / CLEAR / Identity của TrackEval, nhật ký đầy đủ: [`runs/video_1_danh_gia.log`](../runs/video_1_danh_gia.log).

| Cấu hình đủ 600 frame | HOTA | MOTA | IDF1 | TP | FN | FP | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| ByteTrack, conf=0.3, iou=0.5 | 26.025 | 17.389 | 25.464 | 3382 | 15199 | 123 | 28 |
| **BoT-SORT, conf=0.15, iou=0.5 — nộp** | **28.594** | **16.528** | **27.865** | **4218** | **14363** | **1116** | **31** |

Không kết luận đây là tác động riêng của Re-ID, vì hai cấu hình cuối còn khác `conf`. So sánh cùng `conf=0.3, iou=0.5` trên **150 frame đầu** cho ByteTrack HOTA 28.593 / IDF1 26.953 và BoT-SORT HOTA 31.451 / IDF1 29.343. Bảng chọn ngưỡng trên cùng đoạn thử:

| Tracker | conf | iou | HOTA | MOTA | IDF1 |
|---|---:|---:|---:|---:|---:|
| ByteTrack | 0.3 | 0.5 | 28.593 | 15.317 | 26.953 |
| BoT-SORT | 0.15 | 0.5 | 32.365 | 15.732 | 31.199 |
| BoT-SORT | 0.3 | 0.4 | 31.451 | 14.590 | 29.343 |
| BoT-SORT | 0.3 | 0.5 | 31.451 | 14.590 | 29.343 |
| BoT-SORT | 0.3 | 0.7 | 31.451 | 14.590 | 29.343 |
| BoT-SORT | 0.5 | 0.5 | 27.165 | 14.097 | 24.197 |

Bảng 150 frame chỉ phục vụ chọn cấu hình, không thay cho số liệu bản nộp. Khi chấm đoạn thử, dùng bản sao nhãn chỉ gồm frame 1–150 và đặt độ dài chuỗi 150; nhãn gốc được giữ nguyên. Video_2 đến video_5 không có nhãn trong gói lab, không có HOTA/MOTA/IDF1 cho các video đó.

## 3. Phân tích

### Video_1: chọn theo HOTA nhưng chấp nhận thêm hộp giả

Camera tĩnh giúp dự đoán chuyển động tương đối thuận lợi, nhưng detector nano ở kích thước 640 vẫn bỏ nhiều người nhỏ và người trong bóng cây. BoT-SORT 0.15 tăng TP từ 3382 lên 4218 so với baseline, đồng thời tăng FP từ 123 lên 1116. HOTA tăng 2.569 điểm và IDF1 tăng 2.401 điểm, trong khi MOTA giảm 0.861 điểm và số lần đổi ID tăng từ 28 lên 31. Vì vậy lựa chọn này ưu tiên cân bằng phát hiện và liên kết danh tính theo HOTA, không khẳng định rằng mọi kiểu lỗi đều giảm. Recall bản nộp chỉ 22.701%, nên chất lượng còn bị giới hạn mạnh bởi phát hiện; Re-ID không tự tạo được hộp cho người detector bỏ sót.

### Video_2: ngưỡng thấp giúp giữ người xa trong cảnh đêm

Ở đoạn thử, conf=0.15 giữ thêm một số người phía xa và gần đèn sáng, trong khi ngưỡng 0.5 làm mất nhiều hộp hơn. Các nhóm đông dễ che nhau và có ngoại hình tương tự, nên thông tin ngoại hình của BoT-SORT là một tín hiệu bổ sung có ích bên cạnh chuyển động. Tuy nhiên việc có thêm hộp hoặc thêm ID chưa chứng minh giữ đúng danh tính tốt hơn; các mốc đã xem vẫn có mất hộp và đứt track ở vùng chói. Chọn cấu hình này để ưu tiên không bỏ người quá sớm, đồng thời ghi nhận nguy cơ hộp giả và gán nhầm khi người đi sát nhau. Không có nhãn để kết luận định lượng rằng BoT-SORT thắng trên toàn video.

### Video_3 và video_5: camera di chuyển

Chuyển động camera làm vị trí người dịch chuyển trong ảnh ngay cả khi người đó đi chậm, nên mô hình chuyển động đơn thuần dễ gặp khó ở mép ảnh hoặc sau che khuất. BoT-SORT kết hợp ngoại hình và bù chuyển động camera; đây là lý do lựa chọn nó cho hai cảnh này, cùng với các hộp bổ sung quan sát được khi hạ conf. Ở video_3, các người gần camera được bám qua các mốc đối chiếu, nhưng ảnh nhỏ và góc nhìn thay đổi vẫn khiến track không liên tục. Ở video_5, các người trên vỉa hè rất nhỏ khi xe còn xa, nên conf=0.5 làm mất hộp ở nhóm bên trái rõ hơn ngưỡng 0.15. Nhận xét này dựa trên các frame đã kiểm tra và chưa chứng minh giảm số lần đổi ID trên toàn chuỗi.

### Video_4: chọn ByteTrack vì kết quả quan sát

Dù camera tiến tới và có nhiều kính, các người tiền cảnh trong đoạn thử vẫn được ByteTrack giữ ID ổn định: áo đỏ ID 4, áo trắng ID 6 ở frame 30/90/150. BoT-SORT tại frame 90 có hai ID 7 và 17 với hai hộp gần trùng quanh cùng một người; ByteTrack 0.3 không có hộp trùng như vậy trong đoạn 150 frame đã kiểm tra. Vì vậy không mặc định tracker có Re-ID luôn tốt hơn: ở đoạn này ưu tiên hộp rõ và ít bị chồng trùng. Ngưỡng 0.15 của ByteTrack giữ thêm hộp, nhưng cũng xuất hiện một trường hợp hộp gần trùng ở frame 91, nên chọn 0.3. Cấu hình này vẫn có thể mất người ở xa hoặc sau che khuất dài; chưa có nhãn để đo chính xác mức cải thiện.

### Hoàn thiện cài đặt và notebook

Lỗi ban đầu là `ModuleNotFoundError: No module named 'cv2'`; env còn thiếu cả PyTorch. Đã cài các phụ thuộc, dùng PyTorch `2.6.0+cu126`, torchvision `0.21.0+cu126`, Ultralytics `8.4.0`, BoxMOT `10.0.42`, NumPy `1.26.4`, OpenCV `4.11.0.86` và setuptools dưới 81. BoxMOT cũ khóa NumPy 1.23.1 không phù hợp wheel Python 3.11, nên cài BoxMOT riêng bằng `--no-deps` và kiểm chứng trên môi trường hiện tại; không đổi tracker YAML hoặc trọng số.

Kernel notebook đã đăng ký là `Python (cv_robotics_lab21)`. Notebook đã chạy hết trên `cuda:0`, ba nhận định lần lượt là **True / False / True**. Frame đầu video_1 có 12 / 6 / 4 hộp tương ứng conf 0.15 / 0.3 / 0.5; đây là hộp detector, chưa có ID. Script tracking đã sửa để detector dùng đúng `--device` và truyền `torch.device` cho Re-ID; wrapper TrackEval vá alias NumPy trong tiến trình con và chuẩn bị đúng split. Gói dữ liệu thiếu `eval_config.json`, nên bổ sung cấu hình chấm nội bộ `LAB_TRACKING`, split `train`, chỉ cho video_1.

## 4. Nếu có thêm thời gian

Sẽ kiểm tra kỹ từng đoạn bị ngắt ID ở vùng đèn chói và sau che khuất, thử thêm StrongSORT/DeepOCSORT, rồi quét conf mịn quanh 0.15–0.3 trong phạm vi detector và Re-ID cố định. Chỉ sau phần nộp chính mới cân nhắc thử mô hình Re-ID khác và ghi rõ đó là thí nghiệm mở rộng.
