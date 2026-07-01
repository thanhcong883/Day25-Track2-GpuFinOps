# NimbusAI — GPU FinOps: Bài viết ngắn (Lab 25)

## 1. Baseline vs. Optimized

| | Baseline | Optimized | Tiết kiệm |
|---|---|---|---|
| Chi phí (tháng) | $27,133 | $14,626 | **$12,507 (46%)** |
| Inference | $6.488/1M-token | $1.126/1M-token | **82.6%** |
| Purchasing (GPU-hr) | $25,667/tháng | $15,627/tháng | **39.1%** |

Tổng tiết kiệm 46% nằm trong dải mục tiêu 40–95% mà đề bài đặt ra. Điểm quan trọng nhất:
`$/1M-token` giảm từ $6.488 xuống $1.126 (giảm 5.8×) — con số này phản ánh đúng hiệu quả phục vụ,
trong khi `$/GPU-giờ` (M3) chỉ giảm 39%. Nếu chỉ nhìn `$/GPU-giờ`, ta sẽ đánh giá thấp tác động
thực sự của việc tối ưu tầng inference.

## 2. Phân tích từng đòn bẩy

| Đòn bẩy | Tiết kiệm/tháng | Đóng góp |
|---|---|---|
| Purchasing (spot/reserved) | $10,040 | 80.3% |
| Inference (cascade/cache/batch) | $1,212 | 9.7% |
| Right-size util-lies | $655 | 5.2% |
| Kill idle GPUs | $600 | 4.8% |

**Purchasing đóng góp lớn nhất** vì 3/8 workload huấn luyện đủ điều kiện chạy trên spot (giảm
~40–60%) và 3 workload inference ổn định (24h/ngày, chạy đủ 30/30 ngày trong kỳ) đủ điều kiện
reserved 3 năm (giảm 45%) — đây là những khoản tiết kiệm áp dụng lên **toàn bộ GPU-giờ tháng**,
quy mô đô-la lớn hơn nhiều so với tối ưu từng request inference.

Ngược lại, **Inference lever đóng góp ít hơn về số tuyệt đối ở M5** dù tỷ lệ % savings trong M2
(82.6%) cao hơn hẳn — vì M2 chỉ đo trên nhật ký token 1 ngày mẫu, còn M5 quy mô hoá theo GPU-giờ
tháng nơi purchasing chiếm phần bánh lớn hơn. Đây chính là lý do phải đo cả hai đơn vị: `$/1M-token`
cho biết *hiệu quả*, `$/GPU-giờ` cho biết *quy mô đô-la thực tế cần cắt giảm trước*.

**Thứ tự ưu tiên hành động (theo ROI/nỗ lực):**
1. **Purchasing** — chuyển ngay workload interruptible sang spot + checkpoint, và inference 24/7
   ổn định sang reserved. Không cần thay đổi code ứng dụng, chỉ là quyết định mua hàng.
2. **Inference cascade + cache + batch** — cần thay đổi routing logic nhưng tỷ suất tiết kiệm
   trên mỗi request rất cao (82.6%), đáng làm sớm vì token traffic sẽ tăng theo thời gian.
3. **Right-size util-lie + Kill idle** — tiết kiệm nhỏ hơn về số tuyệt đối nhưng dễ làm, rủi ro
   thấp, nên làm song song.

## 3. GPU-Util Lie

Hai GPU bị gắn cờ "lie" (`gpu_util_pct ≥ 90%` nhưng `MFU < 30%`):

- **gpu-h100-4** — util 98.2%, MFU chỉ 0.194
- **gpu-a10g-1** — util 96.9%, MFU chỉ 0.268

**Cơ chế:** `nvidia-smi` GPU-Util chỉ đo "SM clock có đang bận không trong khoảnh khắc lấy mẫu" —
nó không phân biệt được giữa "đang tính FLOPs thật" và "đang stall chờ dữ liệu từ HBM/PCIe" hay
"đang chờ kernel launch tiếp theo". Một kernel decode LLM (arithmetic intensity ~1–2 FLOP/byte,
memory-bound theo roofline) có thể giữ SM ở trạng thái "active" gần 100% thời gian trong khi phần
lớn chu kỳ đó GPU chỉ đang đợi HBM trả dữ liệu — SM "bận" nhưng không sinh ra FLOPs tương ứng.

**Tác động tài chính:** công ty đang trả **100% giá GPU-giờ on-demand** cho `gpu-h100-4` (H100
$2.5/giờ) nhưng chỉ nhận về ~19.4% FLOPs khả dụng. Nếu tính theo "hiệu quả trả tiền", 1 giờ H100
util-lie tương đương lãng phí ~$2.02/giờ so với một workload đạt MFU mục tiêu 40%. Ở M5, right-size
2 GPU này (hạ xuống tier thấp hơn phù hợp) tiết kiệm **$655/tháng** — một con số khiêm tốn ở quy mô
11 GPU của bài lab, nhưng ở quy mô hạm đội thật (hàng trăm GPU) đây là hạng mục điều tra đầu tiên vì
chi phí lãng phí tỷ lệ thuận với số GPU bị "lie".

## 4. Phần mở rộng đã làm

### Extension 1 — Cải thiện `recommend_tier()`
**File:** `finops/pricing.py` (hàm `recommend_tier`, thêm `INTERRUPT_RATE_BY_GPU`), áp dụng trong
`missions/m3_purchasing.py`.

Thêm 2 yếu tố: (a) tỷ lệ bị thu hồi (interrupt rate) khác nhau theo GPU type — H100/H200 hiếm bị
thu hồi (0.03/giờ) vì luôn khan hiếm, trong khi A10G (0.10/giờ) và L4 (0.12/giờ) bị thu hồi thường
xuyên hơn nhiều; (b) so sánh reserved 1yr vs 3yr dựa trên số ngày job thực sự hoạt động trong kỳ
(`job_days/period_days ≥ 85%` mới coi là "đủ ổn định" cho cam kết 3 năm).

**Kết quả đo được:** so với policy v1 (39.1% saved), policy v2 cho **38.3% saved** (giảm 0.8
điểm %). Nguyên nhân: `job-dev-sandbox` (A10G, 8h/ngày, cờ interruptible=1) được v1 gán "spot"
nhưng v2 phát hiện A10G có tỷ lệ thu hồi 10%/giờ — cao hơn ngưỡng an toàn 8%/giờ — nên hạ về
`on_demand`. 3 job inference 24/7 (`infer-chat/rag/search`) vẫn giữ nguyên `reserved_3yr` vì chúng
hoạt động đủ 30/30 ngày trong kỳ (ổn định thật sự).

**Insight quan trọng nhất:** cờ `interruptible=1` trong dữ liệu không đủ để quyết định — cần biết
*GPU nào* đang chạy trên đó. Một quyết định "spot" sai trên GPU dễ bị thu hồi có thể khiến chi phí
thực tế (sau rework do bị ngắt giữa chừng) cao hơn dự kiến; v2 chấp nhận tiết kiệm thấp hơn một
chút để đổi lấy dự báo chi phí đáng tin cậy hơn.

### Extension 3 — `cache_is_worth_it()`
**File:** `finops/pricing.py` (hàm `cache_is_worth_it`, `cache_breakeven_reads`), áp dụng trong
`missions/m2_inference_levers.py`.

Break-even = số lần đọc lại tối thiểu để tiền tiết kiệm từ chiết khấu đọc (90%) bù được chi phí ghi
cache một lần (giả định ghi cache đắt hơn ~1.25× giá input thường).

**Kết quả đo được:** break-even ≈ **1.4 lần đọc** cho cả 2 tier (small/large — vì tỷ lệ chiết khấu
đọc/ghi giống nhau ở cả hai giá). Dữ liệu thực tế (`token_usage.csv`) cho **300 lần đọc trung bình**
mỗi cặp (team, project) — vượt xa ngưỡng hoà vốn hơn 200 lần. `cache_is_worth_it()` trả về `True`
cho cả hai tier.

**Insight quan trọng nhất:** với traffic lặp lại nhiều (system prompt dùng chung cho hàng trăm
request/ngày mỗi team), gần như luôn nên bật cache — ngưỡng hoà vốn quá thấp so với tần suất tái sử
dụng thực tế. Hàm này chỉ thực sự "gate" savings trong kịch bản traffic rất phân mảnh (mỗi prefix
chỉ dùng 1 lần, ví dụ một-off queries) — điều không xảy ra trong fleet hiện tại của NimbusAI.

### Extension 4 — Ngân sách Reasoning
**File:** `missions/m2_inference_levers.py`.

Tách riêng chi phí `$` và năng lượng `Wh` cho `is_reasoning=1` so với traffic thường.

**Kết quả đo được:** reasoning traffic chỉ chiếm **8.4%** số request nhưng **16.5%** chi phí `$`
và **94.0%** tổng năng lượng `Wh`. Vì `is_reasoning=1` đã dưới ngưỡng đề xuất 10% traffic, việc
"cap xuống 10%" không cắt được thêm request nào — nhưng con số 94% Wh so với chỉ 8.4% traffic mới
là phát hiện đáng chú ý.

**Insight quan trọng nhất:** reasoning tốn năng lượng gấp ~80× một query thường (multiplier trong
`sustainability.wh_per_query`) vì phải sinh nhiều token suy luận trung gian trước câu trả lời cuối.
Traffic hiện tại còn nhỏ nên tổng tác động $ chưa lớn, nhưng nếu tỷ trọng reasoning tăng (ví dụ đội
`eval` mở rộng dùng reasoning cho nhiều tác vụ hơn), chi phí *năng lượng* — và carbon đi kèm — sẽ
tăng nhanh hơn nhiều so với chi phí *tiền*. **Đề xuất routing rule:** chỉ bật reasoning khi
confidence-score của route nhỏ ban đầu dưới ngưỡng (ví dụ < 0.6), thay vì bật theo team; đồng thời
đặt cảnh báo giám sát khi tỷ lệ traffic reasoning vượt 10% để chặn trước khi nó trở thành vấn đề.

## 5. Khuyến nghị cho NimbusAI (nếu là FinOps lead)

1. **Chuyển ngay 3 job huấn luyện interruptible sang spot + checkpoint** (`train-llm`,
   `train-embed`, `finetune`) — không cần thay đổi code, chỉ là quyết định mua hàng, tiết kiệm
   ngay ~$7,900/tháng trên 3 job này. Đồng thời audit `job-dev-sandbox`: dù được gắn cờ
   interruptible, nó chạy trên A10G (dễ bị thu hồi) với duty cycle thấp — nên dùng on-demand thay
   vì ép sang spot.
2. **Bật cascade + prompt caching + batch cho toàn bộ traffic inference** — đây là lever hiệu quả
   nhất tính theo `$/1M-token` (giảm 82.6%), và dữ liệu cho thấy caching gần như luôn có lợi
   (break-even ~1.4 lần đọc so với thực tế 300 lần đọc/prefix).
3. **Điều tra và right-size `gpu-h100-4` và `gpu-a10g-1` trong tuần này** — đây là 2 GPU đang bị
   tính tiền full-rate nhưng chỉ trả về <30% FLOPs. Song song, dừng các GPU idle >90% thời gian
   (tiết kiệm $600/tháng chỉ với 1 GPU idle 8h/ngày — con số này nhân lên nhanh khi fleet lớn hơn).

Về dài hạn: thiết lập cảnh báo tự động khi traffic `is_reasoning` vượt 10% (Extension 4), và định kỳ
(hàng tháng) chạy lại `recommend_tier()` v2 để bắt các workload đổi hành vi (ví dụ một job dev trở
nên ổn định 24/7 và đủ điều kiện reserved).
