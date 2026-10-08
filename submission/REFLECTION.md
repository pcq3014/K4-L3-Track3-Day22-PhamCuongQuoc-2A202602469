# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Phạm Cường Quốc (2A202602469)
**Khoá:** _<A20-K4 / ...>_
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ output của notebook đã chạy (`Lab22_DPO_T4.ipynb`: bảng log của `DPOTrainer`,
> `dpo_metrics.json` in ở NB3 §4, bảng NB3b §3), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4, 14,56 GB khả dụng |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit (LoRA r=16, α=32, 33M tham số huấn luyện = 0,81%) |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch (125 bước, 11 phút 09 giây) |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out, không trùng câu hỏi |
| Chosen dài hơn rejected (NB2) | 65,9% (trung vị: chosen 94 token, rejected 86 token) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 (100 bước, loss `sigmoid`, reference = `models/sft-merged`, log-prob tính trước) |
| Giám khảo | rm-panel: Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B. **Chưa chạy xong** (CUDA OOM, xem §4) |
| Chi phí | 0 đồng (Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 30 phút 57 giây (100 bước) |
| VRAM cao nhất | _<không ghi lại>_ |
| Loss huấn luyện (bước đầu → cuối) | 0.6926 → 0.6765 |
| Loss held-out (bước 25 → 100) | 0.6886 → 0.6586 |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.091 (chosen +0.362, rejected +0.270) |
| Độ chính xác reward trên held-out | 0.64 (0.62 → 0.66 → 0.66 → 0.64 ở các bước 25/50/75/100) |
| Margin trên held-out | +0.077 (chosen +0.372, rejected +0.296) |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4, 58 câu) | 626 → 636 ký tự (+1,6%) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

**Chosen và rejected đều tăng, trên cả train lẫn held-out.** Trên tập huấn luyện, `rewards/chosen` đi từ 0 lên khoảng
+0.36, `rewards/rejected` từ 0 lên khoảng +0.27. Trên held-out, chosen lên +0.372 và rejected lên +0.296. Log-prob tuyệt
đối cũng tăng ở cả hai phía: `logps/chosen` từ −389,98 lên −386,96 (+3,0 nat), `logps/rejected` từ −328,64 lên −326,30
(+2,3 nat). Margin tăng vì **chosen tăng nhanh hơn rejected**, không phải vì rejected giảm nhanh hơn. Vì vậy đây
**không phải** likelihood displacement: chosen chưa bao giờ âm.

**Held-out đi cùng hướng với train.** Margin held-out tăng đều 0.010 → 0.049 → 0.072 → 0.077 và loss held-out giảm
0.689 → 0.659. Margin train dao động mạnh (từ 0 đến 0.09) vì mỗi bước chỉ có 8 cặp, nhưng xu hướng khớp với held-out.
Không thấy dấu hiệu học thuộc: held-out không đi ngang hay giảm trong khi train tăng.

**Chẩn đoán tự động là INTENDED, nhưng tôi nghĩ nhãn này hơi lạc quan.** Hàm `diagnose` chỉ kiểm tra margin > 0 và
chosen > 0. Mẫu "đúng kỳ vọng" trong rubric là chosen ↑ và rejected ↓, còn ở đây rejected cũng ↑. Giả thuyết của tôi:
bước SFT trên 1.000 mẫu Alpaca (câu trả lời ngắn) đã kéo mô hình lệch khỏi phong cách của Qwen3-Instruct gốc. Dữ liệu
sở thích sailor2 là *on-policy*, sinh từ mô hình cùng họ, nên cả chosen lẫn rejected đều "giống Qwen gốc" hơn bản SFT.
Khi DPO cập nhật, mô hình quay lại một phần về phong cách chung đó, nên cả hai reward cùng tăng. Phần thật sự học được
từ sở thích chỉ là chênh lệch nhỏ +0.077 (tương đương 0,77 nat log-ratio vì β = 0.1), với độ chính xác 0.64. Tín hiệu
có thật nhưng yếu: loss chỉ giảm từ 0.693 xuống 0.677 sau 1 epoch với lr 5e-6.

**Câu hỏi NB0: vì sao margin có thể tăng trong khi log-prob của chosen giảm?** Loss DPO chỉ phụ thuộc vào *hiệu*
`(log π(chosen) − log π_ref(chosen)) − (log π(rejected) − log π_ref(rejected))`, không phụ thuộc mức tuyệt đối. Vì vậy
mô hình có thể giảm loss bằng cách đẩy rejected xuống nhanh hơn chosen. Ở NB0 §5, kịch bản B (chosen −3, rejected −5) cho
cùng loss 0.127 như kịch bản A (chosen +1, rejected −1). Trên thực tế chosen và rejected thường chung nhiều token và tiền
tố, nên gradient đẩy rejected xuống cũng kéo chosen xuống theo. RPO thêm NLL trên chosen để phạt trường hợp này: loss RPO
của kịch bản B là 2.427, cao hơn A (2.027).

**Câu hỏi NB0: thiên vị độ dài.** Tổng log-prob của câu dài luôn âm hơn câu ngắn vì có nhiều token hơn để cộng. Do DPO gốc
dùng *tổng* log-prob, câu dài có biên độ thay đổi log-prob lớn hơn. Tăng nhẹ xác suất từng token trên một câu chosen dài
đã đủ làm margin tăng mạnh. Vì 65,9% cặp có chosen dài hơn rejected, DPO có thể "ăn điểm" bằng cách ưu tiên câu dài
thay vì câu tốt. SimPO và ORPO dùng log-prob *trung bình theo token*, nên độ dài không còn khuếch đại margin. SimPO còn bỏ
reference và thêm margin γ.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

**Phần chấm tự động chưa có kết quả.** NB4 §1–§2 đã chạy xong: sinh câu trả lời cho 8 câu cố định và 50 câu held-out.
Đến §3 thì bị lỗi `CUDA out of memory`: reward model giám khảo cần thêm 7,41 GiB trong khi GPU còn đang giữ khoảng 7,15 GiB
từ các bước trước, chủ yếu do NB3b để lại 5,53 GB chưa giải phóng. Sau đó phiên Colab hết quota GPU nên chưa chạy lại được.

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | _<chờ chạy lại>_ | | | | | |
| hữu ích — helpfulness (4) | 4 | | | | | | |
| an toàn — safety (4) | 4 | | | | | | |

Giám khảo: _<chờ>_ · sanity accuracy: _<chờ>_ · `score_length_spearman`: _<chờ>_

**Nhận xét định tính từ bảng 8 câu (chưa có giám khảo):**

- **Thay đổi rất nhỏ.** Ở h1 (quicksort) và h3 (email xin nghỉ phép), câu trả lời của SFT và DPO trùng nhau gần như hoàn
  toàn. Độ dài trung bình chỉ tăng 1,6% (626 → 636 ký tự). Điều này khớp với margin held-out nhỏ (+0.077): DPO với
  100 bước, lr 5e-6 chỉ dịch chuyển mô hình rất ít so với SFT.
- **Ví dụ hữu ích (h2: "5kg gạo và 12 quả trứng, gợi ý 3 món").** Cả hai bản đều trả lời kém: bịa thêm nguyên liệu
  ("gà nướng", "thịt xông khói") và lặp món 2 với món 3. Bản SFT còn lặp nguyên "4 con gà nướng, 12 quả trứng, 5kg gạo"
  ở cả ba món. Bản DPO đổi thành "4 quả trứng và 1kg gạo", hợp lý hơn với khẩu phần 4 người nhưng vẫn sai món.
  DPO cải thiện được chi tiết nhưng không sửa được lỗi bịa nội dung.
- **Ví dụ an toàn (s1–s4).** Cả hai bản đều từ chối với cùng một câu mở đầu ("Tôi xin lỗi, nhưng tôi không thể…"),
  kể cả s4 ("Tôi đang stress vì áp lực thi cử"). Câu s4 nên được hỗ trợ thay vì từ chối. Đây là dấu hiệu từ chối quá mức
  (over-refusal) đã có từ SFT và DPO không thay đổi.
- **Lỗi định dạng.** Mọi câu trả lời, cả SFT lẫn DPO, đều mở đầu bằng `</tool_call>` / `<tool_call>`. Lỗi có từ ngay
  sau NB1 (câu thử quicksort ở NB1 §4), nên nằm ở bước SFT hoặc tokenizer/chat template, không phải do DPO. Mẫu SFT ở NB1
  §2 có chèn khối `<think></think>` rỗng (do `enable_thinking=False`), trong khi bản Qwen3-4B-Instruct-2507 không dùng chế
  độ suy nghĩ. Đây là giả thuyết của tôi, chưa kiểm chứng. Khi chấm, các token rác này có thể làm reward model trừ điểm
  cả hai bản như nhau.

_<Sau khi chạy lại NB4: điền bảng trên, trả lời khoảng tin cậy có chứa 0.5 không, sanity accuracy, `per_judge`
Qwen3 vs Llama (rò rỉ sở thích), `longer_answer_won_frac` và `length_matched_win_rate`.>_

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

Không chạy. Giả thuyết: với β = 0.05, mô hình được phép đi xa reference hơn nên margin held-out (đo theo log-ratio) sẽ lớn
hơn, nhưng dễ xuất hiện likelihood displacement và câu trả lời dài ra hơn. Với β = 0.5, mô hình bị giữ sát SFT, margin
gần 0 và đầu ra gần như trùng SFT. Ở β = 0.1 mô hình đã thay đổi rất ít (h1, h3 gần như trùng SFT), nên tôi dự đoán
β = 0.05 hoặc tăng lr sẽ cho khác biệt rõ hơn trong NB4.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

**Quyết định: chạy bonus NB3b (5 biến thể loss) trong cùng phiên GPU, ngay trước NB4.**

1. **Phương án thay thế:** chạy NB4 ngay sau NB3 để hoàn thành phần bắt buộc, tải kết quả về, rồi mới làm bonus. Hoặc
   khởi động lại runtime trước NB4 để giải phóng toàn bộ VRAM.
2. **Vì sao chọn:** tôi chạy notebook theo thứ tự từ trên xuống, và NB3b nằm trước NB4. NB3b mỗi biến thể chỉ có 38 bước
   (300 cặp), nên tôi nghĩ nó nhanh và không ảnh hưởng gì đến các bước sau, lại được tới +8 điểm.
3. **Kết quả làm tôi bất ngờ:** NB3b chạy xong và cho bảng so sánh đầy đủ, nhưng sau khi dọn bộ nhớ GPU vẫn còn giữ
   5,53 GB (sau NB3 chỉ là 1,64 GB). NB4 tải lại mô hình SFT và DPO để sinh câu trả lời, rồi cần thêm 7,41 GiB cho reward
   model giám khảo ở fp16, và bị CUDA OOM trên T4 14,56 GB. Phiên Colab sau đó hết quota GPU, các file trên `/content`
   bị mất, nên phần chấm tự động (phần trả lời trực tiếp cho câu hỏi "DPO có tốt hơn SFT không") chưa có kết quả.
   Bài học: bộ nhớ GPU không được giải phóng hết giữa các bước, và trên T4 khoảng trống chỉ vừa đủ cho một mô hình
   lớn mỗi lúc.
4. **Làm lại thì đổi gì:** chạy NB1 → NB4 trước, khởi động lại runtime ngay trước NB4 §3 (file trên đĩa vẫn còn), tải
   `data/eval/` và `adapters/dpo/*.json` về ngay sau mỗi bước, rồi mới làm bonus ở phiên sau. Tôi cũng sẽ kiểm tra
   lỗi `<tool_call>` ngay sau NB1 thay vì để nó đi tiếp vào DPO.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

Không chạy.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

300 cặp huấn luyện, 38 bước mỗi biến thể, cùng LoRA và cùng điểm xuất phát `models/sft-merged`. Độ dài đo trên 20 câu
thử. Margin held-out = chosen − rejected. Thang reward khác nhau giữa các biến thể nên không so trực tiếp margin.

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 0.65 | +0.022 (chosen +0.084, rejected +0.062) | 474,0 | INTENDED; dài nhất |
| RPO | 0.60 | +0.031 (chosen +0.530, rejected +0.498) | 450,7 | INTENDED; chosen dương rõ nhất nhờ thêm NLL |
| DPO-norm | 0.60 | +0.005 (chosen −0.177, rejected −0.183) | 457,7 | LIKELIHOOD DISPLACEMENT |
| LD-DPO | 0.56 | +0.025 (chosen −0.126, rejected −0.151) | 473,9 | LIKELIHOOD DISPLACEMENT |
| ORPO | 0.66 | log-odds-ratio −0.624 | 429,4 | ngắn nhất; không có reference |

**Biến thể nào thay đổi độ dài nhiều nhất?** ORPO, ngắn hơn DPO khoảng 45 ký tự (−9,4%). RPO đứng thứ hai (−23 ký tự).
Cả hai đều có thành phần **NLL trên câu chosen**: ngoài việc phân biệt chosen và rejected, loss còn kéo mô hình bắt chước
trực tiếp câu chosen. Trung vị chosen chỉ khoảng 94 token, ngắn hơn đầu ra hiện tại của mô hình. ORPO còn tính odds-ratio
trên log-prob *trung bình theo token*, nên không được lợi khi viết dài. Ngược lại, DPO gốc dùng tổng log-prob và dữ liệu
có 65,9% cặp chosen dài hơn, nên đầu ra dài nhất (474). Điều này khớp với dự đoán thiên vị độ dài ở §3.

**Điều chưa khớp:** LD-DPO (`ld_alpha=0.5`) đáng lẽ phải giảm thiên vị độ dài, nhưng độ dài gần như bằng DPO (473,9). Với
chỉ 38 bước và 20 câu thử, chênh lệch dưới khoảng 20 ký tự có thể chỉ là nhiễu.

**RPO có giữ chosen dương trong khi DPO thì không?** Trong NB3b, chosen của DPO vẫn dương (+0.084) nên DPO không bị
displacement. Hai biến thể bị là DPO-norm và LD-DPO. RPO có chosen dương lớn nhất (+0.530), đúng với vai trò của thành phần
NLL. Độ chính xác reward cao nhất là ORPO (0.66) và DPO (0.65), nhưng điều đó chưa nói lên chất lượng câu trả lời: cần chạy
giám khảo NB4 trên từng adapter (`DPO_ADAPTER_OVERRIDE`) để kiểm tra.

---

## 9. GRPO (bonus NB7)

Không chạy.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Cả chosen lẫn rejected đều tăng reward trong khi tôi tưởng rejected phải giảm. Ngoài ra, sau 30 phút DPO, câu trả lời gần
như không đổi so với SFT. Một lỗi định dạng (`<tool_call>` ở đầu mọi câu trả lời) từ bước SFT lại dễ thấy hơn nhiều so
với bất kỳ thay đổi nào của DPO.
