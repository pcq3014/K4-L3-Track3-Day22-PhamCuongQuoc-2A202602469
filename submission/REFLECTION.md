# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Phạm Cường Quốc (2A202602469)
**Khoá:** A20-K4
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ output của notebook đã chạy, không ước lượng bằng mắt. Phần bắt buộc (NB0–NB4) lấy từ
> lần chạy thứ hai, `colab/Lab22_DPO_T4.ipynb`: bảng log của `DPOTrainer`, `dpo_metrics.json` in ở NB3 §4,
> `judge_summary.json` in ở NB4 §4. Bonus NB3b lấy từ lần chạy thứ nhất, `colab/Lab22_DPO_T4_run1_nb3b.ipynb`
> (xem §6 để biết vì sao có hai lần chạy).

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4, 14,56 GB khả dụng |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit (LoRA r=16, α=32, 33M tham số huấn luyện = 0,81%) |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch (125 bước, 10 phút 25 giây, loss 1,884 → 1,284, trung bình 1,360) |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out, không trùng câu hỏi |
| Chosen dài hơn rejected (NB2) | 65,9% (trung vị: chosen 94 token, rejected 86 token) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 (100 bước, loss `sigmoid`, reference = `models/sft-merged`, log-prob tính trước) |
| Giám khảo | rm-panel: Skywork-Reward-V2-Llama-3.2-3B (Skywork-Reward-V2-Qwen3-4B bị loại vì sanity 67% < 80%). Chấm chéo: `openai:gpt-4o-mini` |
| Chi phí | Colab miễn phí; chi phí API không đáng kể cho giám khảo gpt-4o-mini (58 cặp × 2 thứ tự A/B) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 27 phút 02 giây (100 bước) + 50 giây eval cuối |
| VRAM cao nhất | không ghi lại |
| Loss huấn luyện (bước log đầu → cuối) | 0.6952 → 0.6485 (trung bình cả epoch 0.6751) |
| Loss held-out (bước 25 → 100) | 0.6866 → 0.6562 |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.097 (chosen +0.385, rejected +0.288) |
| Độ chính xác reward trên held-out | 0.69 (0.64 → 0.67 → 0.69 → 0.69 ở các bước 25/50/75/100) |
| Margin trên held-out | +0.082 (chosen +0.397, rejected +0.315) |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4, 58 câu) | 569 → 588 ký tự (+3,4%); riêng held-out 574 → 600 (+4,5%) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

**Chosen và rejected đều tăng, trên cả train lẫn held-out.** Trên held-out, `rewards/chosen` đi 0.079 → 0.268 → 0.373
→ 0.397 và `rewards/rejected` đi 0.066 → 0.213 → 0.295 → 0.315 ở các bước 25/50/75/100. Đổi ra log-prob tuyệt đối
(reward = β·Δlog-prob, β = 0.1): trên held-out, `logps/chosen` tăng khoảng +3,97 nat (−390,14 → −386,18) và
`logps/rejected` tăng khoảng +3,15 nat (−328,80 → −325,65). Margin tăng vì **chosen tăng nhanh hơn rejected**, không
phải vì rejected giảm nhanh hơn. Vì vậy đây **không phải** likelihood displacement: đường chosen không lúc nào âm.

**Held-out đi cùng hướng với train.** Margin held-out tăng đều 0.013 → 0.055 → 0.079 → 0.082, độ chính xác 0.64 → 0.69
và loss held-out giảm 0.687 → 0.656. Margin train dao động mạnh (từ 0.03 đến 0.097, có lúc tụt về 0.03 ở bước 60 và
0.04 ở bước 95) vì mỗi bước log chỉ có vài batch 8 cặp, nhưng xu hướng khớp held-out. Không thấy học thuộc: held-out
không đi ngang hay giảm trong khi train tăng. Margin held-out đã gần bão hoà từ bước 75 (0.079 → 0.082), cho thấy với
lr 5e-6 có chạy thêm epoch thì cũng chỉ tăng chậm.

**Chẩn đoán tự động là INTENDED, nhưng tôi nghĩ nhãn này hơi lạc quan.** Hàm `diagnose` chỉ kiểm tra margin > 0 và
chosen > 0. Mẫu "đúng kỳ vọng" trong rubric là chosen ↑ và rejected ↓, còn ở đây rejected cũng ↑. Giả thuyết của tôi:
bước SFT trên 1.000 mẫu Alpaca (câu trả lời ngắn) đã kéo mô hình lệch khỏi phong cách của Qwen3-Instruct gốc. Dữ liệu
sở thích sailor2 là *on-policy*, sinh từ mô hình cùng họ Qwen, nên cả chosen lẫn rejected đều "giống Qwen gốc" hơn bản
SFT. Khi DPO cập nhật, mô hình quay lại một phần về phong cách chung đó, nên cả hai reward cùng tăng. Phần thật sự học
từ sở thích chỉ là chênh lệch +0.082 (0,82 nat log-ratio), với độ chính xác 0.69. Tín hiệu có thật nhưng yếu: loss
held-out chỉ giảm từ 0.693 xuống 0.656, và NB4 (§4) xác nhận nó gần như không đổi hành vi sinh câu trả lời.

**So với lần chạy thứ nhất** (cùng cấu hình, phiên Colab khác): held-out accuracy 0.64 và margin +0.077 (chosen +0.372,
rejected +0.296), lần này 0.69 và +0.082. Hướng và độ lớn giống nhau, nên kết luận ở trên không phụ thuộc lần chạy.
Chênh lệch 0.05 độ chính xác trên 100 cặp nằm trong nhiễu (sai số chuẩn ≈ √(0.67·0.33/100) ≈ 0.047).

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

> Ảnh: `screenshots/04-side-by-side-table.png` · số liệu: `data/eval/judge_summary.json`

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 10 | 8 | 32 | 0.52 [0.44, 0.60] | 0.522 (n=46) | 35,3% |
| hữu ích — helpfulness | 4 | 1 | 0 | 3 | 0.625 [0.50, 0.875] | 0.50 (n=3) | 0% |
| an toàn — safety | 4 | 0 | 0 | 4 | 0.50 [0.50, 0.50] | 0.50 (n=4) | — |
| tổng | 58 | 11 | 8 | 39 | 0.526 [0.448, 0.603] | 0.519 (n=53) | 33,3% |

Giám khảo: `rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: Llama 100% (12/12), Qwen3 67% (8/12, bị
loại) · `score_length_spearman`: Llama 0.057, Qwen3 0.260 · đồng ý giữa hai RM: 77,6% · chấm chéo với `gpt-4o-mini`:
đồng ý 79,3% (58 cặp).

**Kết luận chính: không phát hiện khác biệt.** Win rate held-out 0.52 với khoảng tin cậy [0.44, 0.60] chứa 0.5, nên
không thể nói DPO tốt hơn SFT. 32/50 cặp held-out hoà, phần lớn vì câu trả lời greedy của hai mô hình gần như trùng
nhau. Điều này khớp với margin nhỏ ở §3: DPO với 100 bước, lr 5e-6 chỉ dịch mô hình rất ít khỏi SFT.

**Giám khảo nào đáng tin?** Skywork-Reward-V2-Qwen3-4B chỉ đúng 8/12 cặp tiếng Việt hiển nhiên, nên bị loại khỏi hội đồng
và kết luận chỉ dựa vào RM Llama. Đây là hạn chế: hội đồng chỉ còn một giám khảo. RM Qwen3 cùng họ với mô hình đang học
và với Sailor2 (mô hình sinh dữ liệu), lại cùng lab Skywork với RM đã gán nhãn dữ liệu huấn luyện (Skywork-Reward-Gemma-2-27B),
nên dễ bị rò rỉ sở thích. Trong `per_judge`, hai RM cho cùng số thắng/thua trên held-out (10/8/32, win rate 0.52), nên trên
đầu ra của tôi không thấy RM Qwen3 thiên vị DPO hơn. Khác biệt nằm ở **độ dài**: với RM Qwen3, câu dài hơn thắng 64,7%
số cặp có kết quả và điểm tương quan với độ dài 0.26; với RM Llama chỉ 35,3% và 0.06. Vậy RM Qwen3 vừa đọc tiếng Việt kém
vừa thưởng câu dài, nên loại nó là đúng. RM Llama vẫn là Skywork V2 (cùng lab với RM gán nhãn), nên rò rỉ sở thích chưa
loại trừ hoàn toàn. Giám khảo `gpt-4o-mini` khác họ hẳn và đồng ý 79,3% với hội đồng, đây là bằng chứng độc lập duy nhất.

**Hack độ dài?** Không thấy. DPO dài hơn SFT 4,5% trên held-out (574 → 600 ký tự), nhưng với RM Llama câu dài hơn chỉ
thắng 35,3% và win rate trên các cặp dài gần bằng nhau (0.522) gần như bằng win rate chung (0.52). Ở 8 câu cố định,
DPO thậm chí ngắn hơn ở nhóm hữu ích (643 → 605 ký tự). Dữ liệu có 65,9% cặp chosen dài hơn nhưng mô hình chưa học
theo, có lẽ vì độ dịch chuyển khỏi SFT quá nhỏ.

**Nhận xét định tính từ bảng 8 câu:**

- **Thay đổi rất nhỏ.** Ở h1 (quicksort), h3 (email xin nghỉ phép), h4 (Python và JavaScript) và cả 4 câu an toàn, 300 ký
  tự đầu của SFT và DPO trùng nhau từng chữ. Bốn câu an toàn có độ dài trung bình bằng nhau tuyệt đối (425,5 ký tự) nên
  giám khảo chấm hoà cả 4.
- **Ví dụ hữu ích (h2: "5kg gạo và 12 quả trứng, gợi ý 3 món").** Đây là câu duy nhất có khác biệt rõ và là trận thắng duy
  nhất của DPO ở nhóm hữu ích. Cả hai bản đều trả lời kém: bịa nguyên liệu không có ("gà nướng với khoai tây", "canh cá").
  Bản SFT lặp ý ("nướng trong lò nướng hoặc nướng trong lò nướng"). Bản DPO bỏ câu lặp và gắn món với cơm ("có thể được
  phục vụ với cơm"), tức là dùng nguyên liệu người hỏi có, nhưng vẫn không sửa được lỗi bịa nội dung.
- **Ví dụ an toàn (s1–s4).** Cả hai bản đều từ chối với cùng câu mở đầu ("Tôi xin lỗi, nhưng tôi không thể…"). Với s4
  (stress vì thi cử, hỏi cách tự kết liễu), mô hình từ chối rồi khuyên tìm hỗ trợ. Hướng đi đúng, nhưng câu mở đầu xin lỗi
  máy móc và không đưa số đường dây nóng cụ thể. DPO không thay đổi gì ở nhóm này.
- **Lỗi định dạng.** Mọi câu trả lời, cả SFT lẫn DPO, đều mở đầu bằng `</tool_call>` / `<tool_call>`. Lỗi có từ ngay
  sau NB1 (câu thử quicksort ở NB1 §4), nên nằm ở bước SFT hoặc tokenizer/chat template, không phải do DPO. Mẫu SFT ở NB1
  §2 có chèn khối `<think></think>` rỗng (do `enable_thinking=False`), trong khi bản Qwen3-4B-Instruct-2507 không dùng chế
  độ suy nghĩ. Đây là giả thuyết của tôi, chưa kiểm chứng. Các token rác này có mặt ở cả hai bản nên không làm lệch kết
  quả so sánh, nhưng có thể là một lý do RM Qwen3 chấm kém.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

Không chạy. Giả thuyết: với β = 0.05, mô hình được phép đi xa reference hơn nên margin held-out (đo theo log-ratio) sẽ lớn
hơn, nhưng dễ xuất hiện likelihood displacement và câu trả lời dài ra hơn. Với β = 0.5, mô hình bị giữ sát SFT, margin
gần 0 và đầu ra gần như trùng SFT. Ở β = 0.1, 32/50 cặp held-out đã hoà vì câu trả lời gần như trùng SFT, nên tôi dự
đoán β = 0.05 hoặc lr lớn hơn mới cho khác biệt đo được ở NB4.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

**Quyết định: chạy lại toàn bộ NB1 → NB4 trong một phiên mới và bỏ NB3b, thay vì chỉ chạy lại riêng phần chấm của NB4.**

1. **Bối cảnh:** ở lần chạy thứ nhất, tôi chạy bonus NB3b (5 biến thể loss) ngay trước NB4. Sau NB3b, GPU vẫn giữ 5,53 GB
   (sau NB3 chỉ 1,64 GB). NB4 sinh xong câu trả lời nhưng bị CUDA OOM khi nạp reward model giám khảo (cần thêm 7,41 GiB).
   Sau đó phiên Colab hết quota GPU và các file trên `/content` bị mất, kể cả `models/sft-merged` và `adapters/dpo`.
2. **Phương án thay thế:** chỉ tải lại `adapters/dpo` từ file zip để chạy NB4. Nhưng `models/sft-merged` (8 GB) không được
   tải về, và adapter DPO lưu đường dẫn tới đúng mô hình SFT đó, nên NB4 sẽ chấm một cặp SFT/DPO không khớp với nhau.
3. **Vì sao chọn chạy lại toàn bộ:** bài làm cần một chuỗi SFT → DPO → chấm nhất quán, nhất là khi DPO dùng SFT làm
   reference. Tôi bỏ NB3b ở lần này (đã có kết quả từ lần trước) và tải `data/eval` ngay sau NB4 để không mất lần nữa.
4. **Kết quả:** NB4 chạy hết, kể cả phần chấm: hai RM được nạp lần lượt nên không còn OOM. Lần chạy lại còn có lợi phụ:
   tôi có hai lần chạy DPO độc lập (held-out accuracy 0.64 rồi 0.69, margin +0.077 rồi +0.082), đủ để thấy chênh lệch giữa
   các lần chạy nằm trong nhiễu và kết luận ở §3 không phải do may mắn.
5. **Điều bất ngờ:** kết quả chấm cho thấy câu hỏi "DPO có tốt hơn SFT không" có câu trả lời là "chưa đo được" (khoảng tin
   cậy [0.44, 0.60]). Ngoài ra, RM Qwen3 mà tôi tưởng là giám khảo chính lại trượt bộ kiểm tra tiếng Việt.
6. **Làm lại thì đổi gì:** chạy phần bắt buộc trước, tải kết quả về sau mỗi notebook, rồi mới làm bonus ở phiên riêng.
   Tôi cũng sẽ sửa lỗi `<tool_call>` ngay sau NB1, và thử lr 1e-5 hoặc 2 epoch để DPO dịch mô hình đủ xa mà NB4 đo được.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

Không chạy.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png` · notebook: `colab/Lab22_DPO_T4_run1_nb3b.ipynb` (lần chạy thứ nhất)

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
NLL. Độ chính xác reward cao nhất là ORPO (0.66) và DPO (0.65), nhưng NB4 đã cho thấy độ chính xác reward 0.69 của DPO
chính vẫn không chuyển thành win rate khác 0.5. Muốn biết biến thể nào thật sự tốt hơn cần chạy giám khảo NB4 trên từng
adapter (`DPO_ADAPTER_OVERRIDE`).

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
- [x] Chấm chéo bằng hai họ mô hình (+4): rm-panel + `openai:gpt-4o-mini`, `cross_judge.agreement` = 0.793
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Cả chosen lẫn rejected đều tăng reward trong khi tôi tưởng rejected phải giảm. Sau gần 30 phút DPO, 32/50 câu trả lời
held-out vẫn hoà với SFT và 4 câu an toàn trùng nhau từng chữ. Một lỗi định dạng (`<tool_call>` ở đầu mọi câu trả lời)
từ bước SFT lại dễ thấy hơn nhiều so với bất kỳ thay đổi nào của DPO. Cuối cùng, reward model cùng họ Qwen với mô hình
của tôi lại là giám khảo kém nhất cho tiếng Việt.
