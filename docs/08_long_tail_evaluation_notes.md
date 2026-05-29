# Ghi Chu Danh Gia Long-tail Cho TA-RecMind

Tai lieu nay tom tat cac ket luan khi phan tich notebook `src/TA_RecMind_V2_IntraLayer.ipynb`, tap trung vao muc tieu goi y san pham long-tail va cach danh gia mo hinh trong train/test.

## 1. Muc Tieu Chinh

Muc tieu cua mo hinh khong phai toi uu `Overall Recall` thuong, ma la giam popularity bias va cai thien kha nang goi y san pham long-tail.

Vi vay metric chinh nen tap trung vao:

- `Tail Recall@K`: positive item thuoc TAIL co lot top-K khong.
- `Tail NDCG@K`: positive tail item co xuat hien o vi tri cao khong.
- `Tail Coverage@K`: mo hinh co goi y nhieu tail items khac nhau khong.
- `AvgPopularity@K` / `MedianPopularity@K`: do pho bien trung binh cua item duoc recommend.
- `Overall NDCG@K`: guardrail de dam bao relevance tong the khong sup qua manh.

`Cold Recall@K` nen bao cao rieng, khong nen tron truc tiep vao long-tail score neu cold chua duoc train dung cach.

## 2. Cach Code Hien Tai Hoat Dong

Notebook dang train mo hinh LightGCN + LLM embedding + intra-layer gated fusion.

Luong chinh:

1. Load `gold_edge_index.npy` lam train graph.
2. Load tan suat item/user va nhan `HEAD/MID/TAIL/COLD`.
3. Load hoac encode LLM embedding cho user/item bang `all-MiniLM-L6-v2`.
4. Build sparse adjacency chi tu train edges.
5. Train `TARecMindV2` bang BPR loss, tail-weighted alignment va LAGCL.
6. Danh gia validation bang stratified full-ranking sample.
7. Sau train, load best checkpoint va danh gia test bang full-ranking.

Phan dung:

- Graph train chi dung train edges, khong dua val/test vao adjacency.
- Val/test chi dung lam ground truth.
- Full-ranking test trong Cell 15 la metric nghiem tuc nhat.
- Co train masking: khong recommend lai item user da tuong tac trong train.

Phan can than trong:

- `OVERALL` tren validation hien tai la tren sample can bang HEAD/MID/TAIL/COLD, khong phai phan phoi val that.
- Sampled evaluation 1+100 cho ket qua rat cao nhung khong phan anh full recommendation thuc te.
- `ColdRecall` dang bang 0, khong nen dung de chon checkpoint chinh.
- Negative sampling co ham loc false negative, nhung train loop hien tai khong truyen `u` va `user_train_dict`, nen loc false negative chua hoat dong trong training.

## 3. Head, Mid, Tail Nen Chia Theo Gi

Can tach ro hai viec:

### 3.1. Gan nhan item

`HEAD/MID/TAIL` nen duoc dinh nghia theo so luong san pham, dua tren `train_freq`.

Vi du:

```text
HEAD = top 20% items theo train_freq
MID  = next 10% items
TAIL = bottom 70% items
```

Cach nay tra loi cau hoi: san pham nay pho bien hay long-tail?

Dieu binh thuong trong long-tail:

```text
HEAD: it san pham nhung chiem rat nhieu tuong tac
TAIL: rat nhieu san pham nhung moi san pham it tuong tac
```

### 3.2. Danh gia theo interaction

Sau khi item da co nhan, moi interaction trong val/test duoc gan nhan theo positive item:

```text
(u, i_val) thuoc TAIL neu item i_val la TAIL
```

`Tail Recall@K` duoc tinh tren cac val/test interactions co positive item la TAIL.

## 4. Xu Ly Tap Val Trong Qua Trinh Train

Khong dua val vao graph train. Val chi dung lam ground truth.

Protocol dung:

```text
For each (u, i_val) in val:
  score user u voi toan bo item catalog
  mask train_items[u]
  rank i_val
  tinh Recall@K, NDCG@K
```

### 4.1. Khong can full val moi epoch

Full val co khoang 1.85M interactions, full-ranking moi epoch se rat nang.

Nen dung hai tang validation:

1. `stratified sampled val`: chay nhanh moi vai epoch.
2. `full representative val`: chay thua hon de xac nhan checkpoint.

### 4.2. Validation nhanh

Dung sample co dinh seed, tao mot lan truoc training va dung lai cho moi epoch.

Vi muc tieu la long-tail, sample co the uu tien TAIL:

```text
HEAD:  2,500 - 5,000 interactions
MID:   2,500 - 5,000 interactions
TAIL:  10,000 - 20,000 interactions
COLD:  2,500 - 5,000 interactions, chi de theo doi phu
```

Moi sampled interaction van nen rank voi toan bo item catalog, khong nen dung 1+100 negatives de early stopping.

Can goi ro day la `balanced/stratified validation`, khong phai representative overall.

### 4.3. Full validation

Nen chay:

```text
FULL_VAL_EVERY = 10 epoch
```

hoac chay full val cho mot vai checkpoint tot nhat sau training.

Full val moi la validation dai dien theo phan phoi thuc.

## 5. Co Nen Lay Nhieu Tail Hon Trong Train Khong

Co. Day la cost-sensitive learning / reweighting, phu hop voi muc tieu long-tail.

Neu train theo phan phoi tu nhien, head interactions ap dao gradient va model de bi popularity bias.

Loi ich cua oversampling tail:

- Tang gradient cho tail items.
- Giam popularity bias.
- Tang `TailRecall`, `TailNDCG`, `TailCoverage`.

Rui ro neu qua manh:

- Overall Recall/NDCG giam.
- Head/MID performance giam.
- Model recommend tail qua nhieu nhung khong lien quan.
- De overfit vao mot nhom nho tail users/items.

Goi y batch composition:

```text
60-70% interactions ngau nhien theo train that
20-30% tail-positive interactions
10-20% niche/inactive-user interactions
```

Can co guardrail tren validation representative de dam bao model khong chi "day tail" ma mat relevance.

## 6. Vi Sao Cold Recall Dang Bang 0

Cold bang 0 la hop ly voi code hien tai.

Ly do:

1. `COLD_START` co `train_freq = 0`, khong co positive edge trong train.
2. Cold item khong xuat hien nhu positive trong BPR.
3. Cold item lai co the bi sample lam negative vi `freq_safe = max(freq, 1)`.
4. Graph khong giup cold vi cold item la node co lap trong train adjacency.
5. LLM branch chua duoc train bang objective content matching rieng cho cold.
6. Trong full-ranking, cold positive phai canh tranh voi toan bo catalog nen Recall@20 bang 0 la de hieu.

Ket luan:

- Khong nen dung `(TailRecall + ColdRecall) / 2` lam early stopping score chinh.
- Nen report cold rieng nhu bai toan zero-shot/cold-start.
- Neu muon xu ly cold that su, can co objective rieng.

Huong xu ly cold neu can:

- Khong sample `train_freq == 0` lam negative trong BPR.
- Train content matching giua user profile text va positive item text tren warm train pairs.
- Cho cold item scoring dua nhieu hon vao content branch.
- Danh gia cold bang protocol rieng.

## 7. Co Nen Xu Ly User Khong

Co. Xu ly user la hop ly va co y nghia thuc te.

Long-tail khong chi la item-side, ma con lien quan user-side:

- User it tuong tac co collaborative signal yeu.
- User co khau vi niche de bi model day ve head items.
- User moi/it du lieu can content/profile signal hon graph signal.

`User_INACTIVE/ACTIVE/SUPER_ACTIVE` tra loi cau hoi:

```text
Mo hinh co hoat dong tot voi user it lich su khong?
```

`User_NICHE` tra loi cau hoi khac:

```text
Voi user co lich su thich tail items, mo hinh co tiep tuc recommend dung tail items khong?
```

Hai nhom nay khong thay the nhau.

## 8. Nen Them Nhom User Niche

Nen them `User_NICHE` nhu mot phan tich bo sung.

Dinh nghia:

```text
user_niche_ratio(u) =
  so train interactions cua user voi TAIL items
  / tong so train interactions cua user
```

Chia nhom co the dung nguong:

```text
NICHE_USER      : user_niche_ratio >= 0.5
MIXED_USER      : 0.2 <= user_niche_ratio < 0.5
MAINSTREAM_USER : user_niche_ratio < 0.2
```

Hoac dung percentile neu phan phoi qua lech:

```text
Top 20% user_niche_ratio -> NICHE_USER
Middle 60%               -> MIXED_USER
Bottom 20%               -> MAINSTREAM_USER
```

Metric nen bao cao theo user niche:

- `Recall@20`
- `NDCG@20`
- `TailRecall@20`
- `TailNDCG@20`
- `AvgPopularity@20`

Truoc mat nen dua `User_NICHE` vao evaluation truoc, sau do moi quyet dinh co can dua vao sampler/loss hay khong.

## 9. Early Stopping Va Chon Epoch

Khong nen chon epoch co dinh. Nen dat max epoch lon va chon checkpoint bang validation long-tail.

Goi y:

```text
MAX_EPOCHS:      80-100
MIN_EPOCHS:      20-25
EVAL_EVERY:      2 epoch
FULL_VAL_EVERY:  10 epoch
PATIENCE:        8-12 evals
min_delta:       1e-4 hoac 5e-5
```

Khong nen early stop truoc epoch 20 vi code co LAGCL warmup/ramp, mo hinh chua on dinh.

Early stopping score nen uu tien tail:

```text
LongTailValScore =
  0.50 * TailNDCG@20
+ 0.25 * TailRecall@20
+ 0.15 * TailCoverage@20
+ 0.10 * OverallNDCG@20
```

Phien ban don gian:

```text
LongTailValScore =
  0.60 * TailNDCG@20
+ 0.30 * TailRecall@20
+ 0.10 * TailCoverage@20
```

Khong nen dua `ColdRecall` vao score chinh khi cold dang bang 0.

Guardrail khi save best:

- `OverallNDCG@20` khong giam qua 10-15%.
- `HEAD Recall@20` khong sup qua manh.
- `TailCoverage@20` khong giam.
- `AvgPopularity@20` nen giam hoac on dinh theo muc tieu long-tail.

Quy tac:

```text
if epoch < MIN_EPOCHS:
  train tiep, khong early stop

if LongTailValScore > best_score + min_delta:
  save best
  patience = 0
else:
  patience += 1

if patience >= PATIENCE:
  stop
```

## 10. Danh Gia Test Sau Training

Test chi dung sau khi da chon checkpoint bang validation.

Metric chinh nen la full-ranking, khong phai sampled 1+100.

Protocol:

```text
For each (u, i_test) in test:
  score u voi toan bo item catalog
  mask train_items[u]
  rank i_test
  tinh Recall@K, NDCG@K
```

Neu sau khi chon hyperparameter ma retrain bang train+val, khi test can mask ca train+val.

Bao cao:

- `OVERALL`
- `Item_HEAD`
- `Item_MID`
- `Item_TAIL`
- `Item_COLD`
- `User_INACTIVE`
- `User_ACTIVE`
- `User_SUPER`
- `User_NICHE` neu them
- `Coverage@K`
- `TailCoverage@K`
- `AvgPopularity@K`
- `MedianPopularity@K`

Sampled 1+100 hoac 1+1000 chi nen la metric phu de tham chieu paper, va chi so sanh khi protocol/dataset/negative sampling giong nhau.

## 11. Ket Luan Chinh

Huong code hien tai phu hop voi muc tieu long-tail: co LLM embedding, gated fusion, tail-weighted alignment, tail oversampling, LAGCL va stratified evaluation.

Tuy nhien can dieu chinh cach danh gia:

1. Dung `TAIL` lam metric chinh, khong tron voi `COLD`.
2. Dung stratified sampled val de train nhanh, nhung can full representative val de xac nhan.
3. Dung full-ranking test lam ket qua chinh.
4. Them `User_NICHE` de chung minh mo hinh phuc vu user co khau vi long-tail.
5. Early stopping nen dua vao `TailNDCG`, `TailRecall`, `TailCoverage`, kem guardrail overall.
6. Cold-start nen xu ly rieng neu muon bao cao nhu mot dong gop rieng.

## 12. Phuong Phap Chot Cho Giai Doan Hien Tai: Warm Long-tail, Tam Bo Qua Cold-start

Trong giai doan hien tai, nen tam thoi bo qua `COLD_START` de tap trung vao bai toan chinh:

```text
Goi y dung cac san pham long-tail da tung xuat hien trong train nhung co tan suat thap.
```

Day la bai toan `warm long-tail recommendation`, khac voi `cold-start recommendation`.

### 12.1. Dinh Nghia Tap Danh Gia

Tao `warm_item_mask`:

```text
warm_item_mask = item_train_freq > 0
```

Khi danh gia val/test trong giai doan nay:

- Loai cac interaction co positive item la `COLD_START`.
- Candidate set chi gom warm items.
- Mask train items cua user.
- Khong mask positive val/test item.

Protocol:

```text
For each (u, i_eval) in val/test:
  if item_train_freq[i_eval] == 0:
    skip interaction

  candidates = all warm items
  scores = score(u, candidates)
  scores[train_items[u]] = -inf
  rank i_eval
  tinh Recall@K, NDCG@K
```

Ly do nen mask cold khoi candidate set trong giai doan nay:

- Cold item khong co positive edge trong train.
- Cold item chua duoc train dung objective rieng.
- Neu de cold trong candidate set, chung co the chen vao ranking va lam nhieu metric warm long-tail bi nhieu.
- Muc tieu hien tai la do nang luc long-tail tren item co the hoc tu train graph.

Sau nay neu muon lam cold-start, can tach thanh mot protocol rieng.

### 12.2. Training Sampler

Train graph van chi dung `train_edges`.

Khong dua val/test vao graph, user profile, item profile hay loss.

Trong giai doan warm long-tail, negative sampling nen lay tren warm items:

```text
negative_candidates = items where item_train_freq > 0
```

Ly do:

- Cold items khong phai muc tieu danh gia hien tai.
- Dua cold vao negative qua nhieu co the lam model hoc day cold xuong, gay hai neu sau nay muon xu ly cold-start.

Batch composition nen dung:

```text
70% train interactions ngau nhien theo phan phoi that
30% train interactions co positive item thuoc TAIL
```

Neu them user niche/inactive vao sampler:

```text
60% random train interactions
30% tail-positive interactions
10% niche-user hoac inactive-user interactions
```

Khong nen oversample tail qua manh hon 50% neu chua co guardrail tot, vi co the lam model recommend tail nhieu nhung kem relevance.

### 12.3. Fast Validation Trong Khi Train

Tao mot tap validation sample co dinh seed truoc khi train.

Tap nay lay theo interaction, positive item khong duoc la cold:

```text
HEAD:  5,000 val interactions
MID:   5,000 val interactions
TAIL: 20,000 val interactions
```

Day la `tail-heavy stratified val`, dung de theo doi muc tieu long-tail.

Moi `EVAL_EVERY = 2` epoch:

```text
1. Tinh embedding tu train graph.
2. Danh gia sample val tren full warm catalog.
3. Mask train_items[u].
4. Tinh metric theo group positive item.
```

Khong dung sampled negative 1+100 de early stopping.

Metric log moi lan fast val:

- `TailRecall@20`
- `TailNDCG@20`
- `TailCoverage@20`
- `OverallWarmRecall@20`
- `OverallWarmNDCG@20`
- `HeadRecall@20`
- `MidRecall@20`
- `AvgPopularity@20`
- `MedianPopularity@20`

Neu co `User_NICHE`, log them:

- `NicheUserTailRecall@20`
- `NicheUserTailNDCG@20`
- `NicheUserAvgPopularity@20`

### 12.4. Representative Warm Val

De tranh chon checkpoint bi lech do tail-heavy sample, can co them representative validation sample.

Tao mot sample ngau nhien tu toan bo warm val theo phan phoi that:

```text
RepresentativeWarmVal:
  sample 100,000 - 200,000 warm val interactions
  giu dung phan phoi HEAD/MID/TAIL nhu val that
  exclude COLD_START positives
```

Chay representative warm val:

```text
REP_VAL_EVERY = 10 epoch
```

hoac it nhat chay cho top 3-5 checkpoint sau training.

Representative warm val dung de kiem tra:

- Model co that su tot tren phan phoi val gan thuc te khong.
- Overall warm metric co sup qua manh khong.
- Tail improvement co phai chi do sample tail-heavy khong.

### 12.5. Early Stopping

Cau hinh nen dung:

```text
MAX_EPOCHS:     80-100
MIN_EPOCHS:     20
EVAL_EVERY:     2
REP_VAL_EVERY:  10
PATIENCE:       10 evals
min_delta:      1e-4
```

Khong early stop truoc epoch 20.

Fast long-tail score:

```text
FastLongTailScore =
  0.60 * TailNDCG@20
+ 0.30 * TailRecall@20
+ 0.10 * TailCoverage@20
```

Guardrail tren fast val:

```text
OverallWarmNDCG@20 khong giam qua 10-15%
HeadRecall@20 khong sup qua manh
TailCoverage@20 khong giam
AvgPopularity@20 giam hoac on dinh
```

Quy tac trong train:

```text
if epoch < MIN_EPOCHS:
  train tiep, khong early stop

if FastLongTailScore > best_fast_score + min_delta
   and guardrails pass:
  save checkpoint candidate
  patience = 0
else:
  patience += 1

if patience >= PATIENCE:
  stop
```

Sau khi train xong, khong nen mac dinh lay checkpoint tot nhat theo fast val. Nen lay top 3-5 checkpoint candidates va danh gia lai bang representative warm val hoac full warm val.

Final checkpoint selection:

```text
FinalWarmValScore =
  0.50 * TailNDCG@20
+ 0.25 * TailRecall@20
+ 0.15 * TailCoverage@20
+ 0.10 * OverallWarmNDCG@20
```

Checkpoint co `FinalWarmValScore` cao nhat va qua guardrail la checkpoint cuoi cung.

### 12.6. Danh Gia Sau Train

Sau khi da chon checkpoint bang validation, moi chay test.

Test protocol hien tai nen la warm-only full-ranking:

```text
For each (u, i_test) in test:
  if item_train_freq[i_test] == 0:
    skip interaction

  candidates = all warm items
  scores = score(u, candidates)
  scores[train_items[u]] = -inf
  rank i_test
```

Neu sau nay retrain bang `train + val` truoc khi test, khi test phai mask ca:

```text
train_items[u] + val_items[u]
```

Bao cao ket qua test chinh:

- `OverallWarm Recall@20 / NDCG@20`
- `HEAD Recall@20 / NDCG@20`
- `MID Recall@20 / NDCG@20`
- `TAIL Recall@20 / NDCG@20`
- `TailCoverage@20`
- `Coverage@20`
- `AvgPopularity@20`
- `MedianPopularity@20`
- `User_INACTIVE / ACTIVE / SUPER`
- `User_NICHE / MIXED / MAINSTREAM`, neu da them

Khong bao cao `Item_COLD` trong bang ket qua chinh o giai doan nay.

Neu can nhac den cold, ghi ro:

```text
Cold-start is out of scope for the current warm long-tail evaluation.
```

### 12.7. Ket Luan Protocol Chot

Protocol chot cho bai toan hien tai:

```text
Train:
  train graph only
  negative sampling tren warm items
  70% random train edges + 30% tail-positive train edges

Fast val moi 2 epoch:
  fixed tail-heavy warm val sample
  HEAD 5k, MID 5k, TAIL 20k
  full-ranking tren warm catalog
  early stop bang TailNDCG + TailRecall + TailCoverage

Representative val moi 10 epoch / sau train:
  sample 100k-200k warm val theo phan phoi that
  dung de xac nhan checkpoint

Final checkpoint:
  chon trong top candidates bang representative warm val score

Test:
  full-ranking tren warm test
  exclude cold positives
  candidates = warm items
  report OverallWarm, HEAD, MID, TAIL, TailCoverage, AvgPopularity, User_NICHE
```

Day la phuong phap phu hop nhat neu muc tieu hien tai la giai quyet long-tail item recommendation va tam thoi chua xu ly cold-start.

## 13. Code Update Trong `src/TA-REC.ipynb`

Da cap nhat notebook theo protocol warm long-tail:

- `IGNORE_COLD_ITEMS=True`, `EVAL_PROTOCOL="warm_long_tail_v1"`.
- Val trong training dung fixed stratified sample: `HEAD=5000`, `MID=5000`, `TAIL=20000`, khong co `COLD`.
- Val/test full-ranking chi rank tren warm catalog: `item_train_freq > 0`.
- Positive val/test cold bi skip; train masking van ap dung va positive held-out duoc restore score de khong bi mask nham.
- Negative sampling dung `neg_prob_warm_t`, cold item co probability = 0.
- Early stopping dung `LongTailValScore = 0.60*TailNDCG + 0.30*TailRecall + 0.10*TailCoverage`.
- Final test report bo `Item_COLD` khoi bang chinh va them `MedianPopularity@K`.
