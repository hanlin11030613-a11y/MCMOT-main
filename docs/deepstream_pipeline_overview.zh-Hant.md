# DeepStream 多鏡頭追蹤管線 — 說明文件

**Language:** [English](deepstream_pipeline_overview.md) | 繁體中文

> 📎 簡報連結：[link-to-presentation](https://docs.google.com/presentation/d/1hlOluXIu6J6kOGV_ikMTZ4aSQCSX9h43_XvlXoMiOCo/edit?usp=sharing)

本文說明 `ds_multi_cam_tracking.py`（以及同目錄下的其他相關檔案）是如何組成的，
重點放在 DeepStream / GStreamer 的部分 — PGIE 與 SGIE 各自在做什麼、整條管線
（pipeline）是怎麼接起來的，以及實際上該怎麼執行它。

## 1. 整體架構

真正的追蹤 / ReID 邏輯（`botsort/` 內的 BoT-SORT + `GlobalRegistry` +
`MultiCameraTracker`）是純粹的 Python / NumPy 程式碼，完全不在乎輸入資料是
從哪裡來的。它每一幀要做的事情很單純：給定一組人物偵測結果（bbox、信心值、
一個 256 維的外觀 embedding），更新追蹤軌跡（tracks），並判斷「這是誰」——
包含跨鏡頭的情況。

在這個專案裡，DeepStream 的全部工作就是當那個**前端（front end）**，負責
產生上述輸入資料，而且要跟得上影像的即時速度：解碼 N 路攝影機串流、將它們
批次（batch）在一起、跑一個人物偵測模型、對每一個偵測到的人物裁切區塊
（crop）再跑一個 embedding（ReID）模型，最後把結果交給一個 GStreamer 的
pad probe，由裡面的 Python callback 把資料從 DeepStream 的 metadata 結構中
取出，餵給 `MultiCameraTracker`。

```
camera 1 ─┐
          ├─► nvstreammux（將 N 路串流批次化）─► PGIE（偵測器）─► SGIE（ReID embedder）─► [probe：取出偵測結果與 embedding]─► BoTSORT / GlobalRegistry ─► OSD 繪製 ─► tiler ─► sink
camera 2 ─┘
```

「[probe]」左邊的一切都屬於 DeepStream / GStreamer / TensorRT。「[probe]」
本身以及右邊的一切則屬於 Python 追蹤堆疊（tracking stack）。
`reid_pad_buffer_probe`（位於 `ds_multi_cam_tracking.py`）就是連接這兩個世界
的接縫。

## 2. 這裡用到的 DeepStream / GStreamer 基礎概念

DeepStream 應用程式本質上是 GStreamer pipeline：一連串 `Gst.Element` 以
src-pad → sink-pad 的方式串接起來，每個元件對流經的 buffer 做一件事（解碼、
批次化、推論、轉換、渲染……）。NVIDIA 的 DeepStream 外掛（`nvstreammux`、
`nvinfer`、`nvdsosd` 等）會在 buffer 沿著管線往下流動時，附加結構化的
**metadata**（`NvDsBatchMeta` → `NvDsFrameMeta` → `NvDsObjectMeta` → ...），
而 `pyds`（Python bindings）就是讓你能在 pad probe 裡讀取這些 metadata 的
工具。

這條管線中的關鍵元件，依序如下：

| 元件 | 角色 |
|---|---|
| `uridecodebin` / `nvurisrcbin`（每個來源各一個，位於 `create_source_bin` 內） | 用各自的 GPU 加速解碼器解碼每一路攝影機的檔案 / RTSP URI |
| `nvstreammux` | 把 N 路攝影機串流批次化成一個 batch（`batch-size`、`width`/`height` 來自 `app_config.yml`），讓下游推論一次處理整個 batch，而不是每路串流各跑一次 |
| `nvvideoconvert` + `capsfilter`（RGBA） | 把 muxer 的輸出轉成 CPU 可讀取（mappable）的 RGBA 格式，讓 gallery-snapshot 的程式碼能透過 `pyds.get_nvds_buf_surface()` 把真實像素抓回來 |
| **`nvinfer`（PGIE）** | 主要偵測器 — 詳見第 3 節 |
| **`nvinfer`（SGIE）** | 次要、逐物件的 ReID embedder — 詳見第 4 節 |
| `nvdslogger` | 一個「什麼都不做」的直通元件；`reid_pad_buffer_probe` 就掛在它的 src pad 上（純粹是因為這是兩個推論階段都跑完之後、一個方便又穩定的掛載點） |
| `nvmultistreamtiler` | 把所有攝影機畫面合成（composite）成一張拼貼畫面以供顯示（rows × columns 由 `num_sources` 計算而來） |
| `nvdsosd` | 把 probe 建立的 display meta（框、標籤、線條）畫到畫面上 |
| sink（`nveglglessink` / `nv3dsink` / `fakesink`） | 渲染到螢幕（或在 `PERF_MODE` 下直接丟棄） |

## 3. PGIE — Primary GIE（主要偵測器）

「GIE」= **G**PU **I**nference **E**ngine（GPU 推論引擎）。**主要（primary）**
GIE 是跑在*整張畫面*上的，負責產生最初的一批物件偵測結果 — 這是下游一切
（追蹤、ReID）的基礎。

設定檔：`ds_include/dstest3_pgie_config.yml`，模型：PeopleNet
（`resnet34_peoplenet_int8.onnx`，DeepStream 內建的人物偵測器）。

重要屬性：
- `gie-unique-id: 1` — PGIE 的 ID；SGIE 的設定檔會參照這個 ID，藉此知道要
  對哪個偵測器的結果進行後續處理。
- `num-detected-classes: 3`、`filter-out-class-ids: 1;2` — PeopleNet 會偵測
  人 / 包包 / 臉部三種類別；這裡把類別 1 和 2 濾掉，只保留 **person**
  （類別 0）的框。
- `infer-dims: 3;544;960` — PeopleNet 要求的固定輸入解析度；畫面在推論前
  會先被縮放 / 補邊（letterbox）到這個尺寸。
- `batch-size` — 必須等於攝影機來源數量；`main()` 會在啟動時用
  `num_sources` 覆寫設定檔裡的數值，若真的覆寫了會印出警告。
- `model-engine-file` — 已建置好的 TensorRT engine。如果對應這個
  batch size 的 engine 檔不存在或已過期，`nvinfer` 會在第一次執行時從
  `onnx-file` 重新建置（速度較慢，且每種 batch-size / 精度組合都要各自
  付一次這個成本）。

輸出：每一個被偵測到的人會變成一個 `NvDsObjectMeta`（含 bbox 與
`confidence`），掛在該幀的 `NvDsFrameMeta` 底下。

## 4. SGIE — Secondary GIE（ReID embedder）

**次要（secondary）** GIE 是跑在*主要 GIE 已經找到的物件裁切區塊*上，而不是
整張畫面 — `sgie_config.yml` 裡的 `operate-on-class-ids: 0` 意思是「對每一個
PGIE 找到的類別 0（人）物件執行」。它在這裡的目的不是分類，而是特徵擷取：
針對每一個人物裁切區塊，產生一個 256 維的外觀 embedding，供 ReID 比對使用。

設定檔：`ds_include/sgie_config.yml`，模型：一個以 Market1501 訓練的 ReID
網路（預設為 `resnet50_market1501_aicity156.onnx`；`ds_include/onnx_model/`
底下也有一個 checked-in 的 OSNet 版本，可在被註解掉的區塊中替換使用）。

重要屬性：
- `gie-unique-id: 2`、`process-mode: 2` — 標示這個 `nvinfer` 實例是次要的，
  並且串接在 PGIE（`gie-unique-id: 1`）的偵測結果之後。
- `network-type: 100` — 「其他 / 自訂網路」，因為這不是偵測器也不是分類器，
  只是輸出一個原始特徵向量。
- `output-tensor-meta: 1` — **這個最關鍵。** nvinfer 不會嘗試自己解讀輸出
  （當作框或類別分數），而是把原始輸出張量掛在該物件的 metadata 上
  （`NVDSINFER_TENSOR_OUTPUT_META`），交給下游的某個東西去讀。那個「某個
  東西」就是 pad probe。
- `net-scale-factor` / `offsets` / `model-color-format: 0`（RGB） — 送入
  網路前對像素做的正規化（normalization）；必須跟模型訓練時使用的方式一致
  （確切對應的 mean / std，可參考 `ds_include/onnx_model/deepstream_reid_config.yml`
  裡的註解）。
- `infer-dims: 3;256;128` — 每個人物裁切區塊在做 embedding 前都會被縮放成
  256×128，不管原始長寬比為何（`maintain-aspect-ratio: 0`）。

`ds_include/onnx_model/deepstream_reid_config.yml` 是一份參考 / 筆記檔，
說明如何把「同一個」ReID 模型接進 DeepStream 內建的 NvDCF / DeepSORT 追蹤器
設定格式（`ReID:` 區塊），而不是用獨立的 SGIE 來做 — 目前的管線並沒有使用
這個檔案，但如果之後要把內建追蹤器元件重新啟用的話會用得到（`app_config.yml`
裡也留有一段被註解掉的 `tracker:` 區塊，是當初還在用內建追蹤器時留下的）。

## 5. 把 embedding 取出來：pad probe

`reid_pad_buffer_probe` 掛在 `nvdslogger` 的 src pad 上，每個 batch 執行一次，
分成兩個階段：

1. **蒐集（Collect）** — 走訪 `batch_meta.frame_meta_list`，再走訪每一幀的
   `obj_meta_list`。對每一個物件，走訪 `obj_meta.obj_user_meta_list`，找出
   型別為 `NVDSINFER_TENSOR_OUTPUT_META` 的 `NvDsUserMeta`（這正是因為
   `output-tensor-meta: 1` 而由 SGIE 附加上去的張量）。這個原始 buffer 是
   一個 C 的 float 指標（`pyds.get_ptr`），會透過 `ctypes` 轉型並複製成一個
   NumPy 陣列 — 那就是 256 維的 `reid_vector`。偵測結果會依
   `frame_meta.source_id`（哪一路攝影機）分組。
2. **更新（Update）** — `mct.update_batch(detections_by_source, frame_id=...)`
   會在這一個 tick 內，對每一路攝影機各跑一次 BoT-SORT +
   `GlobalRegistry.step()`，接著把 display meta（框、
   `g<global_id> | t<track_id>`、ROI 線條）建立回各自來源的 `frame_meta`
   上，交由 `nvdsosd` 繪製。

另外還有一個 probe，`save_dets_pad_buffer_probe`，是比較簡單的替代方案：
它只是把每個 batch 的偵測結果與 embedding 傾印（dump）成一個 `.npy` 檔，
而不是就地（inline）執行追蹤器 — 適合用來擷取某次執行的原始
偵測器 / embedder 輸出，供之後重播（replay）使用。

## 6. 如何執行

1. 編輯 `ds_include/app_config.yml`：
   - `source-list.list` — 一個或多個 `file://`、`rtsp://`、`v4l2://` 的
     URI，多路攝影機以 `;` 分隔。
   - `streammux.batch-size` 應該要等於來源數量（不過 PGIE / SGIE 的
     batch-size 在啟動時反正都會被自動覆寫成相符的數值）。
   - `primary-gie.config-file-path` / `secondary-gie-1.config-file-path` —
     指向 `ds_include/dstest3_pgie_config.yml` 與 `ds_include/sgie_config.yml`
     （目前 checked-in 的值指向另一台機器的路徑，需要自行修正）。
2. 把 `ds_multi_cam_tracking.py` 檔案最上方的 `path_to_botsort_parent`，
   更新成 `botsort` 套件實際所在的位置（如果它還沒被加進
   `PYTHONPATH` 的話）。
3. 執行：
   ```bash
   python3 ds_multi_cam_tracking.py
   ```
   - `GALLERY_DIR` 環境變數 — 每個身份（identity）的裁切快照要寫到哪裡
     （預設為 `./gallery_snapshots`）。
   - `NVDS_TEST3_PERF_MODE=1` — 切換成 `nvurisrcbin` / `fakesink`，
     用於不需要顯示畫面的吞吐量（throughput）測試。
   - 在終端機中按下 **空白鍵（SPACE）** 可以暫停 / 繼續正在執行的管線。
4. 針對某個特定的 PGIE / SGIE batch-size 第一次執行時，如果對應的
   `.engine` 檔還不存在，會從 ONNX 檔重新建置 TensorRT engine — 這會有
   一次性的延遲，屬正常現象。

### 這個目錄下的其他進入點

- `ds_single_cam_tracking.py` — 同一條管線，但收斂成單一攝影機 / 單一
  `BoTSORT` 實例，沒有 `MultiCameraTracker` / 跨鏡頭比對。
- `ds_multi_cam_tracking_rtsp.py` — 使用 `nvmultiurisrcbin`（DeepStream
  將 uri-source 與 streammux 合而為一的元件）的變體版本，取代手動建立
  per-source bin 再搭配獨立的 `nvstreammux`；同時也是每路攝影機一有新的一幀
  就各自獨立更新，而不是假設每個 batch 兩路攝影機都會同時到齊。裡面有一些
  寫死 / 實驗性的部分（路徑、`if False:` 區塊）— 請把它當成一份
  仍在進行中的參考實作，而不是正式（canonical）的管線。
- `single_cam_test.py` / `multi_cam_test.py` — 離線、不需要 DeepStream 的
  視覺化工具。它們會讀回由 `save_dets_pad_buffer_probe` 寫出的 `.npy`
  batch，並用 OpenCV 把對應的框與 track ID 畫在空白畫布上，讓你可以在
  不重新執行整條 GStreamer 管線的情況下，回顧某次擷取到的執行結果。
- `visualize_reid_latent.py` — 離線的 embedding 空間視覺化工具（讀取
  `GlobalRegistry` 寫出的 `reid_log_path` JSONL / NPZ 記錄檔）。

## 7. 使用 Deepstream 時需要注意的坑

- **記憶體型別（Memory type）**：gallery snapshot 是透過
  `pyds.get_nvds_buf_surface()` 讀取像素的，這需要 host 端可讀取
  （host-mappable）的記憶體。在 dGPU（x86）上，NvBuf 預設的記憶體型別是
  GPU-only 的裝置記憶體，所以 `streammux` 與負責 RGBA 轉換的元件會明確地
  把 `nvbuf-memory-type` 設成 `NVBUF_MEM_CUDA_UNIFIED`（Jetson /
  `aarch64` 則使用 `NVBUF_MEM_DEFAULT`，因為它本來就是 unified 記憶體）。
  如果漏掉這一步會導致 segfault。
- **RGBA 轉換的位置**：`nvvideoconvert` → RGBA `capsfilter` 這一對元件是
  緊接在 muxer 之後、PGIE / SGIE 動到 buffer 之前放置的，純粹是為了讓
  snapshot 的程式碼之後能拿到一個 CPU 可讀取的畫面 — 解碼器輸出原本是
  NV12 格式。
- **`gie-unique-id` 串接關係**：PGIE 和 SGIE 其實都只是 `nvinfer` 的實例；
  真正讓第二個變成「次要（secondary）」並跟第一個串接起來的，是設定值
  （`gie-unique-id` / `operate-on-class-ids`），而不是不同的元件類型。

## 8. PoC 模型參考指南

以下是上面提到的實際模型權重（weights）的取得方式，適合想在 DeepStream
之外、以純 Python 執行這些網路的人參考。

### 下載連結

- NVIDIA PeopleNet（NGC）：<!-- TODO: link -->
- NVIDIA ReIdentificationNet（NGC）：<!-- TODO: link -->
- 遮蔽（occlusion）情境下較準確的人物偵測 YOLO 權重（見下方說明）：<!-- TODO: link -->

### wget

```bash
# PeopleNet
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_decrypted_v2.3.4/files/resnet34_peoplenet_int8.onnx --output-document resnet34_peoplenet_int8.onnx

# ReIdentificationNet
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.2/files/resnet50_market1501_aicity156.onnx --output-document resnet50_market1501_aicity156.onnx

# YOLO 人物偵測器（擁擠 / 遮蔽場景）
https://drive.google.com/file/d/190rZXaBEBOcMo1gjNG32aerveAbaX2Zm/view?usp=sharing
```

### 說明

- **PeopleNet 不太適合直接搬到純本地端（local）推論腳本上使用。**
  這份 NGC 產物是圍繞著 DeepStream 的 `nvinfer` 管線設計的 — bbox 解碼
  （decode）、anchor 處理、以及分群（`dstest3_pgie_config.yml` 裡的
  `cluster-mode`、`nms-iou-threshold`、`pre-cluster-threshold`）全都是在
  外掛（plugin）內部完成的，而不是在模型本身的計算圖裡，所以如果直接用
  純 `onnxruntime` 跑這個原始 ONNX，就得自己手動重新實作那一段
  decode / NMS 邏輯。而且它在密集 / 有遮蔽的人群場景中本來就不算是很強的
  偵測器 — 它是針對相對乾淨的 CCTV 取景角度調校的。若是要做本地端的
  Python 推論，建議改用一個針對擁擠 / 遮蔽場景訓練過的 YOLO 模型（例如
  以 CrowdHuman 訓練的 YOLOv8 / YOLOv7 checkpoint）— 走標準的 YOLO
  後處理流程，而且在人物互相重疊時明顯更準確。
- **ReIdentificationNet 則簡單得多。** 它是一個單純的 embedding-head
  CNN（不涉及任何偵測解碼），所以 NGC 提供的同一份 `.onnx` 檔可以直接用
  `onnxruntime.InferenceSession` 載入。餵給它 256×128、RGB 格式的裁切
  區塊，並用跟 `sgie_config.yml` 相同的方式正規化（`net-scale-factor` /
  `offsets` / `model-color-format: 0`），`session.run()` 回傳的就會是跟
  現在 SGIE tensor meta 給你的一模一樣的 embedding 向量。
