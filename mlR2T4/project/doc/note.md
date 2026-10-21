ps:以下是在与ai的对话中总结的笔记


# 原理理解（编码前必须搞清楚）
1. HTTP 请求的四个组成部分

|部分|	内容|	在本任务中的体现|作用|
|:----|:----|:----|:----|
|方法 Method|	GET / POST / PUT / DELETE|	本任务必须用 POST（携带请求体）|要做什么操作？|
|URL|	协议://主机:端口/路径|	http://127.0.0.1:5000/predict|对谁做？（资源地址）|
|头部 Headers|	键值对元信息|	Content-Type、Authorization、User-Agent|怎么做 / 附加说明？|
|请求体 Body|	实际数据|	表单字段 Operation/Data_A/Data_B 或上传文件|带什么数据？|

2. Content-Type 与请求体格式的对应关系（最容易出错的地方）

|客户端 Body 选择|	实际 Content-Type|	服务端用哪个 API 解析|
|:----|:----|:----|
|form-data|	multipart/form-data; boundary=...|	request.form（文本字段）+ request.files（文件）|
|x-www-form-urlencoded|	application/x-www-form-urlencoded|	request.form|
|json|	application/json|	request.get_json()|
|raw / binary|	任意|	request.data / request.get_data()|

关键结论：Flask 的 request.form 同时能解析 multipart/form-data 和 x-www-form-urlencoded，所以题目要求的表单数据用 request.form 即可；只有 JSON 才用 get_json()。

常见的"接不到数据"故障，90% 是 Apifox 里 Body 类型选错（比如选了 JSON 却在代码里读 request.form），或 手动写了一个错误的 Content-Type 头。不要在 Apifox 里手写 Content-Type，选了 Body 类型它会自动生成（multipart 还需要自动生成 boundary，手写必错）。

3. 状态码语义（本任务使用）

|码|	含义|	本任务触发场景|
|:----|:----|:----|
|200|	成功|	计算成功|
|400|	请求错误|	缺字段、未知 Operation、除零|
|401|	未认证|	Authorization 头错误|
|405|	方法不允许|	用 GET 访问 /predict|
|415|	媒体类型不支持|	上传文件扩展名与模型不匹配|
|422|	语义/格式错误|	Data_A 不是数字|
|500|	服务器内部错误|	计算函数抛出未预期异常|

4. 请求处理的完整流水线（本题要求的核心逻辑）

接收请求 → 解析请求(读头 + 按 Content-Type 解析 Body)
        → 预处理(类型转换、校验、解码)
        → 调用模型/计算函数
        → 后处理(结果转成可序列化结构)
        → 构造响应(JSON + 状态码 + 头)
        → 返回客户端



# calculate
```
# calculate.py
SYMBOLS = {"sum", "multiply", "subtract", "divide"}


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def calculate(symbol: str, a: float, b: float) -> float:
    """
    异常约定（由调用方 app.py 映射为 HTTP 状态码）：
      TypeError         -> 422  operation 非字符串 / a、b 非数字
      ValueError        -> 400  未知操作
      ZeroDivisionError -> 400  除数为 0
    """
    if not isinstance(symbol, str):
        raise TypeError(f"operation must be str, got {type(symbol).__name__}")

    s = symbol.strip().lower()   # 去除多余空格并转小写

    if s not in SYMBOLS:
        raise ValueError(f"unsupported operation: {symbol}")

    if not _is_number(a) or not _is_number(b):
        raise TypeError(f"{a!r} or {b!r} is not a number")

    if s == "sum":
        return a + b
    elif s == "subtract":
        return a - b
    elif s == "multiply":
        return a * b
    elif s == "divide":
        if b == 0:
            raise ZeroDivisionError("divisor is zero")
        return a / b
    else:
        raise ValueError(f"unsupported operation: {symbol}")
```

# app
1. 验证能否用
```
# app.py（临时最小版）
from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
```
运行后访问http://127.0.0.1:5000/health 或者 用power shell 输入curl http://127.0.0.1:5000/health
在终端按ctrl+c结束

2. 正式程序
```
import os
import math
import logging

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from calculator import calculate

app = Flask(__name__)    #定位模块、区分直接运行/被 import
app.logger.setLevel(logging.INFO)     #只输出 INFO 及以上级别的日志，DEBUG 被过滤掉。
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10MB

MODEL_NAME = "calculator-v1"
EXPECTED_AUTH = os.getenv("API_TOKEN", "Bearer demo-token")


def get_request_data():
    """表单优先，兼容 JSON。"""
    if request.is_json:
        return request.get_json(silent=True) or {}
    return request.form


@app.route("/predict", methods=["POST"])
def predict():
    # ① 读取头部
    auth = request.headers.get("Authorization")
    content_type = request.headers.get("Content-Type", "")
    if auth and auth != EXPECTED_AUTH:
        return jsonify({"error": "Unauthorized",
                        "message": "Authorization 头部无效"}), 401

    # ② 解析请求体
    data = get_request_data()
    op, a_raw, b_raw = data.get("Operation"), data.get("Data_A"), data.get("Data_B")

    if op is None or a_raw is None or b_raw is None:
        return jsonify({"error": "Bad Request",
                        "message": "请求体必须包含 Operation、Data_A、Data_B",
                        "received_content_type": content_type}), 400

    op = str(op).strip().lower()

    # ③ 预处理
    try:
        a, b = float(a_raw), float(b_raw)
    except (TypeError, ValueError):
        return jsonify({"error": "Unprocessable Entity",
                        "message": "Data_A 和 Data_B 必须是数字"}), 422

    if not (math.isfinite(a) and math.isfinite(b)):
        return jsonify({"error": "Unprocessable Entity",
                        "message": "Data_A 和 Data_B 必须是有限数字"}), 422

    # ④ 调用本地另一个 py 程序（模型）
    try:
        result = calculate(op, a, b)
    except ZeroDivisionError:
        return jsonify({"error": "Bad Request", "message": "除数不能为 0"}), 400
    except ValueError as e:
        return jsonify({"error": "Bad Request", "message": str(e)}), 400
    except Exception:
        app.logger.exception("推理异常")          # 堆栈进日志，不进响应
        return jsonify({"error": "Internal Server Error",
                        "message": "预测失败"}), 500

    # ⑤ 后处理
    if isinstance(result, float) and not math.isfinite(result):
        return jsonify({"error": "Unprocessable Entity",
                        "message": "计算结果不是有限数字"}), 422
    if isinstance(result, float) and result.is_integer():
        result = int(result)

    # ⑥ 构造并返回响应
    resp = jsonify({"Result": result})
    resp.headers["X-Model-Name"] = MODEL_NAME
    resp.headers["X-Request-Content-Type"] = content_type
    return resp, 200


# ---------- 错误处理器：保证任何情况下都返回 JSON ----------

@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Not Found", "message": "接口不存在"}), 404


@app.errorhandler(405)
def handle_405(e):
    return jsonify({"error": "Method Not Allowed",
                    "message": "请使用 POST 方法"}), 405


@app.errorhandler(413)
def handle_413(e):
    return jsonify({"error": "Payload Too Large",
                    "message": "上传文件超过大小限制"}), 413


if __name__ == "__main__":
    # 生产环境请用 gunicorn 部署，不要开 debug
    app.run(host="127.0.0.1", port=5000, debug=True)
```

logging 有五个标准级别（从低到高）：

|级别|	数值|	用途|
|:---|:---|:---|
|DEBUG|	10|	调试细节（变量值、SQL 语句）|
|INFO|	20|	正常运行信息（请求进来、模型加载完）|
|WARNING|	30|	值得注意但不影响运行（默认级别）|
|ERROR|	40|	出错了但服务还能跑（异常捕获）|
|CRITICAL|	50|	严重故障（数据库连不上）|


# 解析文件，部署模型
## 第一部分：先理解"文件上传"这件事本身
1. 上传的文件在 HTTP 里长什么样
```
Apifox 里 Body 选 form-data、字段类型选 File，实际发出的请求体是 multipart/form-data：

http
Content-Type: multipart/form-data; boundary=----abc123

------abc123
Content-Disposition: form-data; name="model_name"

image_classifier
------abc123
Content-Disposition: form-data; name="file"; filename="cat.jpg"
Content-Type: image/jpeg

<cat.jpg 的二进制字节>
------abc123--
```
关键点：
每个字段用 boundary 分隔；
文件字段比文本字段多两样东西：filename="cat.jpg" 和 Content-Type: image/jpeg；

2. Flask 怎么接住它

|请求里的位置|	Flask读取方式	|得到什么|
|---|---|---|
|文本字段（model_name）|	request.form.get("model_name")|	str|
|文件字段（file）|	request.files.get("file")|	FileStorage 对象|

FileStorage 对象有这些属性/方法：

```
file = request.files.get("file")

file.filename        # "cat.jpg"（客户端声明的文件名）
file.content_type    # "image/jpeg"（客户端声明的 MIME）
file.read()          # b'\xff\xd8\xff...'（全部字节，读完指针到末尾）
file.seek(0)         # 回到开头，再 read() 能再读一次
file.save("a.jpg")   # 直接存到磁盘
file.stream          # 底层的文件流对象（适合大文件流式处理）
```
坑：file.read() 只能读一次——读完指针在末尾，再读得到 b""。要重读必须先 file.seek(0)。

3. 文件校验的三个层次

```
# ① 文件是否存在
if file is None or file.filename == "":
    return ..., 400

# ② 扩展名是否与模型匹配（题目"模型与文件类型匹配"的要求）
from pathlib import Path
ext = Path(file.filename).suffix.lower()          # ".jpg"
allowed = {".jpg", ".jpeg", ".png"}
if ext not in allowed:
    return ..., 415

# ③ 内容是否为空
content = file.read()
if not content:
    return ..., 400
```
为什么用 Path(...).suffix 而不是 secure_filename：secure_filename("中文图.png") 会变成 "png"（中文被清掉，扩展名也没了），而 Path("中文图.png").suffix 得到 ".png"，对中文文件名友好。

## 第二部分：真实模型接入的完整链路（以 ResNet18 为例）

1. 模型加载：必须在"服务启动时"做一次
```
# app.py 顶部，模块级
import torch
from torchvision import models, transforms
from PIL import Image
import io

# ① 加载预训练模型
MODEL = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
MODEL.eval()                    # 切到推理模式（关掉 Dropout/BatchNorm 的训练行为）

# ② 类别名（ImageNet 1000 类）
CLASSES = models.ResNet18_Weights.DEFAULT.meta["categories"]
```
三个关键点：

|点	|原因|
|---|---|
|模块级加载|	每个请求都加载模型 = 每次几百 MB 内存 + 几秒延迟，服务直接崩|
|.eval()|	Dropout 在推理时会随机关神经元、BatchNorm 用当前 batch 统计量 → 结果不稳定、不准确|
|CLASSES 从权重元数据取|	手写 1000 个类名容易错、顺序容易乱 → 顺序错 = 类别全错|
2. 预处理：必须和训练时完全一致
ResNet18 训练时的预处理是固定的，推理时必须复刻：

```
PREPROCESS = transforms.Compose([
    transforms.Resize(256),                    # ① 短边缩到 256
    transforms.CenterCrop(224),                # ② 中心裁 224×224
    transforms.ToTensor(),                     # ③ PIL → Tensor，HWC→CHW，归一化到 0-1
    transforms.Normalize(                      # ④ 按 ImageNet 的均值/标准差标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```


3. 推理 + 后处理：完整代码
```
def predict_image(content: bytes) -> dict:
    """
    输入：图片字节
    输出：{"label": "cat", "score": 0.98}
    """
    # 解码：bytes → PIL.Image，统一 RGB
    img = Image.open(io.BytesIO(content)).convert("RGB")

    # 预处理：PIL → Tensor(3,224,224)
    tensor = PREPROCESS(img)

    # ③ 加 batch 维度：Tensor(3,224,224) → Tensor(1,3,224,224)
    batch = tensor.unsqueeze(0)

    # 推理
    with torch.no_grad():
        logits = MODEL(batch)              # Tensor(1, 1000)

    # 后处理
    probs = torch.softmax(logits, dim=1)[0]   # Tensor(1000)，和为 1
    idx = int(probs.argmax())                 # 最大概率的下标
    score = float(probs[idx])                 # 最大概率的值

    return {
        "label": CLASSES[idx],                # 类别名，如 "tabby cat"
        "score": round(score, 4),             # 0.9876
    }
```
4. 逐行解释"为什么这么写"
    1. Image.open(io.BytesIO(content))
    content 是 bytes，Image.open 要"文件对象"。io.BytesIO 把 bytes 包装成内存文件——不需要写磁盘，快且干净。

    2. .convert("RGB")
        PNG 可能是 RGBA（4 通道）、GIF 可能是 P（调色板）、扫描件可能是 L（灰度）。模型要 3 通道，不转换会形状不匹配。

    3. .unsqueeze(0)
    模型训练时是批量训练的（一次 32 张），输入形状是 (N, C, H, W)。推理单张图也要"假装是 1 张的批量"，形状 (1, 3, 224, 224)。不加会报维度错误。

    4. torch.no_grad()
    默认 PyTorch 会记录每个操作（为了反向传播）。推理不需要梯度，no_grad() 关掉记录 → 内存减半、速度快 20–30%。这是推理代码的标配。


    6.  float(probs[idx])
    probs[idx] 是 torch.Tensor，不能直接塞进 jsonify（会抛 TypeError: not JSON serializable）。必须 float(...) 转成原生 Python float——这和你代码里 np.generic.item() 是同一个道理。

## 第三部分：把两部分拼起来 —— 完整的 /predict_file
```
python
import io
import torch
from pathlib import Path
from PIL import Image
from torchvision import models, transforms

# ========== 模型加载（启动时一次） ==========
MODEL = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
MODEL.eval()
CLASSES = models.ResNet18_Weights.DEFAULT.meta["categories"]

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ========== 工具函数 ==========
def predict_image(content: bytes) -> dict:
    img = Image.open(io.BytesIO(content)).convert("RGB")
    tensor = PREPROCESS(img).unsqueeze(0)
    with torch.no_grad():
        logits = MODEL(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    idx = int(probs.argmax())
    return {"label": CLASSES[idx], "score": round(float(probs[idx]), 4)}


# ========== 接口 ==========
@app.route("/predict_file", methods=["POST"])
def predict_file():
    # ---- ① 解析请求 ----
    file = request.files.get("file")
    model_name = request.form.get("model_name", "image_classifier")

    # ---- ② 校验 ----
    if file is None or file.filename == "":
        return jsonify({"error": "Bad Request",
                        "message": "缺少 file 文件字段"}), 400

    ext = Path(file.filename).suffix.lower()
    allowed = {".jpg", ".jpeg", ".png"} if model_name == "image_classifier" else set()
    if ext not in allowed:
        return jsonify({
            "error": "Unsupported Media Type",
            "message": f"文件类型 {ext} 与模型 {model_name} 不匹配",
            "allowed_exts": sorted(allowed)
        }), 415

    content = file.read()
    if not content:
        return jsonify({"error": "Bad Request", "message": "上传文件为空"}), 400

    # ---- ③ 预处理 + ④ 推理 + ⑤ 后处理 ----
    try:
        result = predict_image(content)
    except Exception:
        app.logger.exception("图片推理异常 filename=%r size=%d",
                             file.filename, len(content))
        return jsonify({"error": "Internal Server Error",
                        "message": "图片推理失败"}), 500

    # ---- ⑥ 构造响应 ----
    resp = jsonify({"Result": result})
    resp.headers["X-Model-Name"] = model_name
    return resp, 200
```

## 第四部分：验证它能跑
1. 启动
```
python app.py
```
首次启动会下载 ResNet18 权重（约 45MB），要等一会儿，下载完缓存在 ~/.cache/torch/，下次秒开。

2. 准备一张测试图
随便找一张猫/狗的照片，保存为 test.jpg。

3. 用 curl 测试
bash
curl -X POST http://127.0.0.1:5000/predict_file \
  -H "Authorization: Bearer demo-token" \
  -F "file=@test.jpg" \
  -F "model_name=image_classifier"
预期输出（类别名可能是 tabby cat、Egyptian cat、golden retriever 等）：

```
{"Result":{"label":"tabby cat","score":0.8734}}
```
4.   Apifox 测试
方法 POST，URL http://127.0.0.1:5000/predict_file

Headers 加 Authorization: Bearer demo-token

Body → form-data

添加两行：

file，类型选 File，选本地图片

model_name，类型 Text，值 image_classifier

发送 → 看到 {"Result":{"label":"...","score":...}}

5. 测试错误分支
```
# 415：上传 .txt
echo "hello" > test.txt
curl -X POST http://127.0.0.1:5000/predict_file \
  -H "Authorization: Bearer demo-token" \
  -F "file=@test.txt" -F "model_name=image_classifier"
# → {"error":"Unsupported Media Type",...}  415

# 400：不传文件
curl -X POST http://127.0.0.1:5000/predict_file \
  -H "Authorization: Bearer demo-token" -F "model_name=image_classifier"
# → {"error":"Bad Request","message":"缺少 file 文件字段"}  400
```

