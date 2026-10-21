import os
import math
import logging
import numpy as np
from flask import Flask, request, jsonify

from caculate import calculate,UnsupportedOperation

MAX_UPLOAD_BYTES = 10 * 1024 * 1024    # 10MB
app = Flask(__name__)
# 开发模式
app.debug = True
app.logger.setLevel(logging.INFO)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

MODEL_NAME = "calculator-v1"
EXPECTED_AUTH = os.getenv("API_TOKEN", "")
if not EXPECTED_AUTH:
    if app.debug:
        EXPECTED_AUTH = "Bearer demo-token"
        app.logger.warning("开发模式：使用 demo token")
    else:
        raise RuntimeError("生产环境必须设置 API_TOKEN")
# 检查服务器是否正常运行

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME}), 200

# 获取表单或者json内容
def get_request_data():
    if request.is_json:
        return request.get_json(silent=True) or {}
    return request.form

@app.route("/predict", methods=["POST"])
def predict():
    # 读取头部
    auth = request.headers.get("Authorization")
    content_type = request.headers.get("Content-Type", "")
    if auth != EXPECTED_AUTH:
        return jsonify({"error": "Unauthorized",
                        "message": "Authorization 头部无效"}), 401

    # 解析请求体
    data = get_request_data()
    op, a_raw, b_raw = data.get("Operation"), data.get("Data_A"), data.get("Data_B")

    if op is None or a_raw is None or b_raw is None:
        return jsonify({"error": "Bad Request",
                        "message": "请求体必须包含 Operation、Data_A、Data_B",
                        "received_content_type": content_type}), 400

    op = str(op).strip().lower()

    # 预处理
    try:
        a, b = float(a_raw), float(b_raw)
    except (TypeError, ValueError):
        return jsonify({"error": "Unprocessable Entity",
                        "message": "Data_A 和 Data_B 必须是数字"}), 422

    if not (math.isfinite(a) and math.isfinite(b)):
        return jsonify({"error": "Unprocessable Entity",
                        "message": "Data_A 和 Data_B 必须是有限数字"}), 422

    # 调用模型
    try:
        result = calculate(op, a, b)
    except ZeroDivisionError:
        return jsonify({"error": "Bad Request", "message": "除数不能为 0"}), 400
    except UnsupportedOperation as e:
        return jsonify({"error": "Bad Request", "message": str(e)}), 400
    except Exception:
        app.logger.exception(f"推理异常 op={op} a={a} b={b}")          # 堆栈进日志，不进响应
        return jsonify({"error": "Internal Server Error",
                        "message": "预测失败"}), 500

    # 后处理
    if isinstance(result, np.generic):          # 处理非原生数据
        result = result.item()
    if isinstance(result, float) and not math.isfinite(result):
        return jsonify({"error": "Unprocessable Entity",
                        "message": "计算结果不是有限数字"}), 422
    if isinstance(result, float) and result.is_integer():   # 将小数中无意义的零去掉
        result = int(result)
    

    # ⑥ 构造并返回响应
    resp = jsonify({"Result": result})
    resp.headers["X-Model-Name"] = MODEL_NAME
    resp.headers["X-Request-Content-Type"] = content_type
    return resp, 200


# 错误处理器

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
                    "message": f"上传文件超过大小{MAX_UPLOAD_BYTES}限制"}), 413

@app.errorhandler(500)
def handle_500(e):
    app.logger.exception("未处理异常")
    return jsonify({"error": "Internal Server Error",
                    "message": "服务器内部错误"}), 500

if __name__ == "__main__":
    # 生产环境请用 gunicorn 部署，不要开 debug
    app.run(host="127.0.0.1", port=5000, debug=True)