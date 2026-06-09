from ultralytics.models.sam import SAM3SemanticPredictor

# 1. 明确使用语义预测器接口，并通过字典传入参数
# 请确保你的环境下已下载 sam3.pt，或者在运行代码时联网让其自动下载
overrides = dict(conf=0.25, task="segment", mode="predict", model="sam3.pt", half=True)

# 2. 初始化预测器
predictor = SAM3SemanticPredictor(overrides=overrides)

# 3. 执行语义分割（传入文本 prompt）
# 这里的 set_image 用于预加载图像，后续可以对同一张图进行多次不同 prompt 的查询
predictor.set_image("./asset/img3.jpg")
results = predictor(prompt="grass")

# 4. 可视化结果
results[0].show()
