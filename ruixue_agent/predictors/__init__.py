"""预测子系统(predictors):地膜性能预测模型(DR/WVTR/TS)的训练产物与服务加载。

与 rag/ persistence/ 平级,属 harness(不认识 HTTP)。
- schema.py  每个模型的变量字典(特征顺序/单位/合理范围/默认值)
- (后续)registry.py 加载模型+模型卡并校验;predict.py 预测封装
训练在 scripts/train/(离线),此处只负责"服务端"加载与预测。
"""
