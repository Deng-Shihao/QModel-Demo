from nanomodel import AutoNanoModel
from nanomodel.utils.eval import EVAL

model_id = "/home/sd24191/git_project/QModel-Demo/quantized_models/Qwen3-4B-Instruct-2507-GPTQ-4bit"
# model_id = "/home/sd24191/git_project/QModel-Demo/quantized_models/Qwen3-4B-Instruct-2507-AWQ-4bit"

# `lm-eval` framework evaluate the model
lm_eval_data = AutoNanoModel.eval(
    model_id, framework=EVAL.LM_EVAL, tasks=[EVAL.LM_EVAL.ARC_CHALLENGE]
)

# evalplus framework evaluate the model
evalplus_data = AutoNanoModel.eval(
    model_id, framework=EVAL.EVALPLUS, tasks=[EVAL.EVALPLUS.HUMAN]
)
