from nanomodel import AutoNanoModel
from nanomodel.utils.eval import EVAL

model_id = "./quantized_models/qwen3-1.7b-gptq-4bit"

# Use `lm-eval` as framework to evaluate the model
# lm_eval_data = AutoNanoModel.eval(
#     model_id, 
#     framework=EVAL.LM_EVAL,
#     tasks=[EVAL.LM_EVAL.ARC_CHALLENGE]
# )

lm_eval_data = AutoNanoModel.eval(
    model_id, 
    framework=EVAL.MMLU_PRO,
    tasks=[EVAL.MMLU_PRO.MATH],
    output_path="results/mmlu_pro"
)

# Use `evalplus` as framework to evaluate the model
# evalplus_data = AutoNanoModel.eval(model_id,
#                     framework=EVAL.EVALPLUS,
#                     tasks=[EVAL.EVALPLUS.HUMAN])