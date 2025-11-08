from nanomodel import AutoNanoModel
from nanomodel.utils.eval import EVAL

model_id = "./quantized_models/qwen3-1.7b-gptq-4bit"

# Use `lm-eval` as framework to evaluate the model
lm_eval_data = AutoNanoModel.eval(model_id, 
                    framework=EVAL.LM_EVAL, 
                    tasks=[EVAL.LM_EVAL.ARC_CHALLENGE])


# Use `evalplus` as framework to evaluate the model
evalplus_data = AutoNanoModel.eval(model_id, 
                    framework=EVAL.EVALPLUS, 
                    tasks=[EVAL.EVALPLUS.HUMAN])·