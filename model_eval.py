from nanomodel import AutoNanoModel
from nanomodel.utils.eval import EVAL

model_id = "ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1"

# Use `lm-eval` as framework to evaluate the model
lm_eval_data = AutoNanoModel.eval(model_id, 
                    framework=EVAL.LM_EVAL, 
                    tasks=[EVAL.LM_EVAL.ARC_CHALLENGE])


# Use `evalplus` as framework to evaluate the model
evalplus_data = AutoNanoModel.eval(model_id, 
                    framework=EVAL.EVALPLUS, 
                    tasks=[EVAL.EVALPLUS.HUMAN])