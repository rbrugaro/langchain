#WORKS FOR INTEL MODEL
#model_name = "Intel/bge-small-en-v1.5-rag-int8-static" #works with int8 and ipex backend

#FAILS FOR SENTENCE TRANSFORMER MODEL
#model_name = "intfloat/E5-mistral-7b-instruct"


from transformers import AutoConfig
from optimum.intel import IPEXModel

try:
    # use TorchScript model
    config = AutoConfig.from_pretrained("/home/rbrugaro/.cache/huggingface/hub/models--intfloat--E5-mistral-7b-instruct/snapshots/07163b72af1488142a360786df853f237b1a3ca1")
    export = not getattr(config, "torchscript", False)
except RuntimeError:
    #"We will use IPEXModel with export=True to export the model"
    export = True
try:
    transformer_model = IPEXModel.from_pretrained(
        "/home/rbrugaro/.cache/huggingface/hub/models--intfloat--E5-mistral-7b-instruct/snapshots/07163b72af1488142a360786df853f237b1a3ca1", export=export,
    )
    #huggingface/hub/models--Intel--bge-small-en-v1.5-rag-int8-static/snapshots/48f830c1bbf12857661626982f2d4f43601b48a5
except Exception as e:
    raise Exception(
        f"""Failed to load model {model_name}, due to the following error:
        {e}
        """
    )

print(transformer_model)
