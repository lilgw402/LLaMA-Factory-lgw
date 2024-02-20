from typing import TYPE_CHECKING, Optional, Tuple

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import AutoModelForCausalLMWithValueHead

from ..extras.logging import get_logger
from ..extras.misc import count_parameters, get_current_device, try_download_model_from_ms
from .adapter import init_adapter
from .patcher import patch_config, patch_model, patch_tokenizer, patch_valuehead_model
from .utils import load_valuehead_params, register_autoclass


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from ..hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)

#旨在加载预训练模型和分词器。该函数支持训练和推断(inference)模式
def load_model_and_tokenizer(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: Optional[bool] = False, #指示模型是否应该是可训练的。
    add_valuehead: Optional[bool] = False, #是否增加价值头
) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """

    try_download_model_from_ms(model_args) #函数尝试从微软处下载模型

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir, #缓存下载的预训练模型和分词器
        "revision": model_args.model_revision, #指定模型的修订版（revision
        "token": model_args.hf_hub_token,
    }

    #加载预训练的分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="right",
        **config_kwargs,
    ) #ChatGLMTokenizer
    patch_tokenizer(tokenizer)

    #model的参数config，用于初始化model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs) #加载模型配置 ChatGLMConfig 
    patch_config(config, tokenizer, model_args, config_kwargs, is_trainable)

    model = None
    if is_trainable and model_args.use_unsloth:
        from unsloth import FastLanguageModel  # type: ignore

        unsloth_kwargs = {
            "model_name": model_args.model_name_or_path,
            "max_seq_length": model_args.model_max_length,
            "dtype": model_args.compute_dtype,
            "load_in_4bit": model_args.quantization_bit == 4,
            "token": model_args.hf_hub_token,
            "device_map": {"": get_current_device()},
            "rope_scaling": getattr(config, "rope_scaling", None),
        }
        try:
            model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs) #尝试加载一个来自 `unsloth` 库的快速语言模型。
        except NotImplementedError:
            logger.warning("Unsloth does not support model type {}.".format(getattr(config, "model_type", None)))
            model_args.use_unsloth = False

        if model_args.adapter_name_or_path:
            model_args.adapter_name_or_path = None
            logger.warning("Unsloth does not support loading adapters.")

    #如果 `model` 未被初始化或 `use_unsloth` 被禁用，将使用 `AutoModelForCausalLM.from_pretrained` 加载标准的预训练因果语言模型（通常用于文本生成任务）。
    if model is None:
        #用config实例化模型
        model = AutoModelForCausalLM.from_pretrained( #model: ChatGLMForConditionalGeneration
            model_args.model_name_or_path,
            config=config,
            torch_dtype=model_args.compute_dtype,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            **config_kwargs,
        )

    patch_model(model, tokenizer, model_args, is_trainable)
    register_autoclass(config, model, tokenizer) #登记模型和分词器的“自动类”。

    #通过 `init_adapter` 函数初始化模型上的适配器（如果适配器路径被指定的话）。适配器通常用于模型微调任务
    model = init_adapter(model, model_args, finetuning_args, is_trainable) #loar在q，k，v加少量参数微调

    #如果 `add_valuehead` 为 `True`，则在已有的因果语言模型上进一步加载价值头（这是一个额外输出化层，可能用于某些强化学习任务）
    if add_valuehead:
        
        model: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        #使用 `load_valuehead_params` 函数加载价值头参数，并将其与模型的状态字典合并。
        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info("Loaded valuehead from checkpoint: {}".format(vhead_path))

    if not is_trainable:
        #如果 `is_trainable` 为 `False`，则将模型设置为不可训练状态，并移到其指定的数据类型，如果模型指定了量化方法，则不做转换，并将模型置于评估模式。
        model.requires_grad_(False)
        model = model.to(model_args.compute_dtype) if not getattr(model, "quantization_method", None) else model
        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model) #使用 `count_parameters` 函数计算模型的可训练参数和全部参数的数量，并打印相应的信息。
    logger.info(
        "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    )

    if not is_trainable:
        logger.info("This IS expected that the trainable params is 0 if you are using model for inference only.")

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

    return model, tokenizer

