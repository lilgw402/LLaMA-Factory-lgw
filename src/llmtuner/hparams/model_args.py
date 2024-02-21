from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class ModelArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    #指定模型权重的路径
    model_name_or_path: str = field(
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    #指定适配器权重的路径
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapter weight or identifier from huggingface.co/models."}, #metadata提供这个字段的帮助信息
    )
    #用于存储从huggingface.co或modelscope.cn下载的预训练模型的目录路径。
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    #是否使用快速分词器（由tokenizers库支持）
    use_fast_tokenizer: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    #是否调整分词器词汇表和嵌入层的大小
    resize_vocab: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )
    #在分词过程中是否应该分割特殊令牌
    split_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    #默认为"main"，指定要使用的模型版本（可以是分支名、标签名或提交ID）
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model."},
    )
    quantization_type: Optional[Literal["fp4", "nf4"]] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in int4 training."},
    )
    double_quantization: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to use double quantization in int4 training."},
    )
    #可以是"linear"或"dynamic"，指定RoPE嵌入的缩放策略。
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "Which scaling strategy should be adopted for the RoPE embeddings."},
    )
    #是否启用FlashAttention-2以加快训练速度。
    flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."},
    )
    #是否启用LongLoRA提出的S^2-Attn（移位短距离注意力）。
    shift_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."},
    )
    #是否使用unsloth的优化方法进行LoRA训练。
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use unsloth's optimization for the LoRA training."},
    )
    #是否禁用梯度检查点
    disable_gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to disable gradient checkpointing."},
    )
    upcast_layernorm: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to upcast the layernorm weights in fp32."},
    )
    upcast_lmhead_output: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to upcast the output of lm_head in fp32."},
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    ms_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with ModelScope Hub."},
    )
    export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."},
    )
    #默认为1，指定导出模型文件的分片大小（单位为GB）
    export_size: Optional[int] = field(
        default=1,
        metadata={"help": "The file shard size (in GB) of the exported model."},
    )
    export_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the exported model."},
    )
    export_quantization_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the dataset or dataset name to use in quantizing the exported model."},
    )
    export_quantization_nsamples: Optional[int] = field(
        default=128,
        metadata={"help": "The number of samples used for quantization."},
    )
    export_quantization_maxlen: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum length of the model inputs used for quantization."},
    )
    export_legacy_format: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to save the `.bin` files instead of `.safetensors`."},
    )
    export_hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the repository if push the model to the Hugging Face hub."},
    )
    print_param_status: Optional[bool] = field(
        default=False,
        metadata={"help": "For debugging purposes, print the status of the parameters in the model."},
    )

    def __post_init__(self):
        self.compute_dtype = None
        self.model_max_length = None

        if self.split_special_tokens and self.use_fast_tokenizer:
            raise ValueError("`split_special_tokens` is only supported for slow tokenizers.")

        if self.adapter_name_or_path is not None:  # support merging multiple lora weights
            self.adapter_name_or_path = [path.strip() for path in self.adapter_name_or_path.split(",")]

        assert self.quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.export_quantization_bit in [None, 8, 4, 3, 2], "We only accept 2/3/4/8-bit quantization."

        if self.export_quantization_bit is not None and self.export_quantization_dataset is None:
            raise ValueError("Quantization dataset is necessary for exporting.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
