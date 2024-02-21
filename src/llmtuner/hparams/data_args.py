from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """

    #用于训练和推断时构建提示的模板名称。如果设定了特定的模板，将按照该模板格式构建输入数据。
    template: Optional[str] = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )
    # 要使用的数据集名称。如果同时使用多个数据集，可以用逗号分隔。
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of provided dataset(s) to use. Use commas to separate multiple datasets."},
    )
    # 包含数据集的文件夹路径，默认为 `"data"`
    dataset_dir: Optional[str] = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    #用于训练和评估的数据集分割部分，默认为 `"train"`
    split: Optional[str] = field(
        default="train",
        metadata={"help": "Which dataset split to use for training and evaluation."},
    )
    #分词后模型输入的截断长度，默认为1024。
    cutoff_len: Optional[int] = field(
        default=1024,
        metadata={"help": "The cutoff length of the model inputs after tokenization."},
    )
    #分词后为标签预留的最小截断长度，默认为1。
    reserved_label_len: Optional[int] = field(
        default=1,
        metadata={"help": "The minimum cutoff length reserved for label after tokenization."},
    )
    #是否在提示时禁用掩码，默认为 `False`。
    train_on_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to disable the mask on the prompt or not."},
    )
    #是否启用数据集流式传输，默认为 `False`。
    streaming: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )
    #在数据集流式传输中随机抽取示例时使用的缓冲区大小，默认为16384。
    buffer_size: Optional[int] = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."},
    )
    #数据集混合策略（合并/交织），以及交织时的下采样或上采样策略，默认为 `"concat"`。
    mix_strategy: Optional[Literal["concat", "interleave_under", "interleave_over"]] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling)."},
    )
    #不同数据集的抽样概率，多个数据集时用逗号分隔，默认值为 `None`。
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."},
    )
    #是否覆盖缓存的训练和评估数据集，默认为 `False`
    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    #用于预处理的进程数，默认值为 `None`，表示不指定
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."},
    )
    eval_num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"},
    )
    #计算损失时是否忽略与填充标签对应的令牌，默认为 `True`。
    ignore_pad_token_for_loss: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether or not to ignore the tokens corresponding to padded labels in the loss computation."
        },
    )
    val_size: Optional[float] = field(
        default=0,
        metadata={"help": "Size of the development set, should be an integer or a float in range `[0,1)`."},
    )
    sft_packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Packing the questions and answers in the supervised fine-tuning stage."},
    )
    cache_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save or load the preprocessed datasets."},
    )

    def __post_init__(self):
        if self.reserved_label_len >= self.cutoff_len:
            raise ValueError("`reserved_label_len` must be smaller than `cutoff_len`.")

        if self.streaming and self.val_size > 1e-6 and self.val_size < 1:
            raise ValueError("Streaming mode should have an integer val size.")

        if self.streaming and self.max_samples is not None:
            raise ValueError("`max_samples` is incompatible with `streaming`.")
