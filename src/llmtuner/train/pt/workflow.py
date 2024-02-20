# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py

import math
from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForLanguageModeling, Trainer

from ...data import get_dataset, split_dataset
from ...extras.ploting import plot_loss
from ...model import load_model_and_tokenizer
from ...train.utils import create_modelcard_and_push


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_pt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    breakpoint()
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train)
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="pt")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **split_dataset(dataset, data_args, training_args),
    )

    # Training
    #启动模型的训练过程、保存训练后的模型、记录和保存训练的指标，并在训练结束后进行可视化处理。
    if training_args.do_train:
        #resume_from_checkpoint` 参数通常用于指定一个检查点（checkpoint），从中恢复训练，这在训练过程被中断需要重启时非常有用。
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint) #检查点：None
        trainer.save_model() #训练完成后，此行代码用于保存训练后的模型
        trainer.log_metrics("train", train_result.metrics) #记录训练过程的指标（例如损失或准确率等）
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state() #保存训练器的状态，包括优化器、调度器等的状态。这对于以后可能需要从最后一次训练中断的地方恢复训练是必需的。
        if trainer.is_world_process_zero() and finetuning_args.plot_loss: #检查当前进程是否是多进程环境中的主进程（通常在使用分布式训练时使用）
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"]) #绘制损失曲线图

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval") #收集评估指标
        #计算模型的困惑度（perplexity），这是一个衡量模型性能的指标，特别是在语言模型中。它是评估损失的指数（`exp`），其中评估损失通常是交叉熵损失。
        #由于指数函数可能会因输入值过大而造成溢出，所以使用了 `try...except` 语句来捕获 `OverflowError` 异常。如果出现溢出错误，困惑度被设置为无穷大 
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        metrics["perplexity"] = perplexity #将计算得到的困惑度添加到 `metrics` 字典中。
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    #模型卡片是一种文档，提供了关于模型性能、训练数据、使用方法的详细信息，并且常常用来提高模型透明度和可解释性
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
