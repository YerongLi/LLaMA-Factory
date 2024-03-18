import os
import torch
import datasets
import transformers
from typing import Any, Dict, Optional, Tuple
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from llmtuner.extras.logging import get_logger
from llmtuner.extras.misc import parse_args
from llmtuner.hparams import (
    ModelArguments,
    DataArguments,
    EvaluationArguments,
    FinetuningArguments,
    GeneratingArguments
)


logger = get_logger(__name__)


_TRAIN_ARGS = [
    ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, GeneratingArguments
]
_TRAIN_CLS = Tuple[
    ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, GeneratingArguments
]
_INFER_ARGS = [
    ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments
]
_INFER_CLS = Tuple[
    ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments
]
_EVAL_ARGS = [
    ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments
]
_EVAL_CLS = Tuple[
    ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments
]


def _verify_model_args(model_args: "ModelArguments", finetuning_args: "FinetuningArguments") -> None:
    if model_args.quantization_bit is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Quantization is only compatible with the LoRA method.")

    if (
        model_args.checkpoint_dir is not None
        and len(model_args.checkpoint_dir) != 1
        and finetuning_args.finetuning_type != "lora"
    ):
        raise ValueError("Multiple checkpoints are only available for LoRA tuning.")


def parse_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    additional_choices = ['gan']
    # Parse the arguments to initialize the parser's state
    _ = parser.parse_known_args()

    # Retrieve the current choices for the --stage argument
    current_choices = parser._action_groups[1].choices['stage'].choices

    # Add 'gan' to the current choices
    additional_choices = ['gan']
    modified_choices = current_choices + additional_choices

    # Create a new parser with the modified choices
    modified_parser = argparse.ArgumentParser(parents=[parser])
    modified_parser._action_groups[1].choices['stage'].choices = modified_choices
    
    # Parse the arguments again using the modified parser
    args = modified_parser.parse_args()

    return parse_args(parser, args)


def parse_infer_args(args: Optional[Dict[str, Any]] = None) -> _INFER_CLS:
    parser = HfArgumentParser(_INFER_ARGS)
    return parse_args(parser, args)


def parse_eval_args(args: Optional[Dict[str, Any]] = None) -> _EVAL_CLS:
    parser = HfArgumentParser(_EVAL_ARGS)
    return parse_args(parser, args)


def get_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    model_args, data_args, training_args, finetuning_args, generating_args = parse_train_args(args)

    # Setup logging
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Check arguments
    data_args.init_for_training(training_args.seed)

    if finetuning_args.stage != "pt" and data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    if finetuning_args.stage not in ["sft", "gan"] and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

    if finetuning_args.stage == "sft" and training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Please enable `predict_with_generate` to save model predictions.")

    if finetuning_args.stage in ["rm", "ppo"] and training_args.load_best_model_at_end:
        raise ValueError("RM and PPO stages do not support `load_best_model_at_end`.")

    if finetuning_args.stage == "ppo" and not training_args.do_train:
        raise ValueError("PPO training does not support evaluation, use the SFT stage to evaluate models.")

    if finetuning_args.stage in ["rm", "dpo"] and (not all([data_attr.ranking for data_attr in data_args.dataset_list])):
        raise ValueError("Please use ranked datasets for reward modeling or DPO training.")

    if finetuning_args.stage == "ppo" and model_args.shift_attn:
        raise ValueError("PPO training is incompatible with S^2-Attn.")

    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")

    if training_args.do_train and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True while training.")

    if training_args.do_train and finetuning_args.finetuning_type == "lora" and finetuning_args.lora_target is None:
        raise ValueError("Please specify `lora_target` in LoRA training.")

    _verify_model_args(model_args, finetuning_args)

    if training_args.do_train and model_args.quantization_bit is not None and (not finetuning_args.upcast_layernorm):
        logger.warning("We recommend enable `upcast_layernorm` in quantized training.")

    if training_args.do_train and (not training_args.fp16) and (not training_args.bf16):
        logger.warning("We recommend enable mixed precision training.")

    if (not training_args.do_train) and model_args.quantization_bit is not None:
        logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")

    if (not training_args.do_train) and finetuning_args.stage == "dpo" and finetuning_args.ref_model is None:
        logger.warning("Specify `ref_model` for computing rewards at evaluation.")

    # postprocess training_args
    if (
        training_args.local_rank != -1
        and training_args.ddp_find_unused_parameters is None
        and finetuning_args.finetuning_type == "lora"
    ):
        logger.warning("`ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.")
        training_args_dict = training_args.to_dict()
        training_args_dict.update(dict(ddp_find_unused_parameters=False))
        training_args = Seq2SeqTrainingArguments(**training_args_dict)

    if (
        training_args.resume_from_checkpoint is None
        and training_args.do_train
        and os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError("Output directory already exists and is not empty. Please set `overwrite_output_dir`.")

        if last_checkpoint is not None:
            training_args_dict = training_args.to_dict()
            training_args_dict.update(dict(resume_from_checkpoint=last_checkpoint))
            training_args = Seq2SeqTrainingArguments(**training_args_dict)
            logger.info("Resuming training from {}. Change `output_dir` or use `overwrite_output_dir` to avoid.".format(
                training_args.resume_from_checkpoint
            ))

    if finetuning_args.stage in ["rm", "ppo"] and training_args.resume_from_checkpoint is not None:
        logger.warning("Add {} to `checkpoint_dir` to resume training from checkpoint.".format(
            training_args.resume_from_checkpoint
        ))

    # postprocess model_args
    model_args.compute_dtype = (
        torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)
    )
    model_args.model_max_length = data_args.cutoff_len

    # Log on each process the small summary:
    logger.info("Process rank: {}, device: {}, n_gpu: {}\n  distributed training: {}, compute dtype: {}".format(
        training_args.local_rank, training_args.device, training_args.n_gpu,
        bool(training_args.local_rank != -1), str(model_args.compute_dtype)
    ))
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args, generating_args


def get_infer_args(args: Optional[Dict[str, Any]] = None) -> _INFER_CLS:
    model_args, data_args, finetuning_args, generating_args = parse_infer_args(args)

    if data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    _verify_model_args(model_args, finetuning_args)

    return model_args, data_args, finetuning_args, generating_args


def get_eval_args(args: Optional[Dict[str, Any]] = None) -> _EVAL_CLS:
    model_args, data_args, eval_args, finetuning_args = parse_eval_args(args)

    if data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    _verify_model_args(model_args, finetuning_args)

    transformers.set_seed(eval_args.seed)

    return model_args, data_args, eval_args, finetuning_args
