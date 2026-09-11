import copy
import logging
from collections.abc import Mapping

from slime.utils.mask_utils import MultiTurnLossMaskGenerator

__all__ = ["build_prompt_answer_example", "generate_rollout"]

logger = logging.getLogger(__name__)


TOKENIZER = None
MASK_GENERATOR = None
SAMPLE_PRINTED = False


def _training_message(message, *, trainable):
    if not isinstance(message, Mapping):
        raise ValueError(f"SFT message must be an object, got {type(message)}")
    cleaned = copy.deepcopy(dict(message))
    cleaned.pop("thinking", None)
    cleaned.pop("reasoning", None)
    cleaned["step_loss_mask"] = int(trainable)
    return cleaned


def build_prompt_answer_example(prompt, answer, *, mask_generator, tools=None):
    """Render a prompt plus its separate answer with loss on the answer only."""
    if not isinstance(prompt, list):
        raise ValueError(f"prompt-answer SFT expects a message list, got {type(prompt)}")
    if not isinstance(answer, Mapping) or answer.get("role") != "assistant":
        raise ValueError("prompt-answer SFT requires an Assistant answer object")

    messages = [_training_message(message, trainable=False) for message in prompt]
    messages.append(_training_message(answer, trainable=True))
    token_ids, loss_mask = mask_generator.get_loss_mask(messages, tools=tools)
    if len(token_ids) != len(loss_mask):
        raise ValueError(
            "Prompt-answer SFT produced mismatched token and loss-mask lengths: "
            f"{len(token_ids)} != {len(loss_mask)}"
        )
    if not any(loss_mask):
        raise ValueError("Prompt-answer SFT produced an empty answer loss mask")
    return messages, token_ids, loss_mask


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    assert not evaluation
    assert args.rollout_global_dataset

    global TOKENIZER, MASK_GENERATOR, SAMPLE_PRINTED
    if TOKENIZER is None:
        from slime.utils.processing_utils import load_tokenizer

        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    if MASK_GENERATOR is None:
        MASK_GENERATOR = MultiTurnLossMaskGenerator(
            TOKENIZER, tokenizer_type=args.loss_mask_type
        )

    samples = data_buffer.get_samples(args.rollout_batch_size)
    for index, group in enumerate(samples):
        (sample,) = group
        tools = sample.metadata.get("tools")
        messages, token_ids, loss_mask = build_prompt_answer_example(
            sample.prompt,
            sample.label,
            mask_generator=MASK_GENERATOR,
            tools=tools,
        )
        response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]
        if response_length <= 0:
            raise ValueError("Prompt-answer SFT produced a zero-length answer")

        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0
        sample.loss_mask = loss_mask[-response_length:]

        if index == 0 and not SAMPLE_PRINTED:
            logger.info(
                "prompt_answer_sft_rollout example: sample=%s messages=%s "
                "token_ids=%s loss_mask=%s response_length=%s",
                sample,
                messages,
                token_ids,
                loss_mask,
                response_length,
            )
            SAMPLE_PRINTED = True

    return samples
