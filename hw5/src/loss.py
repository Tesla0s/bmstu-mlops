"""Считать вероятности только там, где есть метки, сохраняя весь контекст."""

import torch
import torch.nn.functional as F

from src.data import LABEL_PAD_ID


def supervised_loss_sum(model, batch: dict) -> tuple[torch.Tensor, int]:
    """Токен на позиции t предсказывает метку t+1; промпт читается целиком."""
    labels = batch["labels"][:, 1:]
    positions = torch.nonzero((labels != LABEL_PAD_ID).any(dim=0), as_tuple=True)[0]
    count = int((labels != LABEL_PAD_ID).sum().item())
    if not count:
        raise ValueError("В батче нет обучаемых следующих токенов")
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        logits_to_keep=positions,
    )
    targets = labels.index_select(1, positions)
    loss = F.cross_entropy(
        output.logits.float().reshape(-1, output.logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=LABEL_PAD_ID,
        reduction="sum",
    )
    return loss, count
