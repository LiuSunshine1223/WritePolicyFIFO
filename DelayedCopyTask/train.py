# Shared training loop for delayed-copy task variants.
import torch.nn as nn
import torch.optim as optim
from DelayedCopyTask.config import DEVICE, LR, NUM_STEPS, VOCAB_SIZE
from DelayedCopyTask.dataset import generate_batch


def train(
    model,
    window_size,
    delay_len,
    data_gen,
    batch_fn=None,
    num_steps=None,
    device=None,
):
    resolved_device = DEVICE if device is None else device
    model.to(resolved_device)

    if batch_fn is None:
        batch_fn = generate_batch
    if num_steps is None:
        num_steps = NUM_STEPS

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    acc_history = []

    for step in range(num_steps):
        # Memory is per-example state and must not leak across training steps.
        mem = getattr(model, "memory", None)
        if mem is not None:
            mem.reset()

        inp, tgt = batch_fn(delay_len, data_gen)
        inp, tgt = inp.to(resolved_device), tgt.to(resolved_device)

        logits = model(inp, window_size)
        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            tgt.reshape(-1)
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loss_history.append(loss.item())

        # Token accuracy for the current batch.
        pred = logits.argmax(dim=-1)
        acc = (pred == tgt).float().mean().item()
        acc_history.append(acc)

    return loss_history, acc_history
