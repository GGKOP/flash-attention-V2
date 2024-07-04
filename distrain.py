import math
import time

from torch import nn, optim
from torch.optim import Adam
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time



def setup(rank, world_size):
    dist.init_process_group('nccl',rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

model = Transformer(
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    trg_sos_idx=trg_sos_idx,
    d_model=d_model,
    enc_voc_size=enc_voc_size,
    dec_voc_size=dec_voc_size,
    max_len=max_len,
    ffn_hidden=ffn_hidden,
    n_head=n_heads,
    n_layers=n_layers,
    drop_prob=drop_prob,
    device=device,
)

print(f"The model has {count_parameters(model):,} trainable parameters")
model.apply(initialize_weights)
optimizer = Adam(
    params=model.parameters(), lr=init_lr, weight_decay=weight_decay, eps=adam_eps
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, verbose=True, factor=factor, patience=patience
)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

def train(rank ,world_size, model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src.to(rank)
        trg = batch.trg.to(rank)

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print("step :", round((i / len(iterator)) * 100, 2), "% , loss :", loss.item())

    return epoch_loss / len(iterator)


def run(rank, world_size, model, dataset, optimizer, criterion, clip):
    setup(rank, world_size)
    device_id =rank %torch.cuda.device_count()
    model = model.to(device_id)
    ddp_model = DistributedDataParallel(model, device_ids=[device_id])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    iterator = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    train_loss = train(rank, world_size, ddp_model, iterator, optimizer, criterion, clip)
    print(f"Rank {rank} training finished with loss: {train_loss}")
    cleanup()


if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size, model, train_, optimizer, criterion, clip), nprocs=world_size, join=True)
    