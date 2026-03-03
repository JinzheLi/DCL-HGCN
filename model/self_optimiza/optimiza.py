from transformers.optimization import AdamW
import torch

def load_optimizer(args, model):

    scheduler = None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # TODO: LARS

    return optimizer, scheduler