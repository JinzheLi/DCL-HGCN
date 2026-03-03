from __future__ import absolute_import, division, print_function
from arg_make.args import get_args
import logging
import os
import random
import time

from torch.cuda.amp import GradScaler
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from tqdm import tqdm, trange

from train_model import our_model
from data_utils import RE_Processor

from transformers import AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from model.self_optimiza import nt_xent

from utils import is_main_process
from metrics import (
    get_f1,
    compute_micro_f1,
)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model_param(model):
    n_trainable_params, n_nontrainable_params = 0, 0
    for p in model.parameters():
        n_params = torch.prod(torch.as_tensor(p.shape))
        if p.requires_grad:
            n_trainable_params += n_params.item()
        else:
            n_nontrainable_params += n_params.item()
    logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
    logger.info('> training arguments:')
    return {
        "n_trainable_params": n_trainable_params,
        "n_nontrainable_params": n_nontrainable_params,
        "n_params": n_nontrainable_params + n_trainable_params
    }


def train(args, model, tokenizer, processor, device, n_gpu, results={}):
    results["best_checkpoint"] = 0
    results["best_acc_score"] = 0
    results["best_f1_score"] = 0
    results["best_dev_f1_score"] = 0
    results["best_mrr_score"] = 0
    results["best_checkpoint_path"] = ""

    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    print("lr: {} warm: {} total_step: {}".format(args.learning_rate, args.warmup_proportion,
                                                  num_train_optimization_steps))

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    train_data = processor.build_dataset(train_examples, tokenizer, args.max_seq_length, "train", args)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)


    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scaler = GradScaler()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    for epoch_num in trange(int(args.num_train_epochs), desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_iter = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(train_iter):
            if args.max_steps > 0 and global_step > args.max_steps:
                break

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, valid_ids, segment_ids, label_ids, e1_mask, e2_mask, dep_type_matrix, stanford_pos = batch

            loss = model(input_ids, segment_ids, input_mask, labels=label_ids, e1_mask=e1_mask, e2_mask=e2_mask,
                         valid_ids=valid_ids, dep_adj_matrix=dep_type_matrix,
                         dep_type_matrix=dep_type_matrix, stanford_pos=stanford_pos)

            # loss = z_i

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # loss.backward()

            scaler.scale(loss).backward()

            if is_main_process():
                train_iter.update(1)
                perplexity = torch.exp(torch.as_tensor(loss))
                train_iter.set_postfix_str(f"Step: {global_step} Loss: {loss:.5f} ppl: {perplexity:.5f}")

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()

        if args.local_rank == -1 or torch.distributed.get_rank() == 0 or args.world_size <= 1:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "epoch-{}".format(epoch_num))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # save_zen_model(output_dir, model, processor, tokenizer)
            PATH = os.path.join(output_dir, "model.pt")
            torch.save(model, PATH)
            # eval dev
            result = evaluate(args, model, tokenizer, processor, device, mode="dev")
            logger.info(result)
            print('=============================')
            # eval test
            result = evaluate(args, model, tokenizer, processor, device, mode="test")
            logger.info(result)

    loss = tr_loss / nb_tr_steps if args.do_train else None
    return loss, global_step


def evaluate(args, model, tokenizer, processor, device, mode="test", output_dir='./'):
    label_map = processor.labels_dict
    id2label_map = {i: label for label, i in processor.labels_dict.items()}

    if mode == "test":
        examples = processor.get_test_examples(args.data_dir)
    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    eval_data = processor.build_dataset(examples, tokenizer, args.max_seq_length, mode, args)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    nb_eval_steps, nb_eval_examples = 0, 0
    pred_scores = None
    out_label_ids = None
    eval_start_time = time.time()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)

        input_ids, input_mask, valid_ids, segment_ids, label_ids, e1_mask, e2_mask, dep_type_matrix, stanford_pos = batch

        with torch.no_grad():
            loss = model(input_ids, segment_ids, input_mask, e1_mask=e1_mask, e2_mask=e2_mask,
                           dep_adj_matrix=dep_type_matrix, dep_type_matrix=dep_type_matrix,
                           valid_ids=valid_ids, stanford_pos=stanford_pos)

        nb_eval_steps += 1
        if pred_scores is None:
            pred_scores = loss.detach().cpu().numpy()
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            pred_scores = np.append(pred_scores, loss.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

    preds = np.argmax(pred_scores, axis=1)

    eval_run_time = time.time() - eval_start_time

    if args.task_name == 'tacre':
        _, _, max_f1 = get_f1(out_label_ids, preds)
        result = {
            "f1": max_f1 * 100,
        }
    else:
        result = {
            "f1": compute_micro_f1(preds, out_label_ids, label_map, ignore_label='Other', output_dir=output_dir)
        }

    result["eval_run_time"] = eval_run_time
    result["inference_time"] = eval_run_time / len(examples)

    logging.info(result)

    output_dir = os.path.join(args.output_dir, "text.txt")
    with open(output_dir, "a") as file:
        file.write(mode+" | "+str(result)+"\n")

    return result


def train_func(args):
    args.device = device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.output_dir = os.path.join(args.output_dir, args.model_name)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        print("WARNING: Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and is_main_process():
        os.makedirs(args.output_dir)

    processor = RE_Processor(dep_type=args.dep_type)

    processor.prepare_type_dict(args.data_dir)
    processor.prepare_labels_dict(args.data_dir)
    processor.prepare_pos_dict(args.data_dir)

    label_list = processor.labels_dict.keys()
    dep_type_list = processor.types_dict.keys()
    pos_dict = processor.pos_dict.keys()

    num_labels = len(label_list)
    type_num = len(dep_type_list)
    num_pos = len(pos_dict)

    # if args.vocab_file is None:
    #     args.vocab_file = os.path.join(args.model_path, VOCAB_NAME)
    print("LOAD tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        max_len=args.max_seq_length
    )
    # tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=args.max_seq_length)
    tokenizer.add_tokens(["<e1>", "</e1>", "<e2>", "</e2>"])
    print("LOAD CHECKPOINT from")
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
    )
    # config = BertConfig.from_json_file(os.path.join(args.model_path, "config.json"))
    config.__dict__["model_name_or_path"] = args.model_name_or_path
    config.__dict__["hgcn_num_layers"] = args.hgcn_num_layers
    config.__dict__["gcn_num_layers"] = args.gcn_num_layers
    config.__dict__["max_seq_length"] = args.max_seq_length
    config.__dict__["dropout_prob"] = args.dropout_prob
    config.__dict__["dropout"] = args.dropout
    config.__dict__["bias"] = args.bias
    config.__dict__["drop_BiLSTM"] = args.BiLSTM_dropout
    config.__dict__["num_labels"] = num_labels
    config.__dict__["type_num"] = type_num
    config.__dict__["num_pos"] = num_pos
    config.__dict__["dep_type"] = args.dep_type
    config.__dict__["cuda"] = args.device
    config.__dict__["device"] = args.device_id
    config.__dict__["manifold"] = args.manifold
    config.__dict__["c"] = args.c
    config.__dict__["p"] = args.p
    config.__dict__["act"] = args.act
    config.__dict__["train_batch_size"] = args.train_batch_size

    if args.train_again:
        model = our_model.from_pretrained(args.train_path,config)
    else:
        model = our_model(config)

    model.to(device)

    model.encoder.resize_token_embeddings(len(tokenizer))

    train(args, model, tokenizer, processor, device, args.n_gpu)


def predict_func(args):
    args.device = device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    processor = RE_Processor(dep_type=args.dep_type)

    processor.prepare_type_dict(args.data_dir)
    processor.prepare_labels_dict(args.data_dir)
    processor.prepare_pos_dict(args.data_dir)

    print("LOAD tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        max_len=args.max_seq_length
    )
    # tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=args.max_seq_length)
    tokenizer.add_tokens(["<e1>", "</e1>", "<e2>", "</e2>"])
    print("LOAD CHECKPOINT from")

    model = torch.load(args.model_path)
    model.to(device)

    result = evaluate(args, model, tokenizer, processor, device, mode="test")
    logger.info(result)


# def predict_func(args):
#     args.device = device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
#     args.n_gpu = torch.cuda.device_count()
#
#     if args.vocab_file is None:
#         args.vocab_file = os.path.join(args.model_path, VOCAB_NAME)
#     tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=args.max_seq_length)
#     tokenizer.add_never_split_tokens(["<e1>", "</e1>", "<e2>", "</e2>"])
#     config = BertConfig.from_json_file(os.path.join(args.model_path, "config.json"))
#     model = our_model.from_pretrained(args.model_path, args=args, config=config)
#     dict_bin = torch.load(os.path.join(args.model_path, "dict.bin"))
#     processor = RE_Processor(dep_type=config.dep_type, types_dict=dict_bin["types_dict"],
#                              labels_dict=dict_bin["labels_dict"])
#     model.to(device)
#     result = evaluate(args, model, tokenizer, processor, device, mode="test")
#     logger.info(result)
#     return result["f1"]


def main():
    args = get_args()

    torch.cuda.set_device(args.device_id)

    if args.do_train:
        train_func(args)
    # elif args.do_test:
    #     test_func(args)
    elif args.do_test:
        predict_func(args)


if __name__ == "__main__":
    main()
