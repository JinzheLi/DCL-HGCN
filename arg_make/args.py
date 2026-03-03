import argparse


def get_args():
    parser = argparse.ArgumentParser()

    ## Required parameters

    parser.add_argument("--device_id",
                        default=1,
                        type=int,
                        help="model use device id.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--head_num",
                        default=8,
                        type=int,
                        help="attention head num")

    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0, help='dropout probability')
    parser.add_argument("--bias", type=int, default=1, help='whether to use bias (1) or not (0)')

    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--model_name_or_path",
                        default="bert-base-uncased",
                        type=str)

    # parser.add_argument("--model_name_or_path",
    #                     default="bert-large-uncased",
    #                     type=str)

    parser.add_argument('--BiLSTM_dropout',
                        type=float,
                        default=0.1,
                        help='BiLSTM dropout rate.')
    parser.add_argument("--train_again",
                        action='store_true',
                        help="train_again.")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default="./",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--model_path",
                        default=None,
                        type=str,
                        required=False,
                        help="Model path")
    parser.add_argument("--train_path",
                        default=None,
                        type=str,
                        help="Model path")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        help="Model name")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to predict.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=2,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        # default=5e-6,
                        default=7e-6,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=4.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=-1.0,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--vocab_file',
                        type=str, default=None,
                        help="Vocabulary mapping/file BERT was pretrainined on")
    parser.add_argument("--rank",
                        type=int,
                        default=0,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--world_size",
                        type=int,
                        default=1,
                        help="world size")

    parser.add_argument('--dep_type', type=str, default='full_graph',
                        choices=["full_graph", "local_graph", "global_graph", "local_global_graph"])

    parser.add_argument('--hgcn_num_layers', type=int, default=2)
    parser.add_argument('--gcn_num_layers', type=int, default=2)

    parser.add_argument('--manifold', type=str, default='Euclidean',
                        help="which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]")
    parser.add_argument('--act', type=str, default='relu',
                        help="which activation function to use (or None for no activation)")

    parser.add_argument('--c', type=float, default=1.0, help='hyperbolic radius, set to None for trainable curvature')
    parser.add_argument('--p', type=float, default=0.5)
    args = parser.parse_args()

    args.task_name = args.task_name.lower()


    return args