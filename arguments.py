import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    # ======================== seed ======================== #
    parser.add_argument('--seed', type=int, default=42)

    # ======================== train, test ======================== #
    parser.add_argument('--is_train', type=bool, default=True)

    # ======================== data ======================== #
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--data_name', type=str, default='256')
    parser.add_argument('--bf16', type=bool, default=False)

    # ======================== loss ======================== #
    parser.add_argument('--r1_gamma', type=float, default=10.0)
    parser.add_argument('--pcp_ratio', type=float, default=1.0)
    parser.add_argument('--l1_ratio', type=float, default=1.0)
    parser.add_argument('--bce_ratio', type=float, default=1.0)


    # ======================== training ======================== #
    parser.add_argument('--num_train_epochs', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)

    # ======================== save ======================== #
    parser.add_argument('--save_dir', type=str, default='model')
    parser.add_argument('--load_dir', type=str, default='model')

    return parser.parse_args()