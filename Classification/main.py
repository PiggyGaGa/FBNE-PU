import argparse
import pandas as pd


from MyModel import my_model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--nnPU",
                        action='store_true',
                        help="Whether to us non-negative pu-learning risk estimator.")
    parser.add_argument("--train_batch_size",
                        default=24,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=100,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=1,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()


    data_df = pd.read_pickle('./data/data.pickle')
    true_label_df = pd.read_pickle('./data/label.pickle')

    my_model(data_df, true_label_df, args)



if __name__ == "__main__":
    main()

