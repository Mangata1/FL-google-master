import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10,help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=10,help="the number of local epochs: E")
    parser.add_argument('--dataset', type=str, default='mnist', help="name \of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \of classes")
    parser.add_argument('--iid', type=int, default=1,help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--local_bs', type=int, default=10,help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,help='learning rate')
    parser.add_argument('--model', type=str, default='mlp', help='model name')

    
    args = parser.parse_args()
    return args
