import argparse


def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch', default=4096, type=int, help='batch size')
    parser.add_argument('--tstBat', default=100000, type=int, help='number of interactions in a testing batch')
    parser.add_argument('--reg', default=1e-7, type=float, help='weight decay regularizer')
    parser.add_argument('--epoch', default=120, type=int, help='number of epochs')
    parser.add_argument('--latdim', default=128, type=int, help='embedding size')
    parser.add_argument('--hyperNum', default=128, type=int, help='number of hyperedges')
    parser.add_argument('--gnn_layer', default=3, type=int, help='number of gnn layers')
    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--keepRate', default=0.75, type=float, help='ratio of edges to keep')
    parser.add_argument('--temp', default=0.1, type=float, help='temperature')
    parser.add_argument('--mult', default=1e-1, type=float, help='multiplication factor')
    parser.add_argument('--ssl_reg', default=1e-2, type=float, help='weight for ssl loss')
    parser.add_argument('--data', default='drugbank', type=str, help='name of dataset')
    parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
    parser.add_argument('--gpu', default='2', type=int, help='indicates which gpu to use')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed')
    parser.add_argument('--data_seed', default=42, type=int,
                        help='seed')
    parser.add_argument('--iteration', type=int, default='1', help='iteration')
    parser.add_argument('--is_debug', type=bool, default=False, help='is_debug')
    parser.add_argument('--dense', action='store_true', default=False, help='dense')
    parser.add_argument('--validate', action='store_true', default=False,
                        help='if set, use validation mode which splits all relations into \
	                        train/val/test and evaluate on val only;\
	                        otherwise, use testing mode which splits all relations into train/test')
    parser.add_argument('--gnn_type', default='gcn', type=str,
                        help='GNN架构类型: gcn, gat, graphsage, attention, residual, highorder')
    parser.add_argument('--use_labels', action='store_true', default=True, help='使用标签信息进行有监督对比学习')
    parser.add_argument('--dcl', action='store_true',
                        help='use Decoupled Contrastive Loss')
    parser.add_argument('--hn_k', default=0, type=int,
                        help='top‑K hardest negatives (0 = all)')
    parser.add_argument('--c_mode', default='multi', choices=['sym', 'barlow', 'multi'],
                        help="contrastive mode: sym | barlow | multi‑positive")
    parser.add_argument('--barlow_lambd', default=5e-3, type=float,
                        help="λ for off‑diag penalty in Barlow Twins")
    return parser.parse_args()


args = ParseArgs()
