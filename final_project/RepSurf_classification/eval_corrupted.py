import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from dataset.ModelNetDataset import ModelNet40
from util.utils import get_model
import argparse


def get_parser():
    parser = argparse.ArgumentParser('RepSurf')
        # Basic
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--data_dir', type=str, default='./data', help='data dir')
    parser.add_argument('--log_root', type=str, default='./log', help='log root dir')
    parser.add_argument('--model', default='repsurf.scanobjectnn.repsurf_ssg_umb',
                            help='model file name [default: repsurf_ssg_umb]')
    parser.add_argument('--gpus', nargs='+', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2800, help='Training Seed')
    parser.add_argument('--cuda_ops', action='store_true', default=False,
                            help='Whether to use cuda version operations [default: False]')

        # Training
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training [default: 64]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [Adam, SGD]')
    parser.add_argument('--scheduler', type=str, default='step', help='scheduler for training')
    parser.add_argument('--epoch', default=500, type=int, help='number of epoch in training [default: 500]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--decay_step', default=20, type=int, help='number of epoch per decay [default: 20]')
    parser.add_argument('--n_workers', type=int, default=4, help='DataLoader Workers Number [default: 4]')
    parser.add_argument('--init', type=str, default=None, help='initializer for model [kaiming, xavier]')

        # Evaluation
    parser.add_argument('--min_val', type=int, default=5, help='Min val epoch [default: 100]')

        # Augmentation
    parser.add_argument('--aug_scale', action='store_true', default=False,
                            help='Whether to augment by scaling [default: False]')
    parser.add_argument('--aug_shift', action='store_true', default=False,
                            help='Whether to augment by shifting [default: False]')

        # Modeling
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--return_dist', action='store_true', default=False,
                            help='Whether to use signed distance [default: False]')
    parser.add_argument('--return_center', action='store_true', default=False,
                            help='Whether to return center in surface abstraction [default: False]')
    parser.add_argument('--return_polar', action='store_true', default=False,
                            help='Whether to return polar coordinate in surface abstraction [default: False]')
    parser.add_argument('--group_size', type=int, default=8, help='Size of umbrella group [default: 8]')
    parser.add_argument('--umb_pool', type=str, default='sum', help='pooling for umbrella repsurf [sum, mean, max]')

        # Dataset
    parser.add_argument('--dataset', type=str, default='ModelNet40', help='Datset name')

    return parser


def predict_ds(model, eval_ds, batch_size):
    model.eval()
    eval_dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model.to(device)

    preds = []

    for batch in tqdm(eval_dl):
        with torch.no_grad():
            batch_preds = model(batch[0].to(device))
            for pred in batch_preds.argmax(dim=1):
                preds.append(pred.item())


    return preds


def test_corrupt(batch_size, model, split):
    '''
    Arguments:
        args: necessary arguments like batch size and number of workers
        model: the model to be tested
        split: corruption type
    Return:
        overall_accuracy: overall accuracy (OA)
    '''
    test_ds = ModelNet40(num_points=1024, split=split, data_dir='/home/madusov/nn_project/hse_nn_spring_2024/final_project/RepSurf_classification/data/modelnet_c')

    preds = predict_ds(model, test_ds, batch_size)
    targets = [obj[1] for obj in test_ds]

    overall_accuracy = accuracy_score(targets, preds)

    return overall_accuracy


import pprint


def eval_corrupt_wrapper(model, fn_test_corrupt, args_test_corrupt):
    """
    The wrapper helps to repeat the original testing function on all corrupted test sets.
    It also helps to compute metrics.
    :param model: model
    :param fn_test_corrupt: original evaluation function, returns a dict of metrics, e.g., {'acc': 0.93}
    :param args_test_corrupt: a dict of arguments to fn_test_corrupt, e.g., {'test_loader': loader}
    :return:
    """
    corruptions = [
        'clean',
        'scale',
        'jitter',
        'rotate',
        'dropout_global',
        'dropout_local',
        'add_global',
        'add_local',
    ]
    DGCNN_OA = {
        'clean': 0.926,
        'scale': 0.906,
        'jitter': 0.684,
        'rotate': 0.785,
        'dropout_global': 0.752,
        'dropout_local': 0.793,
        'add_global': 0.705,
        'add_local': 0.725
    }
    OA_clean = None
    perf_all = {'OA': [], 'CE': [], 'RCE': []}
    for corruption_type in corruptions:
        perf_corrupt = {'OA': []}
        for level in range(5):
            if corruption_type == 'clean':
                split = "clean"
            else:
                split = corruption_type + '_' + str(level)
            test_perf = fn_test_corrupt(split=split, model=model, **args_test_corrupt)
            if not isinstance(test_perf, dict):
                test_perf = {'acc': test_perf}
            perf_corrupt['OA'].append(test_perf['acc'])
            test_perf['corruption'] = corruption_type
            if corruption_type != 'clean':
                test_perf['level'] = level
            pprint.pprint(test_perf, width=200)
            if corruption_type == 'clean':
                OA_clean = round(test_perf['acc'], 3)
                break
        for k in perf_corrupt:
            perf_corrupt[k] = sum(perf_corrupt[k]) / len(perf_corrupt[k])
            perf_corrupt[k] = round(perf_corrupt[k], 3)
        if corruption_type != 'clean':
            perf_corrupt['CE'] = (1 - perf_corrupt['OA']) / (1 - DGCNN_OA[corruption_type])
            perf_corrupt['RCE'] = (OA_clean - perf_corrupt['OA']) / (DGCNN_OA['clean'] - DGCNN_OA[corruption_type])
            for k in perf_all:
                perf_corrupt[k] = round(perf_corrupt[k], 3)
                perf_all[k].append(perf_corrupt[k])
        perf_corrupt['corruption'] = corruption_type
        perf_corrupt['level'] = 'Overall'
        pprint.pprint(perf_corrupt, width=200)
    for k in perf_all:
        perf_all[k] = sum(perf_all[k]) / len(perf_all[k])
        perf_all[k] = round(perf_all[k], 3)
    perf_all['mCE'] = perf_all.pop('CE')
    perf_all['RmCE'] = perf_all.pop('RCE')
    perf_all['mOA'] = perf_all.pop('OA')
    pprint.pprint(perf_all, width=200)


if(__name__ == '__main__'):
    parser = get_parser()

    args = parser.parse_args(['--cuda_ops',
                              '--batch_size', '220',
                              '--model', 'repsurf.repsurf_ssg_umb',
                              '--epoch', '100',
                              '--log_dir', 'repsurf_cls_ssg_umb',
                              '--gpus', '0',
                              '--n_workers', '12',
                              '--return_center',
                              '--return_dist',
                              '--return_polar',
                              '--group_size', '8',
                              '--umb_pool', 'sum',
                              '--num_point', '1024',
                              '--min_val', '5'])

    args.num_class = 40
    args.dataset = 'ScanObjectNN'
    args.normal = False

    model = torch.nn.DataParallel(get_model(args))

    checkpoint = torch.load('log/PointAnalysis/log/ModelNet40/repsurf_cls_ssg_umb/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    eval_corrupt_wrapper(model, test_corrupt, {'batch_size': 128})
