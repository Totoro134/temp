import spu
from sklearn.metrics import roc_auc_score
import csv
import numpy as np
import os

import secretflow as sf
from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal, wait
from secretflow.ml.boost.sgb_v import (
    Sgb,
    get_classic_XGB_params,
    get_classic_lightGBM_params,
)
from secretflow.ml.boost.sgb_v.model import load_model
import pprint

pp = pprint.PrettyPrinter(depth=4)

# Check the version of your SecretFlow
print('The version of SecretFlow: {}'.format(sf.__version__))

alice_ip = '127.0.0.1'
bob_ip = '127.0.0.1'
carol_ip = '127.0.0.1'
ip_party_map = {carol_ip: 'carol', bob_ip: 'bob', alice_ip: 'alice'}

_system_config = {'lineage_pinning_enabled': False}
sf.shutdown()
# init cluster
sf.init(
    ['alice', 'bob', 'carol'],
    address='local',
    _system_config=_system_config,
    object_store_memory=2 * 1024 * 1024 * 1024,
)

# SPU settings
cluster_def = {
    'nodes': [
        {'party': 'alice', 'id': 'local:0', 'address': alice_ip + ':12945'},
        {'party': 'bob', 'id': 'local:1', 'address': bob_ip + ':12946'},
        {'party': 'carol', 'id': 'local:2', 'address': carol_ip + ':12347'},
    ],
    'runtime_config': {
        # SEMI2K support 2/3 PC, ABY3 only support 3PC, CHEETAH only support 2PC.
        # pls pay attention to size of nodes above. nodes size need match to PC setting.
        'protocol': spu.spu_pb2.SEMI2K,
        'field': spu.spu_pb2.FM128,
    },
}

# HEU settings
heu_config = {
    'sk_keeper': {'party': 'alice'},
    'evaluators': [{'party': 'bob'}, {'party': 'carol'}],
    'mode': 'PHEU',
    'he_parameters': {
        # ou is a fast encryption schema that is as secure as paillier.
        'schema': 'ou',
        'key_pair': {
            'generate': {
                # bit size should be 2048 to provide sufficient security.
                'bit_size': 2048,
            },
        },
    },
    'encoding': {
        'cleartext_type': 'DT_I32',
        'encoder': "IntegerEncoder",
        'encoder_args': {"scale": 1},
    },
}

alice = sf.PYU('alice')
bob = sf.PYU('bob')
carol = sf.PYU('carol')
heu = sf.HEU(heu_config, cluster_def['runtime_config']['field'])

# 归一化数据
def norm(x):
    return (x - np.mean(x, axis=0, keepdims=True)) / np.std(x, axis=0, keepdims=True)

# 从csv文件读取数字数据
def read_csv(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        file = []
        cnt = 0
        for i in csv.reader(f):
            file.append(i)
            cnt += 1
        f.close()
    return np.array(file, dtype='float64')

current_path = os.path.dirname(__file__)
x1 = read_csv(os.path.join(current_path,'data/x1.csv'))
x1 = norm(x1)
x2 = read_csv(os.path.join(current_path,'data/x2.csv'))
x2 = norm(x2)
x3 = read_csv(os.path.join(current_path,'data/x3.csv'))
x3 = norm(x3)
y = read_csv(os.path.join(current_path,'data/y.csv'))
x_test = read_csv(os.path.join(current_path,'data/x_test.csv'))
x_test = norm(x_test)
y_test = read_csv(os.path.join(current_path,'data/y_test.csv'))

# from sklearn.datasets import load_breast_cancer

# ds = load_breast_cancer()
# x, y = ds['data'], ds['target']

v_data = FedNdarray(
    {
        alice: (alice(lambda: x1)()),
        bob: (bob(lambda: x2)()),
        carol: (carol(lambda: x3)()),
    },
    partition_way=PartitionWay.VERTICAL,
)
label_data = FedNdarray(
    {alice: (alice(lambda: y)())},
    partition_way=PartitionWay.VERTICAL,
)

params = get_classic_XGB_params()
params['num_boost_round'] = 3
params['max_depth'] = 3
# pp.pprint(params)

sgb = Sgb(heu)
model = sgb.train(params, v_data, label_data)


yhat = model.predict(v_data)
yhat = reveal(yhat)
print(f"auc: {roc_auc_score(y, yhat)}")