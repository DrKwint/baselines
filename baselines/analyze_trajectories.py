import argparse
import json
import os
import string

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition, tree
from tqdm import tqdm
import pygtrie

#matplotlib.use('Agg')  # Can change to 'Agg' for non-interactive mode

def split_by_episode(array, done):
    split_points = np.flatnonzero(done)
    return np.split(array, split_points)

def load_dir(dir):
    action = np.load(os.path.join(dir, 'action.npy'))
    state = np.load(os.path.join(dir, 'state.npy'))
    done = np.load(os.path.join(dir, 'done.npy'))
    min_len = min([action.shape[0], state.shape[0], done.shape[0]])
    return state[:min_len], action[:min_len], done[:min_len]

def build_tree(states, sensitive_window=40):
    labels = []
    for s in states:
        label = np.zeros(s.shape[0])
        for i in range(sensitive_window):
            try:
                label[-i] = 1.
            except:
                pass
        labels.append(label) 
    X = np.concatenate(states)
    Y = np.concatenate(labels)
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf.fit(X, Y)
    tree.export_graphviz(clf, out_file='tree.dot', 
                      filled=True, rounded=True)
    return clf

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'dirs', help='List of log directories', nargs='*', default=['./log'])
    parser.add_argument('-o', '--output', type=str, default='regex.txt')
    parser.add_argument('pca_fie', type=str, default='pca')
    args = parser.parse_args()
    args.dirs = [os.path.abspath(dir) for dir in args.dirs]
    print(args.dirs)

    states = []
    actions = []
    max_ep_len = 1000
    for d in tqdm(args.dirs):
        state, action, done = load_dir(d)
        states += split_by_episode(state, done)[:1000]
        actions += split_by_episode(action, done)[:1000]
        del state, action, done
    states, actions = zip(*[pair for i, pair in enumerate(zip(states, actions)) if len(pair[0]) < max_ep_len])
    del max_ep_len

    build_tree(states)

    strings = []
    for i in range(len(states)):
        idx = len(states[i]) - 1
        while states[i][idx][0] <= 1.:# and states[i][idx][6] <= 1.54:
            idx -= 1
        s = actions[i][idx:]
        if len(s) > 10:
            s = s[:10]
        strings.append(s)

    def build_translation_fn(n_components):
        pca = decomposition.PCA(n_components)
        all_strings = np.concatenate(strings)
        pca.fit_transform(all_strings)
        print(pca.explained_variance_ratio_)
        print(pca.components_)
        joblib.dump(pca, 'pca.joblib')

        def action_translation_fn(action, bucket_size=2., num_buckets=3):
            if len(action.shape) < 2:
                action = np.expand_dims(action, axis=0)
            # get everything as integers in the range [0, 2*num_buckets]
            discrete_action = np.rint(np.clip((pca.transform(action) / bucket_size) + num_buckets, 0, 2*num_buckets)).astype(np.int)
            action_id = 0
            for i in range(n_components):
                action_id += discrete_action[:,i] * (2*num_buckets+1)**i
            return list(map(lambda x: string.ascii_letters[x], action_id))

        return action_translation_fn

    translation_fn = build_translation_fn(1)
    symbol_strings = [list(translation_fn(s)) for s in strings]
    t = pygtrie.CharTrie()
    alphabet = set()
    for s in symbol_strings:
        t[''.join(s)] = True
        for c in s:
            alphabet.add(c)
    print(list(t.keys()))
    print(len(alphabet))

    #symbol_strings = '|'.join([''.join(s) for s in symbol_strings])
    with open(args.output, 'w') as regex_file:
        json.dump(sorted(list(t), key=len), regex_file)

if __name__ == "__main__":
    main()
