"""
    Author: Eduardo Rinaldi
    Date: 10/11/2020
    Copyright ©2020
"""

import numpy as np
import json
import ast

classByVal = {0: 'encryption', 1 : 'sort', 2 : 'math', 3 : 'string'}
valByClass = {'encryption' : 0, 'sort' : 1, 'math' : 2, 'string': 3}

def find_root(cfg):
    """
    Find a root node for the given cfg
    """
    toVisit = set()
    for adj in cfg['adjacency']:
        for n in adj:
            toVisit.add(n['id'])
    for i in range(len(cfg['nodes'])):
        if cfg['nodes'][i]['id'] not in toVisit:
            return i
    return 0

def indexByIdMap(cfg):
    """
    Function that returns the dictionary that returns the index of a certain node given the ID.
    """
    return {cfg['nodes'][i]['id']:i for i in range(len(cfg['nodes']))}


def tarjan(cfg):
    """
    Python implementation of tarjan algorithm based to be used on the cfg.
    """
    indexById = indexByIdMap(cfg)
    index = 0
    indexes = [-1 for _ in cfg['nodes']]
    lowlinks = [-1 for _ in cfg['nodes']]
    onStack = [False for _ in cfg['nodes']]
    components = []
    s = []

    def strongconnect(v):
        nonlocal index
        indexes[v] = index
        lowlinks[v] = index
        index+=1
        s.append(v)
        onStack[v] = True
        for w in cfg['adjacency'][v]:
            w = indexById[w['id']]
            if indexes[w] == -1:
                strongconnect(w)
            elif onStack[w]:
                lowlinks[v] = min(lowlinks[v], indexes[w])
        
        if lowlinks[v] == indexes[v]:
            component = []
            while True:
                w = s.pop()
                onStack[w] = False
                component += [w]
                if w == v:
                    break
            components.append(component)

    for v in cfg['nodes']:
        v = indexById[v['id']]
        if indexes[v] == -1:
            strongconnect(v)

    return components

def get_features(list_asm, cfg, normalization_func=lambda x: x):
    """
        :params list_asm: list of asm instructions
        :returns a list such that [#bitwiseOPs, #movOPs, #xmmm*, #arithmeticOPs, #cmpOPs, #swapOPs, #loops, #maxComplexity]
    """

    
    bitwise = {'xor', 'shift', 'pxor', 'xorps', 'sal', 'shl', 'sar', 'not', 'and', 'pand', 'or', 'por'}
    mov = 'mov'
    xmm = 'xmm'
    arithmetic = {'add', 'mul', 'neg', 'sub', 'imul', 'idivl', 'divl', 'idivq', 'divq', 'idiv', 'divsd', 'div'}
    comp = 'cmp'
    swap = 'bswap'
    
    # initialize final list to return
    final = [0,len(list_asm),0,0,0,0,0]
    for instruction in list_asm:
        instruction_splitted = instruction.split(" ")

        # get instruction type
        if instruction_splitted[0] in bitwise:
            final[0] += 1

        if instruction.startswith(mov):
            final[2] += 1

        if xmm in instruction:
            final[3] += 1
        
        if instruction_splitted[0] in arithmetic:
            final[4] += 1
        
        if instruction.startswith(comp):
            final[5] += 1

        if instruction.startswith(swap):
            final[6] += 1
    
    # get list of components in the graph
    components = tarjan(cfg)
    loops = 0
    maxComplexity = 0
    for component in components:
        lenComp = len(component)
        # if there're at least 2 nodes in a component --> there's a loop 
        if lenComp >= 2:
            loops += 1
            # n nodes in a component --> n-1 nested loops
            maxComplexity = max(maxComplexity, lenComp-1)

    final.extend([loops, maxComplexity])
    
    return list(map(normalization_func,final))


def parseDataset(dataset_path, training_data=True):
    # reading jsonl and preparing dataset
    data = []
    with open(dataset_path, 'r') as f:
        for e in f:
            v = json.loads(e)
            row = get_features(ast.literal_eval(v['lista_asm']), v['cfg'])
            if training_data:
                row.append(valByClass[v['semantic']])
            data += [row]

    # convert to numpy array
    data = np.array(data)
    if training_data:
        # extract X and y from data
        X_all = np.log2(data[:,:-1]+1)
        y_all = data[:, -1]
        return X_all, y_all
    else:
        return np.log2(data+1)
