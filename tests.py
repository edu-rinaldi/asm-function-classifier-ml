"""
    Author: Eduardo Rinaldi
    Date: 10/11/2020
    Copyright ©2020
"""
import sklearn as sk
import numpy as np
import ast
import json
import pprint
import queue

data = []

with open('dataset.json', 'r') as f:
    for e in f:
        v = json.loads(e)
        data.append([v['id'], ast.literal_eval(v['lista_asm']), v['semantic']])
        cfg = v['cfg']
        # d = np.append(d, [v['id'], v['semantic'], v['lista_asm']])

data = np.array(data)


def findRoot(cfg):
    toVisit = set()
    for adj in cfg['adjacency']:
        for n in adj:
            toVisit.add(n['id'])
    for i in range(len(cfg['nodes'])):
        if cfg['nodes'][i]['id'] not in toVisit:
            return i
    return 0

indexById = {cfg['nodes'][i]['id']:i for i in range(len(cfg['nodes']))}

def f(cfg, node):
    count = {x:0 for x in range(len(cfg['nodes']))}
    q = queue.Queue()
    q.put(node)
    while not q.empty():
        cur = q.get()
        count[cur] += 1
        # print(cur)
        for child in cfg['adjacency'][cur]:
            child = indexById[child['id']]
            q.put(child)
    return count


def tarjan(cfg):
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
                print(w, v)
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



startNode = findRoot(cfg)
print(tarjan(cfg))

