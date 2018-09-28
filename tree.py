"""
Tree-Clustering Algorithm
0. Scale and Quantize data
1. Determine val(maximum_value), L(number of levels)
2. Insert all data points into the tree and mark those leaves as black
3. Traverse the tree up from the leaves to root, marking parent node black if all children black, white if all white,
    grey if some white some black. Optional, done here: Cut the subtree for the parent if all its children black
4. Determine connected components. This is done implicitly while coloring. Note that all white nodes are leaves.
    Similarly because of the trimming, all black nodes are also leaves. All parents are grey in color. Hence each of the
    black and white nodes returned by color_tree are the largest black and white connected components respectively.
TODO:
5. Subgraphs/ hierarchical clusters with increasing threshold distances
- PUSH TO GITHUB
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import logging
import queue

def child_location(parent, child):
    """
    :param parent: Parent center, list of d numbers
    :param child: Data point value, list of d numbers
    :return: Number of the child hypercube where the point will lie
    """
    compare = ['1' if c > p else '0' for c, p in zip(child, parent)]
    binary_num = "".join(compare)
    child_num = int(binary_num, 2)
    return child_num

def child_center(child_num, parent_val, parent_size):
    """
    :param child_num:  Child number varying from 0 to branch_factor-1
    :param parent_val: Center of the parent hypercube. It is a list of d integers
    :return: Center of the child_num child's hypercube. It is a list of d integers
    """
    d = len(parent_val)
    bin_len = '{0:0'+str(d)+'b}'
    '''Get the binary representation of child_num upto d bits'''
    binary_str = bin_len.format(child_num)
    binary_list = [int(s) for s in binary_str]
    '''Convert the 0,1 representation into -1,1 representation'''
    binary_list = [2*b-1 for b in binary_list]
    child_val = []
    for i in range(len(binary_list)):
        temp = parent_val[i] + (binary_list[i]*parent_size/4)
        '''Appended in the same order'''
        child_val.append(temp)
    return child_val

def build_tree(data, L):
    """
    :param data: Entire dataset of size num_of_points x d
    :param L: Levels in the tree, builds tree of depth L i.e. from l=0 to l=L
    :return: Root of tree
    """
    d = len(data[0])
    extent = 2**L*np.ones(d)
    center = 0.5*extent
    root = node(center)
    for pt in data:
        l = 0
        temp = root
        logger.debug('Point {}'.format(pt))
        logger.debug('Reached {}'.format(temp.val))
        while l < L:
            if temp.children == None:
                parent_size = 2**(L-l)
                temp.create_children(parent_size)
            child_num = child_location(temp.val, pt)
            temp = temp.children[child_num]
            l += 1
            logger.debug('Reached {}'.format(temp.val))
        diff = [i for i, j in zip(temp.val, pt) if i+0.5 != j]
        if len(diff) != 0:
            logger.info('Error reaching the leaf node for {}. Reached {}'.format(pt, temp.val))
        temp.color = 1
    return root

def color_tree(root):
    """
    Color all the nodes of the tree white, black or grey depending on the color of children. Using stack with
    Breadth First Seach
    """
    parents = []
    parents.append({})
    parents.append({})
    parents.append({})
    q = queue.LifoQueue(maxsize=0)
    root.aux = []
    root.aux.append(root.val)
    q.put(root)
    trim = 0
    while q.empty() == False:
        temp = q.get()
        q.put(temp)
        if temp.children != None and temp.index < len(temp.children):
            '''This node has children and not all of them have been traversed'''
            child = temp.children[temp.index]
            child.aux = temp.aux.copy()
            child.aux.append(child.val)
            temp.index += 1
            q.put(child)
        else:
            '''
            This node will be removed, as either it is a child or all it's children have been traversed. Note that white
            nodes will be at the leaves only because a node was made a parent only if atleast one of its child was black.
            Hence, the parent ought to be grey (or black if tree is not trimmed.)
            '''
            if temp.parent != None and temp.parent.index == 1:
                '''This node was the first child of their parent'''
                temp.parent.color = temp.color
            elif temp.parent != None :
                '''Not the first child'''
                if (temp.color == 2) or (temp.color == 1 and temp.parent.color == 0) or (temp.color == 0 and temp.parent.color == 1):
                    temp.parent.color = 2
            if temp.color == 1 and temp.children != None:
                '''
                Trim the tree and remove all children for Black parents. If do not want to trim, 
                remove these if-do statements
                '''
                logger.debug('Trimming {}th black parent!'.format(trim))
                trim +=1
                for child in temp.children:
                    parents[1].pop(tuple(child.val), "None")
                    del child

            parents[temp.color][tuple(temp.val)] = len(temp.aux)
            temp.aux = None
            temp.index = 0
            q.get()
    return parents

class node:
    def __init__(self, val):
        self.val = val
        self.children = None# TODO: If possible change this to a numpy array of fixed length, as it's length is fixed
        self.parent = None
        self.color = 0
        self.aux = None
        self.index = 0

    def create_children(self, parent_size):
        self.children = []
        for i in range(branch_factor):
            child_val = child_center(i, self.val, parent_size)
            temp_node = node(child_val)
            temp_node.parent = self
            self.children.append(temp_node)
        return None

    def print_tree(self):
        """
        Traverse the tree using a stack for Breadth First Search
        """
        black_leaves = []
        q = queue.LifoQueue(maxsize=0)
        self.aux = [self.val]
        q.put(self)
        while q.empty() == False :
            temp = q.get()
            q.put(temp)
            if temp.children == None :
                if temp.color==1:
                    logger.debug('Pt {} path {}'.format(temp.val, temp.aux))
                    black_leaves.append(temp.val)
                temp.aux = None
                t = q.get()
            else:
                if temp.index < len(temp.children):
                    child = temp.children[temp.index]
                    child.aux = temp.aux.copy()
                    child.aux.append(child.val)
                    temp.index += 1
                    q.put(child)
                else:
                    temp.aux = None
                    temp.index = 0
                    t = q.get()
        return black_leaves




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    dataset = '/Users/lavisha/PycharmProjects/Project1/data_aggregation.txt'
    d = 2
    branch_factor = 2**d
    data_handle = open(dataset, "r")
    data = data_handle.read().split('\n')[:-1]
    '''Ignore the 3rd component which represents cluster number.'''
    data = [eg.split()[:d] for eg in data]
    data = [[float(feature) for feature in example]for example in data]
    '''Scale and Quantize the data '''
    data = [[round(f) for f in example] for example in data]
    # data = [[round(f * 10) for f in example] for example in data]
    # data = [[2,2],[4,5],[7,8],[1,6],[7,3],[5,5],[8,5],[3,3],[4,6],[7,7]]
    val = 0
    for example in data:
        if max(example)>val:
            val = max(example)
    L = math.ceil(math.log2(val))
    logger.info('n = %d d = %d val = %d L = %d',len(data), len(data[0]), val, L )

    head = build_tree(data, L)
    black_leaves = np.array(head.print_tree())
    plt.plot(black_leaves[:,0], black_leaves[:,1], 'r.')
    [white, black, grey] = color_tree(head)
    '''
    Above returned black nodes also correspond to black connected components B1, B2, ... Similarly white nodes correspond
    to white connected components W1, W2, ...
    '''
    print('black', black)
    print('white', white)
    print('grey', grey)
    white = np.array([key for key in white.keys()])
    black = np.array([key for key in black.keys()])
    grey = np.array([key for key in grey.keys()])
    plt.figure(2)
    plt.plot(grey[:,0], grey[:,1], 'g.')
    # plt.plot(black[:,0], black[:,1], 'k.', white[:,0], white[:,1], 'y.', grey[:,0], grey[:,1], 'g.')
    plt.show()
    print(len(black))
    # black_parents = np.array(black_parents)
    # grey_parents = np.array(grey_parents)
    # white_parents = np.array(white_parents)
    # plt.plot(black_parents[:, 0], black_parents[:, 1], 'r.', black_leaves[:,0], black_leaves[:,1], 'b.')
    # # plt.plot(black_parents[:,0], black_parents[:,1], 'b.', white_parents[:,0], white_parents[:,1], 'y.', grey_parents[:,0], grey_parents[:,1], 'g.')
    # plt.show()