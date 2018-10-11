"""
Author: Lavisha Aggarwal
University of Illinois at Urbana-Champaign

Tree-Clustering Algorithm
0. Scale and Quantize data
1. Determine val(maximum_value), L(number of levels)
2. Insert all data points into the tree and mark those leaves as black
3. Traverse the tree up from the leaves to root, marking parent node black if all children black, white if all white,
    grey if some white some black. Optional, done here: Cut the subtree for the parent if all its children black
4. Determine connected components. This is done implicitly while coloring. Note that all white nodes are leaves.
    Similarly because of the trimming, all black nodes are also leaves. All parents are grey in color. Hence each of the
    black and white nodes returned by color_tree are the largest black and white connected components respectively.
5. Analyse the black and white connected components by plotting hypercube centers in their respective sizes (depth).
   Also plot histogram of black and white leaves (connected component) sizes
6. Determine subgraphs. Merge subgraphs based on the repetitions dictionary.
7. Visualise black and white subgraphs and evaluate the correctness of subgraph formation. Interestingly we get almostas many
    black subgraphs as there are clusters in there are clusters

TODO:
11. Hierarchical clusters/Erosion/Dilation with increasing threshold distances



- Basically these subgraphs are the lowest level i.e. most finest clusters, just next to individual points
- If we keep increasing the radius, we will get more coarser clusters
- How does erosion and dilation come into the picture
- I didn't understand why we want to get the largest white node / all nC2 boundaries between all pairs of black nodes
- Have to clarify this distinctions -- that by black connected components we mean that connected in the graph or if there exists a path of all black nodes between
2 nodes then they will be part of a connected component. As opposed to black subgraph where there may not be a path of all black nodes between 2 black points but they
share an edge/face for 3D to be part of a subgraph
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import logging
import queue

def child_location(parent, point):
    """
    Compares the value of parent and point to determine in which child will the point lie
    :param parent: Parent center, list of d numbers
    :param point: Data point value, list of d numbers
    :return: Number of the child hypercube where the point will lie
    """
    compare = ['1' if c > p else '0' for c, p in zip(point, parent)]
    binary_num = "".join(compare)
    child_num = int(binary_num, 2)
    return child_num

def child_center(child_num, parent_val, parent_size):
    """
    Converts child_num into binary and based on the binary representation finds the child's center
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
    Starting from the coarsest level, build the tree. Each point is added at a time to the lowest level. We determine
    in which of the children hypercubes the point will lie and that child hypercube is further split into children until
    the desired leaf at level = L is reached. There the point is placed as a black node.
    :param data: Entire dataset of size num_of_points x d
    :param L: Levels in the tree, builds tree of depth L i.e. from l=0 to l=L
    :return: Root of tree
    """
    d = len(data[0])
    extent = 2**L#*np.ones(d)
    center = 0.5*extent
    root_val = [center]*d
    root = node(root_val, 0)
    for pt in data:
        l = 0
        temp = root
        logger.debug('Point {}'.format(pt))
        logger.debug('Reached {}'.format(temp.val))
        while l < L:
            if temp.children == None:
                parent_size = 2**(L-l)
                temp.create_children(parent_size, l+1)
            child_num = child_location(temp.val, pt)
            temp = temp.children[child_num]
            l += 1
            logger.debug('Reached {} level {}'.format(temp.val, temp.level))
        diff = [i for i, j in zip(temp.val, pt) if i+0.5 != j]
        if len(diff) != 0:
            logger.info('Error reaching the leaf node for {}. Reached {}'.format(pt, temp.val))
        temp.color = 1
    return root

def color_tree(root):
    """
    Color all the nodes of the tree white, black or grey depending on the color of children. Using stack with
    Breadth First Search
    :param root: Root node of the tree
    :return: A List of dictionaries of White(0), Black(1), and Grey(2) nodes with vales as their (level+1)
    [{White nodes}, {Black nodes}, {Grey nodes}]
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
            """
            This node will be removed, as either it is a child or all it's children have been traversed. Note that white
            nodes will be at the leaves only because a node was made a parent only if atleast one of its child was black
            Hence, all parents must be grey (or black if tree is not trimmed).
            Below temp!=None condition is put only to handle for when temp=root.
            """
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
                temp.children = None

            parents[temp.color][tuple(temp.val)] = len(temp.aux)
            temp.aux = None
            temp.index = 0
            q.get()
    return parents

def update_dict(dictionary, val2, val1):

    if val1 in dictionary:
        if val2 not in dictionary[val1]:
            dictionary[val1].append(val2)
    else:
        dictionary[val1] = [val2]
    return dictionary


def find_subgraphs(root, L):
    """
    Run a DFS over all nodes. When reach a leaf node which is not already part of a subgraph, build it's subgraph using
    BFS starting from that leaf. Only leaf nodes can start a subgraph hence, subgraphs are either black or white; since
    all leaves are either black or white and all parent nodes are grey nodes. A special case can arrive that for node A,
    its neighboring node B is already part of another subgraph Gj i.e. When the neighbors of node B were visited, node A
    was not reached. This can happen when B is bigger than A, so the immediate neighbor of B is a grey node. Hence, while
    expanding from B, we never reached A. To handle such cases, a dictionary of repetitions is returned which contains
    indexes of subgraphs which need to be merged.
    :param root: Root of the tree
    :param L: Number of levels in tree, the tree goes on from l = 0 till l = L
    :return: Number of subgraphs and whether there are any repetitions, i.e. different subgraph numbers but representing
    the same subgraph and hence which need to be merged.
    """
    stack = queue.LifoQueue(maxsize=0)
    root.aux = [root.val]
    stack.put(root)
    subgraph_num = -1
    repetitions = {}
    subgraph_list = []
    subgraph_color = []
    while stack.empty() == False:
        temp = stack.get()
        print('Stack - ', temp.val)
        stack.put(temp)
        if temp.children == None:
            """Reached a leaf node. Start forming a sub-graph"""
            if temp.subgraph!=-1:
                """Check that this node is part of some other subgraph"""
                temp.aux = None
                stack.get()
            else:
                """Start a BFS for a subgraph formation for this node"""
                q = queue.Queue(maxsize=0)
                subgraph_num +=1
                subgraph_list.append({})
                temp.subgraph = subgraph_num
                subgraph_color.append(temp.color)
                subgraph_list[subgraph_num][tuple(temp.val)] = temp.level
                print('list', subgraph_list)
                print('color', subgraph_color)
                print('subgraph_num', subgraph_num)
                q.put(temp)
                while q.empty() == False:
                    ptr = q.get()
                    if ptr.color == 1:
                        print('B Queue', ptr.val)
                    else:
                        print('W Queue', ptr.val)
                    all_nbrs = ptr.find_neighbors(L, root)
                    for nbr in all_nbrs:
                        if nbr.subgraph == ptr.subgraph:
                            continue
                        elif nbr.color == ptr.color and nbr.subgraph != -1:
                            repetitions = update_dict(repetitions, nbr.subgraph, subgraph_num)
                            print("Added into ", nbr.val, "'s subgraph ", nbr.subgraph, " my  subgraph num", subgraph_num)
                        elif nbr.color == ptr.color:
                            nbr.subgraph = subgraph_num
                            subgraph_list[subgraph_num][tuple(nbr.val)] = nbr.level
                            q.put(nbr)
                """Hopefully the subgraph for that specific node has been created"""
                temp.aux = None
                stack.get()
        else:
            if temp.index < len(temp.children):
                """This parent node has more children"""
                while(temp.index < len(temp.children)) and temp.children[temp.index].subgraph!=-1:
                    """Find the child which has not already been subgraphed"""
                    temp.index +=1
                if temp.index < len(temp.children):
                    """Found a non subgraphed child"""
                    child = temp.children[temp.index]
                    child.aux = temp.aux.copy()
                    child.aux.append(child.val)
                    stack.put(child)
                    temp.index += 1
                else:
                    """All children already subgraphed!"""
                    temp.aux = None
                    temp.index = 0
                    stack.get()
            else:
                """No more children of this parent node"""
                temp.aux = None
                temp.index = 0
                stack.get()

    return [subgraph_num+1, repetitions, subgraph_list, subgraph_color]

class node:
    def __init__(self, val, level):
        if level >L:
            print("Received grwater than L=",L, level)
        self.val = val
        self.children = None# TODO: If possible change this to a numpy array of fixed length, as it's length is fixed
        self.parent = None
        self.color = 0
        self.level = level
        self.subgraph = -1
        self.aux = None
        self.index = 0
        """aux and index used during traversal and coloring. aux saves path from root to the node. index stores the number of children 
        visited. Both aux and index are set back to None and 0 by the end of traversal"""

    def create_children(self, parent_size, level):
        """
        Creates 2**d i.e. branch_factor number of children hypercubes and updates the parent node's children attribute
        :param parent_size: Size of parent hypercube
        :param level: Level of children hypercubes
        :return: None
        """
        self.children = []
        for i in range(branch_factor):
            child_val = child_center(i, self.val, parent_size)
            temp_node = node(child_val, level)
            temp_node.parent = self
            self.children.append(temp_node)
        return None

    def traverse_tree(self):
        """
        Traverse the tree using a stack for Breadth First Search
        :return: Black leaves
        """
        black_leaves = []
        q = queue.LifoQueue(maxsize=0)
        self.aux = [self.val]
        q.put(self)
        while q.empty() == False:
            temp = q.get()
            q.put(temp)
            if temp.children == None:
                """Reached leaf node"""
                if temp.color == 1:
                    logger.debug('Pt {} path {} level{}'.format(temp.val, temp.aux, temp.level))
                    black_leaves.append(temp.val)
                temp.aux = None
                q.get()
            else:
                if temp.index < len(temp.children):
                    """Un-traversed children still there"""
                    child = temp.children[temp.index]
                    child.aux = temp.aux.copy()
                    child.aux.append(child.val)
                    temp.index += 1
                    q.put(child)
                else:
                    """All children traversed"""
                    temp.aux = None
                    temp.index = 0
                    q.get()
        return black_leaves

    def find_node(self, val):
        """
        Given a point value i.e. d dimensional vector, determines the node having that value -- either an exact match or
        the most closest leaf node
        :param val: Value to be searched in the tree
        :return: Node with the most similar value. Either an exact match or the leaf hypercube containing that value
        """
        temp = self
        while temp.val!=val and temp.children != None:
            child_num = child_location(temp.val, val)
            temp = temp.children[child_num]
        return temp

    def find_neighbors(self, L, root):
        """
        Finds the 2*d number of neighbors by determining binary representation of 0,1,..,2d-1. Using the binary representation,
        finding neighbor centers and then finding nodes having that value. While computing neighbor centers, (self) hypercube's
        edge length is added/subtracted i.e. neighbor of the same size are considered.
        :param L: Total number of levels
        :return: All the 2**d neighboring nodes
        """
        l = self.level
        d = len(self.val)
        side_length = 2 ** (L - l)
        neighbors_list = []
        print("For ", self.val, ' neighbors are:')
        for neighbor_num in range(d):
            binary_list = [0]*d
            binary_list[neighbor_num] = 1
            '''Neighbors in the positive direction'''
            neighbor_val = []
            flag = 1
            for i in range(len(binary_list)):
                temp = self.val[i] + (binary_list[i] * side_length)
                if temp <= 0 or temp >= 2 ** L:
                    flag = 0
                    break
                neighbor_val.append(temp)
            if flag:
                found_nbr = root.find_node(neighbor_val)
                print(found_nbr.val)
                neighbors_list.append(found_nbr)
            '''Neighbors in the negative direction'''
            neighbor_val = []
            flag = 1
            for i in range(len(binary_list)):
                temp = self.val[i] - (binary_list[i] * side_length)
                if temp <= 0 or temp >= 2 ** L:
                    flag = 0
                    break
                neighbor_val.append(temp)
            if flag:
                found_nbr = root.find_node(neighbor_val)
                print(found_nbr.val)
                neighbors_list.append(found_nbr)
        return neighbors_list


def unique_subgraphs_wrong(subgraph_num, repetitions):
    R = np.identity(subgraph_num)
    for subgraph in repetitions:
        for others in repetitions[subgraph]:
            R[subgraph - 1, others - 1] = 1
            R[others - 1, subgraph - 1] = 1
    subgraph_map = {}
    for i in range(len(R)):
        map = np.nonzero(R[i])
        subgraph_map[i] = np.min(map)
    unique_subgraphs = set(val for val in subgraph_map.values())
    print(unique_subgraphs, subgraph_map)
    return subgraph_map

def unique_subgraphs2(subgraph_num, repetitions):
    """First I find all incoming edges to all nodes"""
    incoming = {}
    for i in range(subgraph_num):
        incoming[i] = []

    for el in repetitions:
        for others in repetitions[el]:
            incoming[others].append(el)

    merger = {}
    for i in range(subgraph_num):
        merger[i] = incoming[i]
        if i in repetitions:
            merger[i].extend(repetitions[i])

    print("repetitions: ", repetitions)
    print("incoming: ", incoming)
    print("merger", merger)

    visited = np.zeros((subgraph_num,1))
    mynewset = []
    count = -1
    for node in merger:
        if visited[node] == 1:
            continue
        q = queue.Queue(maxsize=0)
        q.put(node)
        visited[node] = 1
        mynewset.append([])
        count += 1
        mynewset[count].append(node)
        while q.empty() == False:
            temp = q.get()
            if temp in merger:
                neighbors = merger[temp]
                for nbr in neighbors:
                    if visited[nbr] != 1:
                        visited[nbr] = 1
                        q.put(nbr)
                        mynewset[count].append(nbr)

    print("new # subgraphs: ", len(mynewset), "subgraphs", mynewset)
    subgraph_map = {}
    for subgraph in mynewset:
        new_no = np.min(subgraph)
        for s in subgraph:
            subgraph_map[s] = new_no

    print("subgraph_map", subgraph_map)
    return subgraph_map


    # R = np.identity(subgraph_num)
    # for subgraph in repetitions:
    #     for others in repetitions[subgraph]:
    #         R[subgraph - 1, others - 1] = 1
    #         R[others - 1, subgraph - 1] = 1
    # subgraph_map = {}
    # for i in range(len(R)):
    #     map = np.nonzero(R[i])
    #     subgraph_map[i] = np.min(map)
    # unique_subgraphs = set(val for val in subgraph_map.values())
    # print(unique_subgraphs, subgraph_map)
    # return subgraph_map



    """
    To plot the black and whote connected components returned by the color_tree function
    1. Make a scatter plot with the corresponding component size
    2. Plot thr histogram of the number of components of each size
    :param black: Dictionary of black leaves (connected components)
    :param white: Dictionary of white leaves (connected components)
    Plots them for analysis
    """
def analyse_conn_components(black, white):
    plt.figure()
    black_sizes = []
    for black_comp in black:
        dep = black[black_comp] - 1
        side = (2 ** (L - dep)) * 5
        plt.scatter(black_comp[0], black_comp[1], marker="s", c='k', s=side * side)
        black_sizes.append(dep)

    white_sizes = []
    for white_comp in white:
        dep = white[white_comp] - 1
        side = (2 ** (L - dep)) * 5
        plt.scatter(white_comp[0], white_comp[1], marker="s", c='r', s=side * side)
        white_sizes.append(dep)

    plt.figure()
    white_histogram = np.histogram(white_sizes, bins=np.arange(L+2))
    plt.bar(np.arange(L + 1), white_histogram[0])
    plt.title('White components histogram')
    print(white_histogram)
    #
    plt.figure()
    black_histogram = np.histogram(black_sizes, bins=np.arange(L + 2))
    plt.bar(np.arange(L + 1), black_histogram[0])
    plt.title('Black components histogram')
    print(black_histogram)

    # This code just plots all black, white, grey components with black, yellow, green dots
    # white = np.array([key for key in white.keys()])
    # black = np.array([key for key in black.keys()])
    # grey = np.array([key for key in grey.keys()])
    # plt.figure()
    # plt.plot(black[:,0], black[:,1], 'k.', white[:,0], white[:,1], 'y.', grey[:,0], grey[:,1], 'g.')
    # # print(len(black))

def plot_subgraphs(subgraph_list, subgraph_map, subgraph_color):
    plt.figure(3)
    subgraph_color_ind = {}

    for i in range(len(subgraph_list)):
        col = np.random.rand(3, )
        subgraph_color_ind[i] = col

    white_sub = []
    black_sub = []
    visited_cols = []
    used_cols = []
    for i in range(len(subgraph_list)):
        "This is a dictionary of center:level, hence size L-level"
        subgraph = subgraph_list[i]
        mapval = subgraph_map[i]
        if subgraph_color[i] != 1:
            if mapval not in visited_cols:
                white_sub.append(mapval)
            # white_sub.append(subgraph_map[i])
            # continue
        else:
            if mapval not in visited_cols:
                black_sub.append(mapval)
            # black_sub.append(subgraph_map[i])
        visited_cols.append(mapval)
        col = subgraph_color_ind[mapval]
        used_cols.append(mapval)
        for point in subgraph:
            side = (2 ** (L - subgraph[point])) * 4
            plt.scatter(point[0], point[1], marker="s", c=col, s=side * side)

    print("No of black subgraphs", len(set(black_sub)), "No of white subgraphs", len(set(white_sub)))
    print("used cols", len(set(used_cols)))


def assign_merged_subgraph_labels(root, subgraph_map):
    """
    Traverse the tree using a stack for Breadth First Search and assign the refined subgraph number
    :return: Black leaves
    """
    unique_subgraph_nums = list(set(subgraph_map.values()))
    newmap = {}
    for i in range(len(unique_subgraph_nums)):
        sub_num = unique_subgraph_nums[i]
        newmap[sub_num] = i

    q = queue.LifoQueue(maxsize=0)
    q.put(root)
    while q.empty() == False:
        temp = q.get()
        q.put(temp)
        if temp.children == None:
            """Reached leaf node"""
            oldnum = temp.subgraph
            temp.subgraph = newmap[subgraph_map[oldnum]]
            q.get()
        else:
            if temp.index < len(temp.children):
                """Un-traversed children still there"""
                child = temp.children[temp.index]
                temp.index += 1
                q.put(child)
            else:
                """All children traversed"""
                temp.index = 0
                q.get()
    return root, len(unique_subgraph_nums)

def find_plot_subgraphs(root, nsubgraph):
    graphs = []
    graphs_c = []
    for i in range(nsubgraph):
        graphs.append({})
        graphs_c.append(-1)

    q = queue.LifoQueue(maxsize=0)
    q.put(root)
    while q.empty() == False:
        temp = q.get()
        q.put(temp)
        if temp.children == None:
            """Reached leaf node"""
            g = temp.subgraph
            graphs[g][tuple(temp.val)] = temp.level
            if graphs_c[g]!=-1 and graphs_c[g]!=temp.color:
                print("Conflict in prev subgraph color!")
            graphs_c[g] = temp.color
            q.get()
        else:
            if temp.index < len(temp.children):
                """Un-traversed children still there"""
                child = temp.children[temp.index]
                temp.index += 1
                q.put(child)
            else:
                """All children traversed"""
                temp.index = 0
                q.get()
    return graphs, graphs_c
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # dataset = '/Users/lavisha/PycharmProjects/Project1/data_aggregation.txt'
    dataset = '/Users/lavisha/PycharmProjects/Project1/data_crescents.txt'
    d = 2
    branch_factor = 2**d
    data_handle = open(dataset, "r")
    data = data_handle.read().split('\n')[:-1]
    '''Ignore the 3rd component which represents cluster number.'''
    data = [eg.split()[:d] for eg in data]
    data = [[float(feature) for feature in example]for example in data]
    '''Scale and Quantize the data '''
    data = [[round(f) for f in example] for example in data]
    data = [[example[0]+45, example[1]+17] for example in data]
    # data = [[2,2],[4,5],[7,8],[1,6],[7,3],[5,5],[8,5],[3,3],[4,6],[7,7]]
    val = 0
    for example in data:
        if max(example)>val:
            val = max(example)
    L = math.ceil(math.log2(val))
    logger.info('n = %d d = %d val = %d L = %d',len(data), len(data[0]), val, L )

    # plt.figure(1)
    # plt.plot(np.array(data)[:,0], np.array(data)[:,1], 'k.')
    head = build_tree(data, L)
    black_leaves = np.array(head.traverse_tree())
    # plt.plot(black_leaves[:,0], black_leaves[:,1], 'r.')
    [white, black, grey] = color_tree(head)
    '''
    Above returned black nodes also correspond to black connected components B1, B2, ... Similarly white nodes correspond
    to white connected components W1, W2, ...
    '''
    print('black', black)
    print('white', white)
    print('grey', grey)
    # analyse_conn_components(black, white)

    [subgraph_num, repetitions, subgraph_list, subgraph_color] = find_subgraphs(head, L)
    print("subgraph_num",subgraph_num, "len(subgraph_list)",len(subgraph_list), "len(subgraph_color)",len(subgraph_color) )
    subgraph_map = unique_subgraphs2(subgraph_num, repetitions)
    # subgraph_map represents the true correspondence of the different subgraph numbers to merged ones. Hence subgraph num
    # of a node  = subgraph_map[node.subgraph]
    # plot_subgraphs(subgraph_list, subgraph_map, subgraph_color)

    head, nsubgraphs = assign_merged_subgraph_labels(head, subgraph_map)
    graphs, graphs_c = find_plot_subgraphs(head, nsubgraphs)
    print(len(graphs))
    for i in range(nsubgraphs):
        col = np.random.rand(3, )
        g = graphs[i]#g is a dictionary
        for pt in g:
            side = (2 ** (L - g[pt])) * 4
            plt.scatter(pt[0], pt[1], marker="s", c=col, s=side * side)





    plt.show()