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
8. Write about adding break into pieces, find boundaries and cluster hierarchy
9. Clean code

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
from PIL import Image
from scipy.misc import imresize

from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors


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
    :param L: Levels in the tree, builds tree of depth L i.e. from l=0 at root to l=L at lowermost leaves
    :return: Root of tree
    """
    d = len(data[0])
    extent = 2**L#*np.ones(d)
    center = 0.5*extent
    root_val = [center]*d
    root = node(root_val, 0, 0)
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

def analyse_conn_components(black, white):
    """
        To plot the black and white connected components returned by the color_tree function
        1. Make a scatter plot with the corresponding component size
        2. Plot thr histogram of the number of components of each size
        :param black: Dictionary of black leaves (connected components)
        :param white: Dictionary of white leaves (connected components)
        Plots them for analysis
        :param black:
        :param white:
        :return:
    """
    if d==2:
        plt.figure()
        black_sizes = []
        for black_comp in black:
            dep = black[black_comp] - 1
            side = (2 ** (L - dep)) * blocksize
            plt.scatter(black_comp[0], black_comp[1], marker="s", c='k', s=side * side)
            black_sizes.append(dep)

        white_sizes = []
        for white_comp in white:
            dep = white[white_comp] - 1
            side = (2 ** (L - dep)) * blocksize
            plt.scatter(white_comp[0], white_comp[1], marker="s", c='grey', s=side * side)
            white_sizes.append(dep)

    elif d == 3:
        # ranges = [0,5,10,15,20,25,30,35,40,45,50,55,60]
        # for i in range(len(ranges)):
        #     if i == 0:
        #         continue
            black_sizes = []
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            for black_comp in black:
                dep = black[black_comp] - 1
                side = (2 ** (L - dep)) * 5
                # if black_comp[2]>ranges[i-1] and black_comp[2]<=ranges[i]:
                ax.scatter(black_comp[0], black_comp[1], black_comp[2], c='k', marker="s")#,  s=side * side)
                black_sizes.append(dep)

            white_sizes = []
            for white_comp in white:
                dep = white[white_comp] - 1
                side = (2 ** (L - dep)) * 5
                # if white_comp[2] > ranges[i - 1] and white_comp[2] <= ranges[i]:
                # ax.scatter(white_comp[0], white_comp[1], white_comp[2], c='grey',marker="s")#,  s=side * side)
                white_sizes.append(dep)
    '''
    plt.figure()
    white_histogram = np.histogram(white_sizes, bins=np.arange(L + 2))
    plt.bar(np.arange(L + 1), white_histogram[0])
    plt.title('White components histogram')
    print(white_histogram)
    #
    plt.figure()
    black_histogram = np.histogram(black_sizes, bins=np.arange(L + 2))
    plt.bar(np.arange(L + 1), black_histogram[0])
    plt.title('Black components histogram')
    print(black_histogram)
    '''
    # This code just plots all black, white, grey components with black, yellow, green dots
    # white = np.array([key for key in white.keys()])
    # black = np.array([key for key in black.keys()])
    # grey = np.array([key for key in grey.keys()])
    # plt.figure()
    # plt.plot(black[:,0], black[:,1], 'k.', white[:,0], white[:,1], 'y.', grey[:,0], grey[:,1], 'g.')
    # # print(len(black))
    return

class node:
    def __init__(self, val, level, cnum):
        if level >L:
            print("Received greater than L=",L, level)
        self.val = val
        self.children = None# TODO: If possible change this to a numpy array of fixed length, as it's length is fixed
        self.parent = None
        self.color = 0
        self.level = level
        self.subgraph = -1
        self.aux = None
        self.index = 0
        self.subgraph_tree = []
        self.nbrs_2d = []
        self.refined_nbrs = []
        self.my_num = cnum
        """aux and index used during traversal and coloring. aux saves path from root to the node. index stores the number 
        of children visited. Both aux and index are set back to None and 0 by the end of traversal. subgraph_tree will 
        store the heirarchy of subgraph numbers. subgraph is used while finding the subgraphs and is reset to -1 after 
        the subgraphs have been determined for each level of hierarchy """

    def create_children(self, parent_size, level):
        """
        Creates 2**d i.e. branch_factor number of children hypercubes and updates the parent node's children attribute
        :param parent_size: Size of parent hypercube
        :param level: Level of children hypercubes
        :return: None
        """
        self.children = []
        print('Parent value', self.val)
        for i in range(branch_factor):
            child_val = child_center(i, self.val, parent_size)
            print("child value",child_val)
            temp_node = node(child_val, level, i)
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
        # print("For ", self.val, ' neighbors are:')
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
                # print(found_nbr.val)
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
                # print(found_nbr.val)
                neighbors_list.append(found_nbr)
        return neighbors_list

    def fneighbor(self, d, side):
        mynum = self.my_num
        D = len(self.val)
        bin_len = '{0:0' + str(D) + 'b}'
        '''Get the binary representation of mynum upto D bits'''
        binary_str = bin_len.format(mynum)
        binary_list = [int(s) for s in binary_str]
        bitd = binary_list[d]
        # print('My val', self.val)
        # if self.parent:
        #     print('My parent val', self.parent.val)
        # check if the neighbor is within bounds after all
        S = 2**(L - self.level)
        nbr_center = self.val.copy()
        nbr_center[d] += (2*side-1)*S
        # print('Finding neighbor', nbr_center)
        if nbr_center[d] > 2** L or nbr_center[d] < 0:
            # print('Returning no neighbor', nbr_center, 2**L)
            return None
        A = self.parent
        neighbor_binary = binary_list.copy()
        if bitd == 0 and side == 1:
            neighbor_binary[d] = 1
            neighbor_binary = [str(e) for e in neighbor_binary]
            binary_num = "".join(neighbor_binary)
            neighbor_c_num = int(binary_num, 2)
            # print('Children', len(self.parent.children), len(A.children))
            nbr = A.children[neighbor_c_num]
        elif bitd == 1 and side == 0:
            neighbor_binary[d] = 0
            neighbor_binary = [str(e) for e in neighbor_binary]
            binary_num = "".join(neighbor_binary)
            neighbor_c_num = int(binary_num, 2)
            nbr = A.children[neighbor_c_num]
        elif bitd == 0 and side == 0:
            B = A.fneighbor(d, 0)
            neighbor_binary[d] = 1
            neighbor_binary = [str(e) for e in neighbor_binary]
            binary_num = "".join(neighbor_binary)
            neighbor_c_num = int(binary_num, 2)
            if B.children == None:
                nbr = B
            else:
                nbr = B.children[neighbor_c_num]
        elif bitd == 1 and side == 1:
            B = A.fneighbor(d, 1)
            neighbor_binary[d] = 0
            neighbor_binary = [str(e) for e in neighbor_binary]
            binary_num = "".join(neighbor_binary)
            neighbor_c_num = int(binary_num, 2)
            if B.children == None:
                nbr = B
            else:
                nbr = B.children[neighbor_c_num]
        else:
            print('Error! No cases are left!!!! Some bug exists!!!')
        # print('Returning neighbor', nbr.val)
        return nbr

def find_neighbors_all(temp):
    temp.nbrs_2d = []
    # print('For', temp.val)
    for i in range(D):
        plusneighbor = temp.fneighbor(i, 1)
        # print('plusneighbor', plusneighbor)
        minusneighbor = temp.fneighbor(i, 0)
        # print('minusneighbor', minusneighbor)
        temp.nbrs_2d.append(plusneighbor)
        temp.nbrs_2d.append(minusneighbor)
        '''Refining the positive neighbor'''
        if plusneighbor and temp.children == None:
            q = queue.Queue(maxsize=0)
            q.put(plusneighbor)
            while(q.empty() == False):
                ptr = q.get()
                # print('Plusneighbor queue', ptr.val)
                if ptr.children:
                    # print('has children')
                    for j in range(2**D):
                        bin_len = '{0:0' + str(D) + 'b}'
                        '''Get the binary representation of j upto D bits'''
                        binary_str = bin_len.format(j)
                        # print('child', binary_str, binary_str[i])
                        if binary_str[i] == '0':
                            q.put(ptr.children[j])
                            # print('Putting in queue', ptr.children[i].val)
                else:
                    temp.refined_nbrs.append(ptr)
        '''Refining the negative neigbor'''
        if minusneighbor and temp.children == None:
            q = queue.Queue(maxsize=0)
            q.put(minusneighbor)
            while(q.empty() == False):
                ptr = q.get()
                print('Minusneighbor queue', ptr.val)
                if ptr.children:
                    for j in range(2**D):
                        bin_len = '{0:0' + str(D) + 'b}'
                        '''Get the binary representation of j upto D bits'''
                        binary_str = bin_len.format(j)
                        if binary_str[i] == '1':
                            q.put(ptr.children[j])
                            # print('Putting in queue', ptr.children[i].val)
                else:
                    temp.refined_nbrs.append(ptr)

    return


def traverse_get_neighbors(root):
    '''
    Traverse the tree in a Breadth First Search manner and for each node, find its neighbors.
    While traversing in the BFS fashion, insert into the tree children of the parent. Need not check for visited or not
    because the children can be only reached from their parent, hence will be unvisited always.
    In addition, when we find neighbors in this fashion: When a child's neighbors are being reached, its parent's neighbors
    would already have been found it. Hence, helping it.
    :param root: Root if the tree
    :return: Nothing
    '''
    q = queue.Queue(maxsize=0)
    q.put(root)
    while q.empty() == False:
        temp = q.get()
        find_neighbors_all(temp)
        print('AAAAA', temp.val, 'nbrs:')
        for nbr in temp.nbrs_2d:
            if nbr:
                print(nbr.val)
        print('BBBB', temp.val, 'allnbrs:')
        for nbr in temp.refined_nbrs:
            if nbr:
                print(nbr.val)

        if temp.children != None:
                for child in temp.children:
                    q.put(child)
    return

def numberToBase(n, b, d):
    '''
    Takes an integer n and converts it to a list of numbers to the base of b. Makes sure that the list has d numbers
    '''
    if n == 0:
        return [0]*d
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    while len(digits)<d:
        digits.append(0)
    while len(digits)<d:
        digits.append(0)
    return digits[::-1]

def getcenters(count, d):
    #There will be total count^d combinations
    total = count**d
    cc_list = []
    for i in range(total):
        #convert each of them to the base of count and get the representation till d numbers
        rep = numberToBase(i,count, d)
        cc_list.append(rep)
    return cc_list

def callf_allsize(ptr, create):
    '''For the given ptr, for each cubes circumscribing it, considers all points inside that cc_cube. If all points are white, returns a list of all those points,
    for all possible circumcubes.'''
    '''We realize that in only 1 of the calls to this function callf_allsize from BFS1(for white nodes) the all_list returned by this list is used and added to the
    list of white nodes. Hence, this flag called create tells, whether we need to break the square or not. Hence, if create is 1 and also we have found a possible big 
    square i.e. size of len(all_size)>1 then we have to break it.'''
    if ptr.subgraph !=-1:
        print('OOPS')
        return []
    create = 0
    '''for the time being just setting create = 0 for all cases, since otherwise it is running in an inifite loop'''
    all_list = []
    ptr_level = ptr.level
    side = 2**(L-ptr_level)
    count = side +1
    ptr_center = ptr.val
    edgel = [x - side/2 for x in ptr_center]
    cc_list = getcenters(count, d)
    seen = {}
    in_alllist = {}
    print('DDD For ', ptr.val)
    for item in cc_list:
        cc_center = [sum(x) for x in zip(item, edgel)]
        #Got a possible circumcube cente. Now traverse through each point lying in this circumcube and check if it is white or not
        locallist = []
        possible_points = getcenters(2*side, d)
        ccl = [x - side for x in cc_center]
        move = 0
        # print('cc center ', cc_center, ' ccl ', ccl)
        for it2 in possible_points:
            pt_center = [sum(x)+0.5 for x in zip(it2, ccl)]
            for c in pt_center:
                if c <= 0 or c >= 2 ** L:
                    move = 1
                    break
            if move:
                break
            # print('inpoint ', pt_center)
            if tuple(pt_center) in seen and seen[tuple(pt_center)] == 0:
                '''for this big cube, one of the child does not meet requirement hence move to next big cube'''
                move = 1
                break
            elif tuple(pt_center) in seen and seen[tuple(pt_center)] == 1:
                '''Hurray! this child meets requirements'''
                newptr = head.find_node(pt_center)
                locallist.append(newptr)
            else:
                newptr = head.find_node(pt_center)
                if newptr.color == 0:
                    locallist.append(newptr)
                    seen[tuple(pt_center)] = 1
                else:
                    move = 1
                    seen[tuple(pt_center)] = 0
                    break
        if move == 0:
            # alllist.extend(locallist)
            # print('CCCC For ', ptr.val, ' circumcube at ', cc_center, ' list ', locallist)
            for x in locallist:
                if tuple(x.val) not in in_alllist:
                    all_list.append(x)
                    in_alllist[tuple(x.val)] = 1
            # print("Found an all white CC cube ", locallist)

    print('From cc function ', len(all_list))


    if create == 1 and len(all_list)>0:
        '''This is being called from BFS1 and this square has the potential to form a big square. Hence, break the white nodes and only add that portion which
        is actually part of the circumscribing cube.'''
        final_list = redo_callf_task(ptr)
        return final_list
    else:
        return all_list

    return 0

"""
def redo_callf_task(ptr):
    '''
    
    :param ptr: 
    :return: 
    '''
    print('ENTERRRRRRRRRED THEEEEEE REDO FUNCTION')
    all_list = []
    ptr_level = ptr.level
    side = 2 ** (L - ptr_level)
    count = side + 1
    ptr_center = ptr.val
    edgel = [x - side / 2 for x in ptr_center]
    cc_list = getcenters(count, d)
    seen = {}
    in_alllist = {}
    print('DDD For ', ptr.val)
    for item in cc_list:
        cc_center = [sum(x) for x in zip(item, edgel)]
        # Got a possible circumcube cente. Now traverse through each point lying in this circumcube and check if it is white or not
        locallist = []
        possible_points = getcenters(2 * side, d)
        ccl = [x - side for x in cc_center]
        move = 0
        print('cc center ', cc_center, ' ccl ', ccl)
        for it2 in possible_points:
            pt_center = [sum(x) + 0.5 for x in zip(it2, ccl)]
            for c in pt_center:
                if c <= 0 or c >= 2 ** L:
                    move = 1
                    break
            if move:
                break
            print('inpoint ', pt_center)
            if tuple(pt_center) in seen and seen[tuple(pt_center)] == 0:
                '''for this big cube, one of the child does not meet requirement hence move to next big cube'''
                move = 1
                break
            elif tuple(pt_center) in seen and seen[tuple(pt_center)] == 1:
                '''Hurray! this child meets requirements'''
                newptr = head.find_node(pt_center)
                locallist.append(newptr)
            else:
                newptr = head.find_node(pt_center)
                if newptr.color == 0:
                    '''Okay, now we need to check its size'''
                    seen[tuple(pt_center)] = 1
                    if newptr.level == L:
                        locallist.append(newptr)
                    else:
                        '''if it is a bigger node then break it'''
                        mylevel = newptr.level
                        print('OKAY', mylevel, ptr_level)
                        while mylevel<L and mylevel>ptr_level:
                            '''create its children'''
                            print('Breaking', newptr.val)
                            parent_size = 2 ** (L - mylevel)
                            newptr.create_children(parent_size, mylevel + 1)
                            mylevel +=1
                            print('Created level of children at', mylevel)
                            child_num = child_location(newptr.val, pt_center)
                            newptr = newptr.children[child_num]
                            find_neighbors_all(newptr)
                            print('reached', newptr.val)
                        finalnewptr = head.find_node(pt_center)
                        if finalnewptr.level>L:
                            print('SOOOOOOOOOOOOOMMMMMMMMMMMMEEEEEEEEE Problem\n SOOOOOOOOOOOOOMMMMMMMMMMMMEEEEEEEEE Problem\n SOOOOOOOOOOOOOMMMMMMMMMMMMEEEEEEEEE Problem')
                        locallist.append(finalnewptr)
                        '''also find neighbors of this finalnewptr'''
                        find_neighbors_all(finalnewptr)
                else:
                    move = 1
                    seen[tuple(pt_center)] = 0
                    break
        if move == 0:
            # alllist.extend(locallist)
            # print('CCCC For ', ptr.val, ' circumcube at ', cc_center, ' list ', locallist)
            for x in locallist:
                if tuple(x.val) not in in_alllist:
                    all_list.append(x)
                    in_alllist[tuple(x.val)] = 1
            # print("Found an all white CC cube ", locallist)

    print('From cc function ', len(all_list))
    return all_list





"""
def callf(ptr):
    '''Returns an empty list if can't form a big hypercube. Else it returns a list of all nodes that can form the big hypercube(s)
    takes ptr --  which can be part'''

    # nbrs = ptr.neighbors_2d
    mylist = []
    seen = {}
    ptr_level = ptr.level
    seen[tuple(ptr.val)] = 1
    for i in range(2**d):
        bin_len = '{0:0' + str(d) + 'b}'
        '''Get the binary representation of child_num upto d bits'''
        binary_str = bin_len.format(i)
        binary_list = [int(s) for s in binary_str]
        '''Convert the 0,1 representation into -1,1 representation'''
        binary_list = [2 * b - 1 for b in binary_list]
        edge_val = []
        parent_size = 2**(L-ptr.level)
        ignore = 0
        for k in range(len(binary_list)):
            temp = ptr.val[k] + (binary_list[k] * parent_size / 2)
            '''Appended in the same order'''
            if temp<=0 or temp>=2**L:
                ignore = 1
                break
            edge_val.append(temp)
        if ignore:
            continue
        print('For ', ptr.val, 'considering hypercube center ', edge_val)
        '''For a hypercube centered at edge_val, go to all of its 2**d children and see if they are white and of same level as ptr'''
        locallist = []
        for j in range(2 ** d):
            bin_len1 = '{0:0' + str(d) + 'b}'
            '''Get the binary representation of child_num upto d bits'''
            binary_str1 = bin_len1.format(j)
            binary_list1 = [int(s) for s in binary_str1]
            '''Convert the 0,1 representation into -1,1 representation'''
            binary_list1 = [2 * b - 1 for b in binary_list1]
            child_val = []
            parent_size = 2 ** (L - ptr.level - 1)
            for k in range(len(binary_list1)):
                temp = edge_val[k] + (binary_list1[k] * parent_size / 4)
                '''Appended in the same order'''
                child_val.append(temp)
            '''For each child find out if it is white and of the same size'''
            if tuple(child_val) in seen and seen[tuple(child_val)] == 0:
                '''for this big cube, one of the child does not meet requirement hence move to next big cube'''
                break
            elif tuple(child_val) in seen and seen[tuple(child_val)] == 1:
                '''Hurray! this child meets requirements'''
                newptr = head.find_node(child_val)
                locallist.append(newptr)
                print('adding to locallist', newptr.val)
                continue
            else:
                newptr = head.find_node(child_val)
                if newptr.level <= ptr_level and newptr.color == 0:
                    seen[tuple(child_val)] = 1
                    locallist.append(newptr)
                    print('adding to locallist', newptr.val)
                    continue
                else:
                    seen[tuple(child_val)] = 0
                    break

        if len(locallist) == 2**d:
            mylist.extend(locallist)
    print('Seen is', seen)
    print('From cc function ', len(mylist))
    return mylist


def BFS1(temp, subgraph_num, threshold, mustadd_elements):
    '''For white nodes'''
    q = queue.Queue(maxsize=0)
    temp.subgraph = subgraph_num
    q.put(temp)
    for el in mustadd_elements:
        if el.val == temp.val:
            continue
        else:
            el.subgraph = subgraph_num
            q.put(el)
    while q.empty() == False:
        ptr = q.get()
        for nbr in ptr.refined_nbrs:
            if nbr.subgraph != -1:
                continue
            if nbr.color == 0 and nbr.level <= threshold:
                '''Big white node'''
                nbr.subgraph = subgraph_num
                q.put(nbr)
            elif nbr.color == 0 and nbr.level == threshold + 1:
                """2nd big white node"""
                print('call from bfs1')
                circumsube_elements = callf_allsize(nbr,1)
                for el in circumsube_elements:
                    el.subgraph = subgraph_num
                    q.put(el)
    return


def BFS2(temp, subgraph_num, threshold):
    '''For black nodes'''
    q = queue.Queue(maxsize=0)
    temp.subgraph = subgraph_num
    q.put(temp)

    while q.empty() == False:
        ptr = q.get()
        for nbr in ptr.refined_nbrs:
            if nbr.subgraph != -1:
                continue
            if nbr.color == 0 and nbr.level <= threshold:
                '''Big white node'''
                continue
            elif nbr.color == 0 and nbr.level == threshold + 1:
                '''Second big white node'''
                print('call from bfs2')
                circumcube_elements = callf_allsize(nbr,0)
                if len(circumcube_elements) == 0:
                    '''This can't form a big white cube'''
                    nbr.subgraph = subgraph_num
                    q.put(nbr)
            else:
                nbr.subgraph = subgraph_num
                q.put(nbr)
    return

def find_subgraphs(root, L, threshold):
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
    # repetitions = {}
    # subgraph_list = []
    subgraph_color = []
    while stack.empty() == False:
        temp = stack.get()
        # print('Stack - ', temp.val)
        stack.put(temp)
        if temp.children == None:
            """Reached a leaf node. Start forming a sub-graph"""
            if temp.subgraph!=-1:
                """Check that this node is part of some other subgraph"""
                temp.aux = None
                stack.get()
            else:
                """Start a BFS for a subgraph formation for this node"""
                subgraph_num +=1
                # subgraph_list.append({})
                # subgraph_list[subgraph_num][tuple(temp.val)] = temp.level
                # print('list', subgraph_list)
                # print('color', subgraph_color)
                # print('subgraph_num', subgraph_num)
                if temp.color == 0 and temp.level <= threshold:
                    '''This is a big white node'''
                    cicumcube_elements = []
                    BFS1(temp, subgraph_num, threshold,cicumcube_elements)
                    subgraph_color.append(0)
                elif temp.color == 1 or temp.level >threshold + 1:
                    '''This is either a black node or a small white node'''
                    BFS2(temp, subgraph_num, threshold)
                    subgraph_color.append(1)
                elif temp.color == 0 and temp.level == threshold+1:
                    '''This is the next big sized white node'''
                    print('call from find_subgraphs')
                    circumcube_elements = callf_allsize(temp,0)
                    if len(circumcube_elements) == 0:
                        '''But this can not form a big white square hence consider as a black node'''
                        BFS2(temp, subgraph_num, threshold)
                        subgraph_color.append(1)
                    else:
                        '''Yes, this can form a big white square hence consider it as a white node'''
                        BFS1(temp, subgraph_num, threshold, circumcube_elements)
                        subgraph_color.append(0)

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
    # return [subgraph_num+1, repetitions, subgraph_list, subgraph_color]
    return [subgraph_num+1, subgraph_color]

def traverse_get_subgraphs(root, nsubgraph):
    """
    This is a generic function which simply traverses the entire tree through depth first search and outputs the
    subgraphs at a specific level
    :param root: Root of tree
    :param nsubgraph: No of subgraphs expected at that level
    :param h_num: Hierarchy number of subgraphs to return
    :return: A list of dictionaries. Each dictionary is a subgraph with keys=node values and val as their level
    """
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
            # g = temp.subgraph_tree[h_num]
            g = temp.subgraph
            graphs[g][tuple(temp.val)] = temp.level
            if graphs_c[g]!=-1 and graphs_c[g]!=temp.color:
                print("Conflict in prev subgraph color!", graphs_c[g],temp.color )
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
    # dataset = '/Users/lavisha/PycharmProjects/Project1/data_crescents.txt'
    # dataset = '/Users/lavisha/PycharmProjects/Project1/noisy_circles.txt'
    # dataset = '/Users/lavisha/PycharmProjects/Project1/noisy_moons.txt'
    # dataset = '/Users/lavisha/PycharmProjects/Project1/blobs.txt'
    # dataset = '/Users/lavisha/PycharmProjects/Project1/aniso.txt'
    dataset = '/Users/lavisha/PycharmProjects/Project1/varied.txt'
    # dataset = '/Users/lavisha/PycharmProjects/Project1/no_structure.txt'
    # dataset = '/Users/lavisha/PycharmProjects/Project1/my3d.txt'
    # dataset = '/Users/lavisha/PycharmProjects/Project1/my3d_analysis_2.txt'
    #
    # pix1 = np.array(Image.open('/Users/lavisha/Desktop/seg2.jpg'))
    # pix2 = imresize(pix1, 0.1)
    # # plt.imshow(pix2)
    # # plt.show()
    # pix3 = pix2.reshape((-1,3))
    d = 2
    D = d
    blocksize = 1
    branch_factor = 2**d
    data_handle = open(dataset, "r")
    data = data_handle.read().split('\n')[:-1]
    '''Ignore the 3rd component which represents cluster number.'''
    data = [eg.split()[:d] for eg in data]
    data = [[float(feature)*5 for feature in example]for example in data]
    '''Scale and Quantize the data '''
    data = [[round(f) for f in example] for example in data]
    # data = [[example[0]+45, example[1]+17] for example in data]#crescent
    # data = [[example[0]+62, example[1]+60 ] for example in data]#noisy circles
    # data = [[example[0] + 55, example[1] + 50] for example in data]  # noisy moons
    # data = [[example[0] + 80, example[1] + 126] for example in data]  # blobs
    # data = [[example[0] + 60, example[1] + 50] for example in data]  # aniso
    data = [[example[0] + 70, example[1] + 50] for example in data]  # varied
    # data = [[example[0] + 10, example[1] + 10] for example in data]  # no_Structure
    # data = [[example[0] + 35, example[1] + 30, example[2] + 40] for example in data]#3d
    # data = [[2,2],[4,5],[7,8],[1,6],[7,3],[5,5],[8,5],[3,3],[4,6],[7,7]]
    # data = pix3
    val = 0
    for example in data:
        if max(example)>val:
            val = max(example)
    print(val)
    L = math.ceil(math.log2(val))
    logger.info('n = %d d = %d val = %d L = %d',len(data), len(data[0]), val, L )

    if d == 2:
        print('2d')
        # plt.figure()
        # plt.plot(np.array(data)[:,0], np.array(data)[:,1], 'k.')
    elif d == 3:
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(np.array(data)[:,0], np.array(data)[:,1], np.array(data)[:,2])
        print("max",np.max(np.array(data)[:,0]),np.max(np.array(data)[:,1]),np.max(np.array(data)[:,2]), "min",np.min(np.array(data)[:,0]),np.min(np.array(data)[:,1]),np.min(np.array(data)[:,2]))
    # plt.show()
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

    print('b = ', len(black), '  w = ', len(white), '   g = ',len(grey))
    print('Total = ', len(black) + len(white) + len(grey), '  Nodes = ', len(data))
    print('n = ', len(data), ' d = ', len(data[0]),' val = ',  val, ' L = ', L)

    analyse_conn_components(black, white)
    print('Finding neighbors')
    traverse_get_neighbors(head)
    print('Done with neighbors')

    [subgraph_num, subgraph_color] = find_subgraphs(head, L, 6)
    graphs2, graphs_c2 = traverse_get_subgraphs(head, subgraph_num)  # Traverse the tree and from there, get subgraph numbers for each node

    print(len(graphs2))
    fig = plt.figure()
    for i in range(subgraph_num):
        col = np.random.rand(3, )
        g = graphs2[i]  # g is a dictionary
        for pt in g:
            side = (2 ** (L - g[pt])) * blocksize
            plt.scatter(pt[0], pt[1], marker="s", c=col, s=side * side)
    print(subgraph_num, sum(subgraph_color), subgraph_color)
    #######################
    # """
    num_knn = 2**d
    '''Rectify the 1 point clusters'''
    data = np.unique(data, axis = 0)
    knn = NearestNeighbors(n_neighbors=num_knn).fit(data)
    distances, indices = knn.kneighbors(data)
    meanval = np.mean(distances, axis=1)
    
    for g in range(len(graphs2)):
        if len(graphs2[g]) == 1 and subgraph_color[g] == 1:
            '''This is a 1 point black cluster'''
            print('mmm', list(graphs2[g].keys())[0])
            key = list(graphs2[g].keys())[0]
            x1 = key[0]+0.5
            y1 = key[1]+0.5
            ans = knn.kneighbors([[x1, y1]], 2, return_distance=True)
            nbr_closest = ans[1][0][1]
            ratio = (ans[0][0][1]/meanval[nbr_closest])
            print('ratio ', ratio, ' nbr', data[nbr_closest])
            if ratio < np.pi:
                data2 = [ex - 0.5 for ex in [x1, y1]]
                nodeptr = head.find_node(data2)
                data3 = [ex - 0.5 for ex in data[nbr_closest]]
                bigptr = head.find_node(data3)
                print('Old subgraph: ', nodeptr.subgraph)
                nodeptr.subgraph = bigptr.subgraph
                print('New subgraph: ', nodeptr.subgraph)

    # """
    ##################
    num_clusters = 350
    cols = []
    # print('Detected ', num_clusters, ' clusters')
    for i in range(num_clusters):
        col = np.random.rand(3, )
        cols.append(col)
    if d == 2:
        plt.figure()
        cl_num = []
        for i in range(len(data)):
            # print(Y[i]+1)
            data2 = [ex-0.5 for ex in data[i]]
            y = head.find_node(data2).subgraph
            if y not in cl_num:
                cl_num.append(y)
            # plt.scatter(data[i][0]- 45, data[i][1] - 17, s=3.5, c=cols[y])
            plt.scatter(data[i][0], data[i][1], s=9, c=cols[y])
        print(len(cl_num), 'cl_num', cl_num)
    elif d == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        cl_num = []
        for i in range(len(data)):
            data2 = [ex - 0.5 for ex in data[i]]
            y = head.find_node(data2).subgraph
            if y not in cl_num:
                cl_num.append(y)
            # plt.scatter(data[i][0]- 45, data[i][1] - 17, s=3.5, c=cols[y])
            ax.scatter(data[i][0], data[i][1], data[i][2], c=cols[y])
        print(len(cl_num), 'cl_num', cl_num)
    #############
    logger.info('n = %d d = %d val = %d L = %d', len(data), len(data[0]), val, L)
    plt.show()