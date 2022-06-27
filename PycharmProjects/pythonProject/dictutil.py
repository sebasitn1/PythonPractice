# Copyright 2013 Philip N. Klein
from typing import Optional, Any, Type


def dict2list(dct, keylist): return [dct[n] for n in keylist]


def list2dict(L, keylist): return {x: y for (x, y) in zip(keylist, L)}


def listrange2dict(L): return {L[n]: n for n in range(len(L))}


def makeInverseIndex(strlist):
    line2number = list(enumerate(strlist))
    linecount = range(len(line2number))
    for n in linecount:
        words = [(w, n) for n in linecount for w in line2number[n][1].split()]
    wordset = {x for (x, y) in words}
    wordlist = list(wordset)
    reccurList = dict()
    for w in range(len(wordlist)):
        current = str(wordlist[w])
        reccur = set()
        for n in range(len(words)):
            if current in words[n]:
                reccur.add(words[n][1])
        reccurList[current] = reccur


def orSearch(inverseIndex, query):
    return {y for n in range(len(query)) for y in inverseIndex[query[n]]}


def andSearch(inverseIndex, query):
    intersection = set()
    group = list(orSearch(inverseIndex, query))
    for n in range(len(query)):
        current = query[n]
        for k in range(len(group)):
            selection = group[k]
            compart = inverseIndex[current]
            if selection not in compart:
                return {}
            else:
                intersection.add(group[k])
    return intersection


def increments(L):
    return [x + 1 for x in L]


def cubes(L):
    return [x ** 3 for x in L]


def tuple_sum(A, B):
    return [(x + n, y + k) for i in range(len(A)) for (x, y) in A for (n, k) in B if x in A[i] and n in B[i]]


def inv_dict(d):
    return {f: e for n in range(len((d.keys()))) for e in d.keys() for f in d.values() if d[e] == f}


def row(p, n):
    return [p + n for n in range(n)]


# [[row(p,15) for p in range(20)]]
# [[p+n for n in range(15)] for p in range(20)]\
# im = [{x-1j*y for y in range(len(data)) for x in range(len(data[y])) for q in data[y][x]  if q  < 120}]
# im = [{x-1j*y+1j*len(data) for y in range(len(data)) for x in range(len(data[y])) for q in data[y][x]  if q  < 120}]
def f(z):
    zreals = [z.real for z in z]
    zimags = [z.imag for z in z]
    length = (max(zreals) - min(zreals)) / 2 + min(zreals)
    height = (max(zimags) - min(zimags)) / 2 + min(zimags)
    return {z - length - 1j * height for z in z}


# w = [math.e**((2*math.pi*1j)/n) for n in range(20) if n>0]
# plot(w,3)
# Spi4 = {S*math.e**((math.pi/4)*1j) for S in S}
# shiftpic = {math.e**((math.pi/4)*1j) * f * 1/2 for f in f(im[0])}

# alpha = {'A': 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4, 'F':5, "G" : 6, "H" : 7, "I" : 8, "J" : 9,
#  'K':10, 'L':11, 'M':12, 'N':13, "O":14, "P":15, "Q":16, "R":17, "S":18, "T":19, "U":20,
# 'V':21, 'W':22, "X":23, 'Y':24, "Z":25}


# alphabase2
# {'A': (0, 0, 0, 0, 0), 'B': (0, 0, 0, 0, 1), 'C': (0, 0, 0, 1, 0), 'D': (0, 0, 0, 1, 1), 'E': (0, 0, 1, 0, 0), 'F': (0, 0, 1, 0, 1), 'G': (0, 0, 1, 1, 0), 'H': (0, 0, 1, 1, 1), 'I': (0, 1, 0, 0, 0), 'J': (0, 1, 0, 0, 1), 'K': (0, 1, 0, 1, 0), 'L': (0, 1, 0, 1, 1), 'M': (0, 1, 1, 0, 0), 'N': (0, 1, 1, 0, 1), 'O': (0, 1, 1, 1, 0), 'P': (0, 1, 1, 1, 1), 'Q': (1, 0, 0, 0, 0), 'R': (1, 0, 0, 0, 1), 'S': (1, 0, 0, 1, 0), 'T': (1, 0, 0, 1, 1), 'U': (1, 0, 1, 0, 0), 'V': (1, 0, 1, 0, 1), 'W': (1, 0, 1, 1, 0), 'X': (1, 0, 1, 1, 1), 'Y': (1, 1, 0, 0, 0), 'Z': (1, 1, 0, 0, 1)}

# nualpha = inv_dict(alpha)
# alphabase2 = {nualpha[n]: (x,y,z,a,b) for n in range(26) for x in base2 for y in base2 for z in base2 for a in base2 for b in base2
# if x*2**4+y*2**3+z*2**2+a*2**1+b*2**0==n}

def myfilter(L, num):
    return [l for l in L if l % num != 0]


def my_lists(L):
    return [[n + l for n in range(l)] for l in L]


def my_function_composition(f, g):
    return {k: v for k in f.keys() for v in g.values() if v == g[f[k]]}


def mySum(L):
    current = 0
    for x in L:
        current = x + current
    return current


def myProduct(L):
    current = 1
    for x in L:
        current = x * current
    return current


def myMin(L):
    current = L[0]
    for x in L:
        current = x if x < current else current
    return current


def myConcat(L):
    current = ""
    for x in L:
        current = current + x
    return current


def myUnion(L):
    current = set()
    for x in L:
        current = current.union(x)
    return current


def transform(a, b, L):
    return [l * a + b for l in L]


def add2(v, w):
    return [v[0] + w[0], v[1] + w[1]]


def quiz242(v):
    return [v[0] + 1, v[1] + 2]


def addn(v, w):
    return [v[n] + w[n] for n in range(len(v))]


def scalar_vector_mult(alpha, v):
    return [alpha * v[i] for i in range(len(v))]


def addn2(v, w): return [x + y for (x, y) in zip(v, w)]


def segment(pt1, pt2):
    return [x + y for (x, y) in zip(v, w) for a in range(101) for b in range(101) for x in
            scalar_vector_mult(a / 100, pt1) for y in scalar_vector_mult(b / 100, pt2) if a + b == 100]


def findTimings(strlist):
    for n in range(8):
        datapoint2number = list(strlist)[n].split()


def pfilelist(a_file):
    list_of_lists = []
    listOfIts = []
    listOftim = []
    my_finallist = []
    for line in a_file:
        stripped_line = line.strip()
        line_list = stripped_line.split()
        list_of_lists.append(line_list)
    for n in range(len(list_of_lists)):
        nCount = int(list_of_lists[n][0])
        listOfIts.append(nCount)
        timeValue = float(list_of_lists[n][1])
        listOftim.append(timeValue)
    for n in range(len(list_of_lists)):
        rightTuple = [(l, t) for l in listOfIts for t in listOftim for
                      n in range(len(listOfIts)) if l == listOfIts[n] and t == listOftim[n]]
        [my_finallist.append(n) for n in rightTuple if n not in my_finallist]

    print(' \n'.join(str(elt) + " " + "[" + k.__str__() + "]"
                     for elt in my_finallist for k in range(20) if
                     (k + 1) * 100000 == elt[0] and my_finallist[k] == elt))


def matrixL(rows, columns):
    matrix = []
    localVals = " "
    rowParam = ""
    colParam = ""
    for n in range(rows):
        for n in range(columns):
            rowVal = input("Enter row by entering each item and pressing enter after each: ")
            localVals = localVals + rowVal
            if n < columns - 1:
                localVals = localVals + " & "
        localVals = localVals + "\\\\" + "\n"
        matrix.append(localVals)
        localVals = " "
    for n in range(columns):
        colParam = colParam + "r"

    latexRepresentation = "\left[\n" + "\\begin{array}{" + colParam + "}\n"
    for n in range(rows):
        latexRepresentation = latexRepresentation + matrix[n]

    latexRepresentation = latexRepresentation + "\end{array}\n\\right]\n"
    return print(latexRepresentation)


def arrowL(args):
    arrowL = ""
    rowVals = []
    if args == 0:
        rowVal = input("Enter row to be replaced: ")
        scalar = input("Enter an intended scalar for the operate-with row")
        return print("\overset{(" + scalar + ")" + "\\rho_" + rowVal + "}{\longrightarrow}")
    if args == 1:
        for n in range(1):
            rowVal = input("Enter row to be replaced: ")
            rowVals.append(rowVal)
            rowVal = input("Enter row to operate with: ")
            rowVals.append(rowVal)
            operation = input("Enter 0 to exchange rows or 1 to replace same row: ")
            if operation == "0":
                return  print("\overset{\\rho_" + rowVals[0] + " \leftrightarrow \\rho_" + rowVals[1] + "}{\longrightarrow}")
            else:
                scalar = input("Enter an intended scalar for the operate-with row")
                return  print("\overset{\\rho_" + rowVals[0] + " +" + " (" + scalar + ")\\rho_" + rowVals[1] + "}{\longrightarrow}")


# PythonPractice
# PythonPractice
# f = open('JavaSortTymes.txt')
# pfilelist(f)
# a_file = f
