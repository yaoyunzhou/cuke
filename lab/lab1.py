import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ir import *
from codegen.cpu import *


def PrintCCode(ir):
    code = ''
    for d in ir:
        if d:
            code += to_string(d)
    print(code)


def Loop0():
    ir = []

    N = Scalar('int', 'N')
    M = Scalar('int', 'M')
    L = Scalar('int', 'L')
    A = Ndarray('int', (N, M, L), 'A')
    B = Ndarray('int', (N, M, L), 'B')

    loopi = Loop(0, N, 1, [])
    loopj = Loop(0, M, 1, [])
    loopk = Loop(0, L, 1, [])

    loopi.body.append(loopj)
    loopj.body.append(loopk)

    lhs1 = Index(Index(Index(A, Expr(loopi.iterate, 1, '+')), loopj.iterate), loopk.iterate)
    lhs2 = Index(Index(Index(B, Expr(loopi.iterate, 1, '+')), Expr(loopj.iterate, 2, '+')), Expr(loopk.iterate, 1, '-'))
    rhs1 = Index(Index(Index(B, Expr(loopi.iterate, 1, '+')), loopj.iterate), Expr(loopk.iterate, 1, '-'))
    rhs2 = Index(Index(Index(A, loopi.iterate), loopj.iterate), Expr(loopk.iterate, 1, '+'))
    rhs3 = Index(Index(Index(B, loopi.iterate), Expr(loopj.iterate, 2, '+')), loopk.iterate)

    # body = Assignment(lhs, Expr(rhs1, rhs2, '+'))
    loopk.body.extend([Assignment(lhs1, Expr(rhs1, 2, '+')), Assignment(lhs2, Expr(rhs2, rhs3, '+'))])

    ir.extend([Decl(L)])
    ir.extend([Decl(M)])
    ir.extend([Decl(N)])
    ir.extend([Decl(A)])
    ir.extend([loopi])

    return ir


# for ( k = 0; k < L ; ++k ){
# 	for ( j = 0; j < M; ++ j ){
# 		for ( i = 0; i < N; ++ i ){
# 			a[i+1] [j+1] [k] = a [i] [j] [k] + a [i] [j + 1] [k + 1] ;
# 		}
# 	}
# }

# Distance Vector:
# [1, 1, 0] :  a[i+1] [j+1] [k] and a [i] [j] [k]
# [1, 0, -1] : a[i+1] [j+1] [k] and  a [i] [j + 1] [k + 1]

# Direction Vector:
# [<, <, =]
# [<, =, >]

def Loop1():
    ir = []

    L = Scalar('int', 'L')
    M = Scalar('int', 'M')
    N = Scalar('int', 'N')
    A = Ndarray('int', (N, M, L), 'A')

    loopk = Loop(0, L, 1, [])
    loopj = Loop(0, M, 1, [])
    loopi = Loop(0, N, 1, [])
    loopk.body.append(loopj)
    loopj.body.append(loopi)

    lhs = Index(Index(Index(A, Expr(loopi.iterate, 1, '+')), Expr(loopj.iterate, 1, '+')), loopk.iterate)
    rhs1 = Index(Index(Index(A, loopi.iterate), loopj.iterate), loopk.iterate)
    rhs2 = Index(Index(Index(A, loopi.iterate), Expr(loopj.iterate, 1, '+')), Expr(loopk.iterate, 1, '+'))

    body = Assignment(lhs, Expr(rhs1, rhs2, '+'))
    loopi.body.append(body)

    ir.extend([Decl(L)])
    ir.extend([Decl(M)])
    ir.extend([Decl(N)])
    ir.extend([Decl(A)])
    ir.extend([loopk])

    return ir


# for ( i = 0; i < N ; ++i ){
# 	for ( j = 0; j < N; ++ j ){
# 			a[i][j] = a[i+1][j-1];
# 	}
# }

# Distance Vector:
# [-1, 1]

# Direction Vector:
# [<, >]

def Loop2():
    ir = []

    N = Scalar('int', 'N')
    A = Ndarray('int', (N, N), 'A')

    loopi = Loop(0, N, 1, [])
    loopj = Loop(0, N, 1, [])

    loopi.body.append(loopj)

    lhs = Index(Index(A, loopi.iterate), loopj.iterate)
    rhs = Index(Index(A, Expr(loopi.iterate, 1, '+')), Expr(loopj.iterate, 1, '-'))

    loopj.body.append(Assignment(lhs, rhs))

    ir.extend([Decl(N)])
    ir.extend([Decl(A)])
    ir.extend([loopi])

    return ir


'''
A[l0+1][l1][l2] = B[l0+1][l1][l2-1] + 2;
B[l0+1][l1+2][l2-1] = A[l0][l1][l2+1] + B[l0][l1+2][l2];


write statement array: [A[l0+1][l1][l2], B[l0+1][l1+2][l2-1]]
read statement array: [B[l0+1][l1][l2-1], A[l0][l1][l2+1], B[l0][l1+2][l2]]

Group index statement by names
write dict: {'A': [A[l0+1][l1][l2]], 'B': [B[l0+1][l1+2][l2-1]]}
read dict: {'A': [A[l0][l1][l2+1]], 'B': [B[l0+1][l1][l2-1], B[l0][l1+2][l2]]}

write read & write write

Compute the direction vector
Distance vector
A: [1, 0, -1]
B1:[0, 2, 0]
B2:[1, 0, -1]

Direction vector
A: [<, =, >]
B1:[=, <, =]
B2:[<, =, >]

safety checking based on direction vector
Exchange [0,1] first non-equal is '<', this is safe
A: [=, <, >]
B1:[<, =, =]
B2:[=, <, >]

Exchange [0,2]  not safe
A: [>, =, <]    The first non-equal is '>', not valid.
B1:[=, <, =]
B2:[>, =, <]

Exchange [1,2] 
A: [<, >, =]
B1:[=, =, <]
B2:[<, >, =]
'''


def FindBody(loop):
    if not isinstance(loop, Loop):
        return loop
    if isinstance(loop.body[0], Loop):
        return FindBody(loop.body[0])
    else:
        return loop.body
    

def get_index(statement, is_write, write_expr=[], read_expr=[]):
    if isinstance(statement, Ndarray) or isinstance(statement, Index):
        if is_write:
            write_expr.append(statement)
        else:
            read_expr.append(statement)
    elif isinstance(statement, Assignment):
        get_index(statement.lhs, True, write_expr, read_expr)
        get_index(statement.rhs, False, write_expr, read_expr)
    elif isinstance(statement, Expr):
        get_index(statement.left, is_write, write_expr, read_expr)
        get_index(statement.right, is_write, write_expr, read_expr)
    else:
        return


def get_array(wr_dict, stmt, parent_stmt):
    if isinstance(stmt.dobject, Ndarray):
        if stmt.dobject.name() in wr_dict.keys():
            wr_dict[stmt.dobject.name()].append(parent_stmt)
        else:
            wr_dict[stmt.dobject.name()] = [parent_stmt]
    elif isinstance(stmt.dobject, Index):
        get_array(wr_dict, stmt.dobject, parent_stmt)


def get_iterates(idx, vecs):
    if isinstance(idx.dobject, Index):
        # print(idx.index)
        if isinstance(idx.index, Scalar):
            if idx.index.val is None:
                vecs.append(0)
        elif isinstance(idx.index, Expr):
            if idx.index.op == '+':
                vecs.append(idx.index.right)
            elif idx.index.op == '-':
                vecs.append(-idx.index.right)
        get_iterates(idx.dobject, vecs)
    else:
        if isinstance(idx.index, Scalar):
            if idx.index.val is None:
                vecs.append(0)
        elif isinstance(idx.index, Expr):
            if idx.index.op == '+':
                vecs.append(idx.index.right)
            elif idx.index.op == '-':
                vecs.append(-idx.index.right)


def direction_vec(write_idx, read_idx):
    wvec = []
    rvec = []
    get_iterates(write_idx, wvec)
    get_iterates(read_idx, rvec)
    print(wvec, rvec)
    return [x - y for x, y in zip(wvec, rvec)]


def swap_matrix_columns(matrix, i, j):
    # i, j in the range of matrix
    if i < 0 or i >= len(matrix[0]) or j < 0 or j >= len(matrix[0]):
        return matrix  # if it above the range, then no swap 

    # create a new matrix to 
    swapped_matrix = [row[:] for row in matrix]

    # swap the i element and the j element 
    for row in swapped_matrix:
        row[i], row[j] = row[j], row[i]

    return swapped_matrix

def safety_checking(vec):
    for irow in vec:
        cnt = 0
        if irow[0]==0:
            while irow[cnt]==0:
                cnt += 1
            if irow[cnt]>0: continue
            if irow[cnt]<0: return False
        
        elif irow[0]>0:
            continue
        else:
            return False
    return True




def InterchangeLoop(ir, loop_idx=[]):
    ir_res = []
    interchangeable = True
    our_loop = None
    loop_vec = [] # store loop L,M,N ...
    write_expr = []
    read_expr = []
    write_dict = dict()
    read_dict = dict()
    for ir_item in ir:
        if isinstance(ir_item, Loop):
            # loop_vec.append(ir_item)
            body = FindBody(ir_item)
            # loop_vec.append(body)
            # print("----------------")
            # PrintCCode(loop_vec)
            for body_item in body:
                get_index(body_item, False, write_expr, read_expr)

            # PrintCCode(write_expr)
            # PrintCCode(read_expr)
            print(write_expr, read_expr)
            for i in write_expr:
                get_array(write_dict, i, i)
            for i in read_expr:
                get_array(read_dict, i, i)
            # print(write_dict)
            # print(read_dict)

            # PrintCCode(write_dict['A'])
            # PrintCCode(write_dict['B'])

            # PrintCCode(read_dict['A'])
            # PrintCCode(read_dict['B'])

            # vecs = []
            # get_iterates(write_dict['A'][0], vecs)
            # get_iterates(write_dict['B'][0], vecs)

            # get_iterates(read_dict['A'][0], vecs)
            # get_iterates(read_dict['B'][0], vecs)
            # get_iterates(read_dict['B'][1], vecs)
            # print(vecs)
            d_vec = []
            
            for key in write_dict.keys():
                tmp = read_dict[key]
                print("read = ",tmp)
                print(write_dict[key])
                print(write_dict[key][0])
                for i in tmp:
                    vec = direction_vec(write_dict[key][0], i)
                    d_vec.append(vec[::-1])
            # print("d_vec",d_vec[0])
            # print(d_vec)
            # print(d_vec[0])
            swap_vec = swap_matrix_columns(d_vec,loop_idx[0],loop_idx[1])
            interchangeable = safety_checking(swap_vec)
            if interchangeable:
                loops = []
                tmp = ir_item
                while isinstance(tmp,Loop):
                    loops.append(tmp)
                    tmp = tmp.body[0]
                print(loops)

                body = loops[-1].body
                print(body)
                tmp = loops[loop_idx[0]]
                loops[loop_idx[0]] = loops[loop_idx[1]]
                loops[loop_idx[1]] = tmp
                optimized_code = loops[0]
                for idx,item in enumerate(loops):
                    # print(idx,item,end="==")
                    if idx > 0:
                        loops[idx-1].body = [item]
                loops[-1].body = body
                ir_res.append(optimized_code)    
        else:
            ir_res.append(ir_item)
        

    # print("Please implement the pass here")
    return interchangeable, ir_res


if __name__ == "__main__":
    loop0_ir = Loop0()
    loop1_ir = Loop1()
    loop2_ir = Loop2()
    # PrintCCode(loop0_ir)

    optimized_loop0_ir, ir_res = InterchangeLoop(loop0_ir, [0, 1])
    PrintCCode(ir_res)
    print(optimized_loop0_ir)
    # PrintCCode(optimized_loop0_ir)
    # optimized_loop1_ir = InterchangeLoop(loop1_ir, [1, 2]):
    # optimized_loop2_ir = InterchangeLoop(loop2_ir, [0, 1]):

    # optimized_ir = LoopInterchange(ir)
    # print("Loop after interchange:")
    # PrintCCode(optimized_ir)
