import torch
import time

'''
	1) Multiple views of a storage
'''
def ex1():

	# Init all fields to 1
	M = torch.full((13,13),1)
	M = M.int()

	# Generate the lines equals to 2 (slicing operator)
	M[1,:] = 2
	M[6,:] = 2
	M[11,:] = 2
	M[:,1] = 2
	M[:,6] = 2
	M[:,11] = 2

	# Generate the blocks of 3
	M[3:5,3:5] = 3
	M[3:5,8:10] = 3
	M[8:10,3:5] = 3
	M[8:10,8:10] = 3

	print(M)
	return

'''
	2) Eigendecomposition
'''
def ex2():

	# Generate the matrix M and show its dimension
	M = torch.empty(20,20)
	M.normal_(mean=0, std=1)

	Minv = M.inverse()

	diag = torch.arange(start=1, end=21, step=1,dtype=torch.float)
	Diag = torch.diag(diag)

	Mnew = Minv.mm(Diag).mm(M)
	#Mnew = Mnew.mm(M)

	# Compute its eigenvalues
	[e,v] = Mnew.eig(eigenvectors=True)
	# Display the eigenvalues
	print(e)
	# Display the eigenvectors (column vectors)
	#print(v)
	return

'''
	3) Flops per second
'''
def ex3():
	
	M = torch.empty(5000,5000)
	M.normal_(mean=0,std=1)

	# Measure the time required for a large multiplication
	t_start = time.perf_counter()
	Result = M.mm(M)
	t_stop = time.perf_counter()

	# Compute the number of floating point products per seconds
	num_op = 5000**3
	op_per_sec = num_op/(t_stop-t_start)

	print("Elapsed time: %.3f [sec]" % ((t_stop-t_start)))
	print("Floating point products per seconds: %.3f [op/sec]" % op_per_sec)
	# about 50 billions/sec
	return


'''
	4) Playing with stride
'''
def mul_row(m):
	dim = m.size()
	for i in range(1,dim[0]):
		for j in range(0,dim[1]):
			m[i,j] = m[i,j]*(i+1)
	return m
def mul_row_fast(m):
	dim = m.size()
	coeffs = torch.arange(start=1, end=dim[0]+1, step=1,dtype=torch.float).view(dim[0],1)
	m = m*coeffs # Using braodcast mechanism
	return m
def ex4():
	
	m = torch.full((400, 1000), 2.0)
	# Use the slow function
	t_start_slow = time.perf_counter()
	m_slow = mul_row(m)
	t_stop_slow = time.perf_counter()
	print("Elapsed time (slow): %.10f [sec]" % ((t_stop_slow-t_start_slow)))

	m = torch.full((400, 1000), 2.0)
	# Use the fast function
	t_start_fast = time.perf_counter()
	m_fast = mul_row_fast(m)
	t_stop_fast = time.perf_counter()
	print("Elapsed time (fast): %.10f [sec]" % ((t_stop_fast-t_start_fast)))
	return

def main():
	print("Practical 1:")
	ex4()
	return

if __name__ == "__main__":
    main()