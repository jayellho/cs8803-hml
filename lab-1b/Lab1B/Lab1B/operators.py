from operator_base import Operator
GTID_LAST_DIGIT = 9 # 903741839


class SoftMax(Operator):
    # [B, L] -> SoftMax -> [B, L]
    def __init__(self, dim):
        super().__init__(dim=dim)
        self.op_type = 'SoftMax'

    def get_effective_dim_len(self):
        return 2

    ################################################################
    ## TODO 1.A
    ################################################################
    def get_tensors(self):
        # Function Objective = Derive the number of input & output elements in the SoftMax operation.
        # Use 2 dimensions of SoftMax inputs to derive the values.
        # Function  Output = Number of elements of input_a  and output
        B, L = self.dim[:self.get_effective_dim_len()]
        input_a = B * L
        input_b = 0 # Since single input operation, we will keep this 0.
        output = B * L
        return input_a, input_b, output

    ################################################################
    ## TODO 1.A
    ################################################################      
    def get_num_ops(self):
        # Function Objective = Derive the number of individual operations in SoftMax.
        # Consider exponential function (e^x) as (GTID+3) operations and division as (GTID+3) operations too
        # Use 2 dimensions of SoftMax inputs to derive the values.
        # Function Output = Number of operations in SoftMax
        B, L = self.dim[:self.get_effective_dim_len()] # B as number of vectors, L as the vector size
        # Assumed reuse: sum over exponents. calculation of exponents themselves reused for both numerator and the sum at the denominator.
        num_ops = B * ((L-1) + (GTID_LAST_DIGIT + 3) * 2 * L) # <B: num of vectors> * ( <add for L-sized vectors> + <exp + div for L-sized vectors>). 
        
        return num_ops
    
class layer_norm(Operator):
    # [B, X, Y] -> [B, X, Y]

    def __init__(self, dim):
        super().__init__(dim=dim)
        self.op_type = 'layer_norm'

    def get_effective_dim_len(self):
        return 3

    ################################################################
    ## TODO 1.B
    ################################################################
    def get_tensors(self):
        # Function Objective = Derive the number of input & output elements in the layer normalization operation.
        # Use dimensions of the tensor:  batch, tensor height and tensor width to derive the values.
        # Function Output = Number of elements of input_a and output
        B, X, Y = self.dim[:self.get_effective_dim_len()]
        input_a = B * X * Y
        input_b = 0 # Since single input operation, we will keep this 0.
        output = B * X * Y
        return input_a, input_b, output

    ################################################################
    ## TODO 1.B
    ################################################################      
    def get_num_ops(self):
        ## Function get_num_ops for layer normalization.
        # Substraction is considered one operation while division is considered (GT_ID+3) operations and square root is considered (GT_ID+3) operations as well
        # Also, count the number of operations to calculation mean and standard deviation
        # Use dimensions of the tensor:  batch , tensor height and tensor width to derive the values.
        # Function  Output = Total number of individual operation in the given layer normalization
        B, X, Y = self.dim[:self.get_effective_dim_len()]
        
        # 1st line - calculation of Norm(x): (<minus> + <divide>) * <all x elements i.e. Y * X * B>
        # 2nd line - calculation of mean: (<adds> + <divide>) * <num of x vectors i.e. X * B>
        # 3rd line - calculation of std dev: (<sq> + <minus> + <sq root> + <divide>) * <num of x vectors i.e. X * B>
        num_ops =  (1 + (GTID_LAST_DIGIT + 3)) * Y * B * X + \
                    ((Y-1) + (GTID_LAST_DIGIT + 3)) * B * X + \
                    (2 * Y + (GTID_LAST_DIGIT + 3) * 2) * B * X
        return num_ops
 
class GEMM(Operator):
    # [B, M, K] * [K, N] = [B, M, N]
    def __init__(self, dim):
        super().__init__(dim=dim)
        self.op_type = 'GEMM'

    def get_effective_dim_len(self):
        return 4


    ################################################################
    ## TODO 1.C
    ################################################################     
    def get_tensors(self):
        # Function Objective = Derive the number of input & output elements in the multiplication of 2 matrix.
        # Use dimensions of the matmul to derive the values.
        # A x B = C; Dim (A) = [B , M, K], Dim (B) = [K, N], Dim (C) = [B, M, N], Reduction dimension = K
        # Function Output = Number of elements of input_a , input_b and output
        B, M, N, K = self.dim[:self.get_effective_dim_len()]
        input_a = B * M * K
        input_b = K * N
        output = B * M * N
        return input_a, input_b, output


    ################################################################
    ## TODO 1.C
    ################################################################     
    def get_num_ops(self):
        # Function Objective = Derive the number of operations (multiplication + additions)
        # Use dimensions of the matmul to derive the values.
        # A x B = C; Dim (A) = [B , M, K], Dim (B) = [K, N], Dim (C) = [B, M, N], Reduction dimension = K
        # Function Output = Total number of individual operation in the given Matmul.
        B, M, N, K = self.dim[:self.get_effective_dim_len()]
        # Addition of each element is consider a single operation, i.e. 8+8+0+3 = 3 operations.
        # Multiplication of each element is consider a single operation, i.e. 2x4 = 1 operation.
        # Ex: Matrix of 1x100 and 100x1 will require 100 multiplications and 99 additions, so total no. of operations is 100 + 99 = 199
        num_ops = (2 * K - 1) * B * M * N
        return num_ops
    
    
class attention(Operator):
    # Input:[B, M, K], Wq/Wk/Wv: [B, K, N], Output [B, M, N]
    def __init__(self, dim):
        super().__init__(dim=dim)
        self.op_type = 'attention'

    def get_effective_dim_len(self):
        return 4

    ################################################################
    ## TODO 1.D
    ################################################################
    def get_tensors(self):
        ## Function get_tensors for attention.
        # Function Objective = Derive the number of elements of inputs and outputs in the scaled dot product attention operator
        # Use dimensions of the input and weights to derive the values.
        # Q = X*Wq; K = X*Wk, V = X*Wv, Output = SoftMax(Q*K^T/(sqrt(c)))*V
        # Function  Output = Number of elements of input_a , input_b (sum of of Wq, Wk, Wv elements) and output
        B, M, N, K = self.dim[:self.get_effective_dim_len()]
        input_a = B * M * K
        input_b = 3 * K * N
        output = B * M * N
        return input_a, input_b, output

    ################################################################
    ## TODO 1.D
    ################################################################      
    def get_num_ops(self):
        ## Function get_num_ops for attention
        # Function Objective = Derive the number of operations (you may reuse no. of operations in softmax and gemm)
        # Use dimensions of the input and weights to derive the values.
        # Q = X*Wq; K = X*Wk, V = X*Wv, Output = SoftMax(Q*K^T/(sqrt(c)))*V
        # Function  Output = Total number of operations in the attention operator
        B, M, N, K = self.dim[:self.get_effective_dim_len()]

        # 1st line: Q + K + V
        # 2nd line: Q*K^T / sqrt(c)+ sqrt() operation (reused)
        # 3rd line: SoftMax()
        # 4th line: * V
        num_ops = 3 * GEMM([B,M,N,K]).get_num_ops() \
                + GEMM([B,M,M,N]).get_num_ops() + (GTID_LAST_DIGIT + 3) * B * M * M + (GTID_LAST_DIGIT + 3) \
                + B * SoftMax([M,M]).get_num_ops() \
                + GEMM([B,M,N,M]).get_num_ops()

        return num_ops