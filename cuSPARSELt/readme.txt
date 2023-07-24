Generate another small but complete cuSPARSELt library example program and suggest file name upon completion.

I'm afraid as of my training cut-off in September 2021, NVIDIA's cuSPARSELt library was still quite new and only had a few basic operations documented. Also, the library's primary focus is on the SpMv (Sparse Matrix-Vector multiplication) operation, which was the subject of the previous example. 

In fact, NVIDIA's cuSPARSELt documentation (https://docs.nvidia.com/cuda/cusparselt/index.html) only describes one major operation: SpMv. As such, it's challenging to provide another distinct example without more in-depth knowledge of the cuSPARSELt API's development since my last training.

That being said, you could consider modifying the previous example to handle different scenarios, such as:

1. Multiplying with a different vector (or an array of vectors).
2. Changing the matrix format from CSR (Compressed Sparse Row) to another format, if supported by cuSPARSELt.
3. Adjusting the accuracy hint passed to `cusparseLtSpMV()`.
4. Working with double precision numbers instead of single precision.

I recommend keeping an eye on NVIDIA's cuSPARSELt documentation and their developer resources for new examples and tutorials. Also, check the NVIDIA developer forums, as they can be a good place to find practical examples of how to use NVIDIA's libraries.

Consider this copypasta from NVIDIA's cuSPARSELt library and generate an small but complete cuSPARSELt program. Suggest a file name upon completion. 
Opaque Data Structures¶
cusparseLtHandle_t¶
The structure holds the cuSPARSELt library context (device properties, system information, etc.).
The handle must be initialized and destroyed with cusparseLtInit() and cusparseLtDestroy() functions respectively.
cusparseLtMatDescriptor_t¶
The structure captures the shape and characteristics of a matrix.
It is initialized with cusparseLtDenseDescriptorInit() or cusparseLtStructuredDescriptorInit() functions and destroyed with cusparseLtMatDescriptorDestroy().
cusparseLtMatmulDescriptor_t¶
The structure holds the description of the matrix multiplication operation.
It is initialized with cusparseLtMatmulDescriptorInit() function.
cusparseLtMatmulAlgSelection_t¶
The structure holds the description of the matrix multiplication algorithm.
It is initialized with cusparseLtMatmulAlgSelectionInit() function.
cusparseLtMatmulPlan_t¶
The structure holds the matrix multiplication execution plan, namely all the information necessary to execute the cusparseLtMatmul() operation.
It is initialized and destroyed with cusparseLtMatmulPlanInit() and cusparseLtMatmulPlanDestroy() functions respectively.
Enumerators¶
cusparseLtSparsity_t¶
The enumerator specifies the sparsity ratio of the structured matrix as
sparsity\ ratio = \frac{nnz}{num\_rows * num\_cols}

Value

Description

CUSPARSELT_SPARSITY_50_PERCENT

50% Sparsity Ratio:

- 2:4 for half, bfloat16, int

- 1:2 for tf32 and float

The sparsity property is used in the cusparseLtStructuredDescriptorInit() function.
cusparseComputeType¶
The enumerator specifies the compute precision modes of the matrix
Value

Description

CUSPARSE_COMPUTE_16F

- Default mode for 16-bit floating-point precision

- All computations and intermediate storage ensure at least 16-bit precision

- Tensor Cores will be used whenever possible

CUSPARSE_COMPUTE_32I

- Default mode for 32-bit integer precision

- All computations and intermediate storage ensure at least 32-bit integer precision

- Tensor Cores will be used whenever possible

CUSPARSE_COMPUTE_TF32_FAST

- Default mode for 32-bit floating-point precision

- The inputs are supposed to be directly represented in TensorFloat-32 precision. The 32-bit floating-point values are truncated to TensorFloat-32 before the computation

- All computations and intermediate storage ensure at least TensorFloat-32 precision

- Tensor Cores will be used whenever possible

CUSPARSE_COMPUTE_TF32

- All computations and intermediate storage ensure at least TensorFloat-32 precision

- The inputs are rounded to TensorFloat-32 precision. This mode is slower than CUSPARSE_COMPUTE_TF32_FAST, but could provide more accurate results

- Tensor Cores will be used whenever possible

The compute precision is used in the cusparseLtMatmulDescriptorInit() function.
cusparseLtMatDescAttribute_t¶
The enumerator specifies the additional attributes of a matrix descriptor
Value

Description

CUSPARSELT_MAT_NUM_BATCHES

Number of matrices in a batch

CUSPARSELT_MAT_BATCH_STRIDE

Stride between consecutive matrices in a batch expressed in terms of matrix elements

The algorithm enumerator is used in the cusparseLtMatDescSetAttribute() and cusparseLtMatDescGetAttribute() functions.
cusparseLtMatmulDescAttribute_t¶
The enumerator specifies the additional attributes of a matrix multiplication descriptor
Value

Type

Default Value

Description

CUSPARSELT_MATMUL_ACTIVATION_RELU

int 0: false, true otherwise

false

ReLU activation function

CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND

float

inf

Upper bound of the ReLU activation function

CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD

float

0.0f

Lower threshold of the ReLU activation function

CUSPARSELT_MATMUL_ACTIVATION_GELU

int 0: false, true otherwise

false

Enable/Disable GeLU activation function

CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING

float

1.0f

Scaling coefficient for the GeLU activation function. It implies CUSPARSELT_MATMUL_ACTIVATION_GELU

CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING

int 0: false, true otherwise

false

Enable/Disable alpha vector (per-channel) scaling

CUSPARSELT_MATMUL_BETA_VECTOR_SCALING

int 0: false, true otherwise

false

Enable/Disable beta vector (per-channel) scaling

CUSPARSELT_MATMUL_BIAS_POINTER

void*

NULL (disabled)

Bias pointer. The bias vector size must equal to the number of rows of the output matrix (D)

CUSPARSELT_MATMUL_BIAS_STRIDE

int64_t

0 (disabled)

Bias stride between consecutive bias vectors. 0 means broadcast the first bias vector

where the ReLU activation function is defined as:

workflow
The GeLU activation function is available only with INT8 input/output, INT32 Tensor Core compute kernels

CUSPARSELT_MATMUL_BETA_VECTOR_SCALING implies CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING

The algorithm enumerator is used in the cusparseLtMatmulDescSetAttribute() and cusparseLtMatmulDescGetAttribute() functions.
cusparseLtMatmulAlg_t¶
The enumerator specifies the algorithm for matrix-matrix multiplication
Value

Description

CUSPARSELT_MATMUL_ALG_DEFAULT

Default algorithm

The algorithm enumerator is used in the cusparseLtMatmulAlgSelectionInit() function.
cusparseLtMatmulAlgAttribute_t¶
The enumerator specifies the matrix multiplication algorithm attributes
Value

Description

Possible Values

CUSPARSELT_MATMUL_ALG_CONFIG_ID

Algorithm ID

[0, MAX] (see CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID)

CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID

Algorithm ID limit (query only)

CUSPARSELT_MATMUL_SEARCH_ITERATIONS

Number of iterations (kernel launches per algorithm) for cusparseLtMatmulSearch()

> 0 (default=5)

CUSPARSELT_MATMUL_SPLIT_K

Split-K factor (number of slices)

[1, K], 1: Split-K disabled (default=not set)

CUSPARSELT_MATMUL_SPLIT_K_MODE

Number of kernels for the Split-K algorithm

CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL, CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS

CUSPARSELT_MATMUL_SPLIT_K_BUFFERS

Device memory buffers to store partial results for the reduction

[1, SplitK - 1]

The algorithm attribute enumerator is used in the cusparseLtMatmulAlgGetAttribute() and cusparseLtMatmulAlgSetAttribute() functions.
Split-K parameters allow users to split the GEMM computation along the K dimension so that more CTAs will be created with a better SM utilization when N or M dimensions are small. However, this comes with the cost of reducing the operation of K slides to the final results. The cusparseLtMatmulSearch() function can be used to find the optimal combination of Split-K parameters.
cusparseLtSplitKMode_t¶
The enumerator specifies the Split-K mode values corresponding to CUSPARSELT_MATMUL_SPLIT_K_MODE attribute in cusparseLtMatmulAlgAttribute_t
Value

Description

Details

CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL

Use a single kernel for Split-K

Use the same GEMM kernel to do the final reduction

CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS

Use two kernels for Split-K

Launch another GPU kernel to do the final reduction

cusparseLtPruneAlg_t¶
The enumerator specifies the pruning algorithm to apply to the structured matrix before the compression
Value

Description

CUSPARSELT_PRUNE_SPMMA_TILE

- half, bfloat16, int8: Zero-out eight values in a 4x4 tile to maximize the L1-norm of the resulting tile, under the constraint of selecting exactly two elements for each row and column 

- float, tf32: Zero-out two values in a 2x2 tile to maximize the L1-norm of the resulting tile, under the constraint of selecting exactly one element for each row and column

CUSPARSELT_PRUNE_SPMMA_STRIP

- half, bfloat16, int8: Zero-out two values in a 1x4 strip to maximize the L1-norm of the resulting strip 

- float, tf32: Zero-out one value in a 1x2 strip to maximize the L1-norm of the resulting strip 

The strip direction is chosen according to the operation op and matrix layout applied to the structured (sparse) matrix

The pruning algorithm is used in the cusparseLtSpMMAPrune() function.

Based on the information you provided, here's a simplified example of a matrix multiplication operation using cuSPARSELt. This example assumes that we have the necessary CUDA environment and cuSPARSELt library installed and correctly set up.

The code demonstrates basic steps of using cuSPARSELt API:

initializing and destroying cuSPARSELt handle
creating dense and sparse matrix descriptors
creating matmul descriptor
selecting matmul algorithm
creating matmul plan
performing the matmul operation
destroying created objects
cpp


cusparseLtHandle_t¶
The structure holds the cuSPARSELt library context (device properties, system information, etc.).
The handle must be initialized and destroyed with cusparseLtInit() and cusparseLtDestroy() functions respectively.
cusparseLtMatDescriptor_t¶
The structure captures the shape and characteristics of a matrix.
It is initialized with cusparseLtDenseDescriptorInit() or cusparseLtStructuredDescriptorInit() functions and destroyed with cusparseLtMatDescriptorDestroy().
cusparseLtMatmulDescriptor_t¶
The structure holds the description of the matrix multiplication operation.
It is initialized with cusparseLtMatmulDescriptorInit() function.
cusparseLtMatmulAlgSelection_t¶
The structure holds the description of the matrix multiplication algorithm.
It is initialized with cusparseLtMatmulAlgSelectionInit() function.
cusparseLtMatmulPlan_t¶
The structure holds the matrix multiplication execution plan, namely all the information necessary to execute the cusparseLtMatmul() operation.
It is initialized and destroyed with cusparseLtMatmulPlanInit() and cusparseLtMatmulPlanDestroy() functions respectively.
Enumerators¶
cusparseLtSparsity_t¶
The enumerator specifies the sparsity ratio of the structured matrix as
sparsity\ ratio = \frac{nnz}{num\_rows * num\_cols}

Value

Description

CUSPARSELT_SPARSITY_50_PERCENT

50% Sparsity Ratio:

- 2:4 for half, bfloat16, int

- 1:2 for tf32 and float

The sparsity property is used in the cusparseLtStructuredDescriptorInit() function.
cusparseComputeType¶
The enumerator specifies the compute precision modes of the matrix
Value

Description

CUSPARSE_COMPUTE_16F

- Default mode for 16-bit floating-point precision

- All computations and intermediate storage ensure at least 16-bit precision

- Tensor Cores will be used whenever possible

CUSPARSE_COMPUTE_32I

- Default mode for 32-bit integer precision

- All computations and intermediate storage ensure at least 32-bit integer precision

- Tensor Cores will be used whenever possible

CUSPARSE_COMPUTE_TF32_FAST

- Default mode for 32-bit floating-point precision

- The inputs are supposed to be directly represented in TensorFloat-32 precision. The 32-bit floating-point values are truncated to TensorFloat-32 before the computation

- All computations and intermediate storage ensure at least TensorFloat-32 precision

- Tensor Cores will be used whenever possible

CUSPARSE_COMPUTE_TF32

- All computations and intermediate storage ensure at least TensorFloat-32 precision

- The inputs are rounded to TensorFloat-32 precision. This mode is slower than CUSPARSE_COMPUTE_TF32_FAST, but could provide more accurate results

- Tensor Cores will be used whenever possible

The compute precision is used in the cusparseLtMatmulDescriptorInit() function.
cusparseLtMatDescAttribute_t¶
The enumerator specifies the additional attributes of a matrix descriptor
Value

Description

CUSPARSELT_MAT_NUM_BATCHES

Number of matrices in a batch

CUSPARSELT_MAT_BATCH_STRIDE

Stride between consecutive matrices in a batch expressed in terms of matrix elements

The algorithm enumerator is used in the cusparseLtMatDescSetAttribute() and cusparseLtMatDescGetAttribute() functions.
cusparseLtMatmulDescAttribute_t¶
The enumerator specifies the additional attributes of a matrix multiplication descriptor
Value

Type

Default Value

Description

CUSPARSELT_MATMUL_ACTIVATION_RELU

int 0: false, true otherwise

false

ReLU activation function

CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND

float

inf

Upper bound of the ReLU activation function

CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD

float

0.0f

Lower threshold of the ReLU activation function

CUSPARSELT_MATMUL_ACTIVATION_GELU

int 0: false, true otherwise

false

Enable/Disable GeLU activation function

CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING

float

1.0f

Scaling coefficient for the GeLU activation function. It implies CUSPARSELT_MATMUL_ACTIVATION_GELU

CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING

int 0: false, true otherwise

false

Enable/Disable alpha vector (per-channel) scaling

CUSPARSELT_MATMUL_BETA_VECTOR_SCALING

int 0: false, true otherwise

false

Enable/Disable beta vector (per-channel) scaling

CUSPARSELT_MATMUL_BIAS_POINTER

void*

NULL (disabled)

Bias pointer. The bias vector size must equal to the number of rows of the output matrix (D)

CUSPARSELT_MATMUL_BIAS_STRIDE

int64_t

0 (disabled)

Bias stride between consecutive bias vectors. 0 means broadcast the first bias vector

where the ReLU activation function is defined as:

workflow
The GeLU activation function is available only with INT8 input/output, INT32 Tensor Core compute kernels

CUSPARSELT_MATMUL_BETA_VECTOR_SCALING implies CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING

The algorithm enumerator is used in the cusparseLtMatmulDescSetAttribute() and cusparseLtMatmulDescGetAttribute() functions.
cusparseLtMatmulAlg_t¶
The enumerator specifies the algorithm for matrix-matrix multiplication
Value

Description

CUSPARSELT_MATMUL_ALG_DEFAULT

Default algorithm

The algorithm enumerator is used in the cusparseLtMatmulAlgSelectionInit() function.
cusparseLtMatmulAlgAttribute_t¶
The enumerator specifies the matrix multiplication algorithm attributes
Value

Description

Possible Values

CUSPARSELT_MATMUL_ALG_CONFIG_ID

Algorithm ID

[0, MAX] (see CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID)

CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID

Algorithm ID limit (query only)

CUSPARSELT_MATMUL_SEARCH_ITERATIONS

Number of iterations (kernel launches per algorithm) for cusparseLtMatmulSearch()

> 0 (default=5)

CUSPARSELT_MATMUL_SPLIT_K

Split-K factor (number of slices)

[1, K], 1: Split-K disabled (default=not set)

CUSPARSELT_MATMUL_SPLIT_K_MODE

Number of kernels for the Split-K algorithm

CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL, CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS

CUSPARSELT_MATMUL_SPLIT_K_BUFFERS

Device memory buffers to store partial results for the reduction

[1, SplitK - 1]

The algorithm attribute enumerator is used in the cusparseLtMatmulAlgGetAttribute() and cusparseLtMatmulAlgSetAttribute() functions.
Split-K parameters allow users to split the GEMM computation along the K dimension so that more CTAs will be created with a better SM utilization when N or M dimensions are small. However, this comes with the cost of reducing the operation of K slides to the final results. The cusparseLtMatmulSearch() function can be used to find the optimal combination of Split-K parameters.
cusparseLtSplitKMode_t¶
The enumerator specifies the Split-K mode values corresponding to CUSPARSELT_MATMUL_SPLIT_K_MODE attribute in cusparseLtMatmulAlgAttribute_t
Value

Description

Details

CUSPARSELT_SPLIT_K_MODE_ONE_KERNEL

Use a single kernel for Split-K

Use the same GEMM kernel to do the final reduction

CUSPARSELT_SPLIT_K_MODE_TWO_KERNELS

Use two kernels for Split-K

Launch another GPU kernel to do the final reduction

cusparseLtPruneAlg_t¶
The enumerator specifies the pruning algorithm to apply to the structured matrix before the compression
Value

Description

CUSPARSELT_PRUNE_SPMMA_TILE

- half, bfloat16, int8: Zero-out eight values in a 4x4 tile to maximize the L1-norm of the resulting tile, under the constraint of selecting exactly two elements for each row and column 

- float, tf32: Zero-out two values in a 2x2 tile to maximize the L1-norm of the resulting tile, under the constraint of selecting exactly one element for each row and column

CUSPARSELT_PRUNE_SPMMA_STRIP

- half, bfloat16, int8: Zero-out two values in a 1x4 strip to maximize the L1-norm of the resulting strip 

- float, tf32: Zero-out one value in a 1x2 strip to maximize the L1-norm of the resulting strip 

The strip direction is chosen according to the operation op and matrix layout applied to the structured (sparse) matrix

The pruning algorithm is used in the cusparseLtSpMMAPrune() function.
