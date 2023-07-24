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


cusparseLtInit¶
cusparseStatus_t
cusparseLtInit(cusparseLtHandle_t* handle)
The function initializes the cuSPARSELt library handle (cusparseLtHandle_t) which holds the cuSPARSELt library context. It allocates light hardware resources on the host, and must be called prior to making any other cuSPARSELt library calls. Calling any cusparseLt function which uses cusparseLtHandle_t without a previous call of cusparseLtInit() will return an error.
The cuSPARSELt library context is tied to the current CUDA device. To use the library on multiple devices, one cuSPARSELt handle should be created for each device.
Parameter

Memory

In/Out

Description

handle

Host

OUT

cuSPARSELt library handle

See cusparseStatus_t for the description of the return status.

cusparseLtDestroy¶
cusparseStatus_t
cusparseLtDestroy(const cusparseLtHandle_t* handle)
The function releases hardware resources used by the cuSPARSELt library. This function is the last call with a particular handle to the cuSPARSELt library.
Calling any cusparseLt function which uses cusparseLtHandle_t after cusparseLtDestroy() will return an error.
Parameter

Memory

In/Out

Description

handle

Host

IN

cuSPARSELt library handle

See cusparseStatus_t for the description of the return status.

cusparseLtGetVersion¶
cusparseStatus_t
cusparseLtGetVersion(const cusparseLtHandle_t* handle,
                     int*                      version)
This function returns the version number of the cuSPARSELt library.
Parameter

Memory

In/Out

Description

handle

Host

IN

cuSPARSELt library handle

version

Host

OUT

The version number of the library

See cusparseStatus_t for the description of the return status.

cusparseLtGetProperty¶
cusparseStatus_t
cusparseLtGetProperty(libraryPropertyType propertyType,
                      int*                value)
The function returns the value of the requested property. Refer to libraryPropertyType for supported types.
Parameter

Memory

In/Out

Description

propertyType

Host

IN

Requested property

value

Host

OUT

Value of the requested property

libraryPropertyType (defined in library_types.h):

Value

Meaning

MAJOR_VERSION

Enumerator to query the major version

MINOR_VERSION

Enumerator to query the minor version

PATCH_LEVEL

Number to identify the patch level

See cusparseStatus_t for the description of the return status.

Matrix Descriptor Functions¶
cusparseLtDenseDescriptorInit¶
cusparseStatus_t
cusparseLtDenseDescriptorInit(const cusparseLtHandle_t*  handle,
                              cusparseLtMatDescriptor_t* matDescr,
                              int64_t                    rows,
                              int64_t                    cols,
                              int64_t                    ld,
                              uint32_t                   alignment,
                              cudaDataType               valueType,
                              cusparseOrder_t            order)
The function initializes the descriptor of a dense matrix.

Parameter

Memory

In/Out

Description

Possible Values

handle

Host

IN

cuSPARSELt library handle

matDescr

Host

OUT

Dense matrix description

rows

Host

IN

Number of rows

cols

Host

IN

Number of columns

ld

Host

IN

Leading dimension

≥ rows if column-major, ≥ cols if row-major

alignment

Host

IN

Memory alignment in bytes

Multiple of 16

valueType

Host

IN

Data type of the matrix

CUDA_R_16F, CUDA_R_16BF, CUDA_R_I8, CUDA_R_32F

order

Host

IN

Memory layout

CUSPARSE_ORDER_COL, CUSPARSE_ORDER_ROW

Constrains:

rows, cols, and ld must be a multiple of

16 if valueType is CUDA_R_I8

8 if valueType is CUDA_R_16F or CUDA_R_16BF

4 if valueType is CUDA_R_32F

The total size of the matrix cannot exceed:

2^{32}-1 elements for CUDA_R_8I

2^{31}-1 elements for CUDA_R_16F or CUDA_R_16BF

2^{30}-1 elements for CUDA_R_32F

See cusparseStatus_t for the description of the return status.

cusparseLtStructuredDescriptorInit¶
cusparseStatus_t
cusparseLtStructuredDescriptorInit(const cusparseLtHandle_t*  handle,
                                   cusparseLtMatDescriptor_t* matDescr,
                                   int64_t                    rows,
                                   int64_t                    cols,
                                   int64_t                    ld,
                                   uint32_t                   alignment,
                                   cudaDataType               valueType,
                                   cusparseOrder_t            order,
                                   cusparseLtSparsity_t       sparsity)
The function initializes the descriptor of a structured matrix.

Parameter

Memory

In/Out

Description

Possible Values

handle

Host

IN

cuSPARSELt library handle

matDescr

Host

OUT

Dense matrix description

rows

Host

IN

Number of rows

cols

Host

IN

Number of columns

ld

Host

IN

Leading dimension

≥ rows if column-major, ≥ cols if row-major

alignment

Host

IN

Memory alignment in bytes

Multiple of 16

valueType

Host

IN

Data type of the matrix

CUDA_R_16F, CUDA_R_16BF, CUDA_R_I8, CUDA_R_32F

order

Host

IN

Memory layout

CUSPARSE_ORDER_COL, CUSPARSE_ORDER_ROW

sparsity

Host

IN

Matrix sparsity ratio

CUSPARSELT_SPARSITY_50_PERCENT

Constrains:

rows, cols, and ld must be a multiple of

32 if valueType is CUDA_R_I8

16 if valueType is CUDA_R_16F, or CUDA_R_16BF

8 if valueType is CUDA_R_32F

The total size of the matrix cannot exceed:

2^{32}-1 elements for CUDA_R_8I

2^{31}-1 elements for CUDA_R_16F or CUDA_R_16BF

2^{30}-1 elements for CUDA_R_32F

See cusparseStatus_t for the description of the return status.

cusparseLtMatDescriptorDestroy¶
cusparseStatus_t
cusparseLtMatDescriptorDestroy(const cusparseLtMatDescriptor_t* matDescr)
The function releases the resources used by an instance of a matrix descriptor. After this call, the matrix descriptor, the matmul descriptor, and the plan can no longer be used.
Parameter

Memory

In/Out

Description

matDescr

Host

IN

Matrix descriptor

See cusparseStatus_t for the description of the return status.

cusparseLtMatDescSetAttribute¶
cusparseStatus_t
cusparseLtMatDescSetAttribute(const cusparseLtHandle_t*    handle,
                              cusparseLtMatDescriptor_t*   matmulDescr,
                              cusparseLtMatDescAttribute_t matAttribute,
                              const void*                  data,
                              size_t                       dataSize)
The function sets the value of the specified attribute belonging to matrix descriptor such as number of batches and their stride.
Parameter

Memory

In/Out

Description

Possible Values

handle

Host

IN

cuSPARSELt library handle

matmulDescr

Host

OUT

Matrix descriptor

matAttribute

Host

IN

Attribute to set

CUSPARSELT_MAT_NUM_BATCHES, CUSPARSELT_MAT_BATCH_STRIDE

data

Host

IN

Pointer to the value to which the specified attribute will be set

dataSize

Host

IN

Size in bytes of the attribute value used for verification

See cusparseStatus_t for the description of the return status.

cusparseLtMatDescGetAttribute¶
cusparseStatus_t
cusparseLtMatDescGetAttribute(const cusparseLtHandle_t*        handle,
                              const cusparseLtMatDescriptor_t* matmulDescr,
                              cusparseLtMatDescAttribute_t     matAttribute,
                              void*                            data,
                              size_t                           dataSize)
The function gets the value of the specified attribute belonging to matrix descriptor such as number of batches and their stride.
Parameter

Memory

In/Out

Description

Possible Values

handle

Host

IN

cuSPARSELt library handle

matmulDescr

Host

IN

Matrix descriptor

matAttribute

Host

IN

Attribute to retrieve

CUSPARSELT_MAT_NUM_BATCHES, CUSPARSELT_MAT_BATCH_STRIDE

data

Host

OUT

Memory address containing the attribute value retrieved by this function

dataSize

Host

IN

Size in bytes of the attribute value used for verification

See cusparseStatus_t for the description of the return status.

Matmul Descriptor Functions¶
cusparseLtMatmulDescriptorInit¶
cusparseStatus_t
cusparseLtMatmulDescriptorInit(const cusparseLtHandle_t*        handle,
                               cusparseLtMatmulDescriptor_t*    matmulDescr,
                               cusparseOperation_t              opA,
                               cusparseOperation_t              opB,
                               const cusparseLtMatDescriptor_t* matA,
                               const cusparseLtMatDescriptor_t* matB,
                               const cusparseLtMatDescriptor_t* matC,
                               const cusparseLtMatDescriptor_t* matD,
                               cusparseComputeType              computeType)
The function initializes the matrix multiplication descriptor.

Parameter

Memory

In/Out

Description

Possible Values

handle

Host

IN

cuSPARSELt library handle

matmulDescr

Host

OUT

Matrix multiplication descriptor

opA

Host

IN

Operation applied to the matrix A

CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE

opB

Host

IN

Operation applied to the matrix B

CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE

matA

Host

IN

Structured or dense matrix descriptor A

matB

Host

IN

Structured or dense matrix descriptor B

matC

Host

IN

Dense matrix descriptor C

matD

Host

IN

Dense matrix descriptor D

computeType

Host

IN

Compute precision

CUSPARSE_COMPUTE_16F, CUSPARSE_COMPUTE_32I, CUSPARSE_COMPUTE_TF32_FAST, CUSPARSE_COMPUTE_TF32

The structured matrix descriptor can used for matA or matB but not both.

Data types Supported:

Input

Output

Compute

CUDA_R_16F

CUDA_R_16F

CUSPARSE_COMPUTE_16F

CUDA_R_16BF

CUDA_R_16BF

CUSPARSE_COMPUTE_16F

CUDA_R_8I

CUDA_R_8I

CUSPARSE_COMPUTE_32I

CUDA_R_8I

CUDA_R_16F

CUSPARSE_COMPUTE_32I

CUDA_R_32F

CUDA_R_32F

CUSPARSE_COMPUTE_TF32_FAST

CUDA_R_32F

CUDA_R_32F

CUSPARSE_COMPUTE_TF32

Constrains:

CUDA_R_8I data type only supports (the opposite if B is structured):

opA/opB = TN if the matrix orders are orderA/orderB = Col/Col

opA/opB = NT if the matrix orders are orderA/orderB = Row/Row

opA/opB = NN if the matrix orders are orderA/orderB = Row/Col

opA/opB = TT if the matrix orders are orderA/orderB = Col/Row

See cusparseStatus_t for the description of the return status.

cusparseLtMatmulDescSetAttribute¶
cusparseStatus_t
cusparseLtMatmulDescSetAttribute(const cusparseLtHandle_t*       handle,
                                 cusparseLtMatmulDescriptor_t*   matmulDescr,
                                 cusparseLtMatmulDescAttribute_t matmulAttribute,
                                 const void*                     data,
                                 size_t                          dataSize)
The function sets the value of the specified attribute belonging to matrix descriptor such as activation function and bias.
Parameter

Memory

In/Out

Description

handle

Host

IN

cuSPARSELt library handle

matmulDescr

Host

OUT

Matrix descriptor

matmulAttribute

Host

IN

Attribute to set

CUSPARSELT_MATMUL_ACTIVATION_RELU, CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND, CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD, CUSPARSELT_MATMUL_ACTIVATION_GELU, CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING, CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, CUSPARSELT_MATMUL_BETA_VECTOR_SCALING, CUSPARSELT_MATMUL_BIAS_POINTER, CUSPARSELT_MATMUL_BIAS_STRIDE

data

Host

IN

Pointer to the value to which the specified attribute will be set

dataSize

Host

IN

Size in bytes of the attribute value used for verification

See cusparseStatus_t for the description of the return status.

cusparseLtMatmulDescGetAttribute¶
cusparseStatus_t
cusparseLtMatmulDescGetAttribute(const cusparseLtHandle_t*           handle,
                                 const cusparseLtMatmulDescriptor_t* matmulDescr,
                                 cusparseLtMatmulDescAttribute_t     matmulAttribute,
                                 void*                               data,
                                 size_t                              dataSize)
The function gets the value of the specified attribute belonging to matrix descriptor such as activation function and bias.
Parameter

Memory

In/Out

Description

handle

Host

IN

cuSPARSELt library handle

matmulDescr

Host

IN

Matrix descriptor

matmulAttribute

Host

IN

Attribute to retrieve

CUSPARSELT_MATMUL_ACTIVATION_RELU, CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND, CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD, CUSPARSELT_MATMUL_ACTIVATION_GELU, CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING, CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, CUSPARSELT_MATMUL_BETA_VECTOR_SCALING, CUSPARSELT_MATMUL_BIAS_POINTER, CUSPARSELT_MATMUL_BIAS_STRIDE

data

Host

OUT

Memory address containing the attribute value retrieved by this function

dataSize

Host

IN

Size in bytes of the attribute value used for verification

See cusparseStatus_t for the description of the return status.

Matmul Algorithm Functions¶
cusparseLtMatmulAlgSelectionInit¶
cusparseStatus_t
cusparseLtMatmulAlgSelectionInit(const cusparseLtHandle_t*           handle,
                                 cusparseLtMatmulAlgSelection_t*     algSelection,
                                 const cusparseLtMatmulDescriptor_t* matmulDescr,
                                 cusparseLtMatmulAlg_t               alg)
The function initializes the algorithm selection descriptor.

Parameter

Memory

In/Out

Description

Possible Values

handle

Host

IN

cuSPARSELt library handle

algSelection

Host

OUT

Algorithm selection descriptor

matmulDescr

Host

IN

Matrix multiplication descriptor

alg

Host

IN

Algorithm mode

CUSPARSELT_MATMUL_ALG_DEFAULT

See cusparseStatus_t for the description of the return status.

cusparseLtMatmulAlgSetAttribute¶
cusparseStatus_t
cusparseLtMatmulAlgSetAttribute(const cusparseLtHandle_t*       handle,
                                cusparseLtMatmulAlgSelection_t* algSelection,
                                cusparseLtMatmulAlgAttribute_t  attribute,
                                const void*                     data,
                                size_t                          dataSize)
The function sets the value of the specified attribute belonging to algorithm selection descriptor.

Parameter

Memory

In/Out

Description

Possible Values

handle

Host

IN

cuSPARSELt library handle

algSelection

Host

OUT

Algorithm selection descriptor

attribute

Host

IN

The attribute to set

CUSPARSELT_MATMUL_ALG_CONFIG_ID, CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, CUSPARSELT_MATMUL_SEARCH_ITERATIONS, CUSPARSELT_MATMUL_SPLIT_K, CUSPARSELT_MATMUL_SPLIT_K_MODE, CUSPARSELT_MATMUL_SPLIT_K_BUFFERS

data

Host

IN

Pointer to the value to which the specified attribute will be set

dataSize

Host

IN

Size in bytes of the attribute value used for verification

See cusparseStatus_t for the description of the return status.

cusparseLtMatmulAlgGetAttribute¶
cusparseStatus_t
cusparseLtMatmulAlgGetAttribute(const cusparseLtHandle_t*             handle,
                                const cusparseLtMatmulAlgSelection_t* algSelection,
                                cusparseLtMatmulAlgAttribute_t        attribute,
                                void*                                 data,
                                size_t                                dataSize)
The function returns the value of the queried attribute belonging to algorithm selection descriptor.

Parameter

Memory

In/Out

Description

Possible Values

handle

Host

IN

cuSPARSELt library handle

algSelection

Host

IN

Algorithm selection descriptor

attribute

Host

IN

The attribute that will be retrieved by this function

CUSPARSELT_MATMUL_ALG_CONFIG_ID, CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, CUSPARSELT_MATMUL_SEARCH_ITERATIONS, CUSPARSELT_MATMUL_SPLIT_K, CUSPARSELT_MATMUL_SPLIT_K_MODE, CUSPARSELT_MATMUL_SPLIT_K_BUFFERS

data

Host

OUT

Memory address containing the attribute value retrieved by this function

dataSize

Host

IN

Size in bytes of the attribute value used for verification

See cusparseStatus_t for the description of the return status.

Matmul Functions¶
cusparseLtMatmulGetWorkspace¶
cusparseStatus_t
cusparseLtMatmulGetWorkspace(const cusparseLtHandle_t*     handle,
                             const cusparseLtMatmulPlan_t* plan,
                             size_t*                       workspaceSize)
The function determines the required workspace size associated to the selected algorithm.

Parameter

Memory

In/Out

Description

handle

Host

IN

cuSPARSELt library handle

plan

Host

IN

Matrix multiplication plan

workspaceSize

Host

OUT

Workspace size in bytes

See cusparseStatus_t for the description of the return status.

cusparseLtMatmulPlanInit¶
cusparseStatus_t
cusparseLtMatmulPlanInit(const cusparseLtHandle_t*             handle,
                         cusparseLtMatmulPlan_t*               plan,
                         const cusparseLtMatmulDescriptor_t*   matmulDescr,
                         const cusparseLtMatmulAlgSelection_t* algSelection)
Parameter

Memory

In/Out

Description

handle

Host

IN

cuSPARSELt library handle

plan

Host

OUT

Matrix multiplication plan

matmulDescr

Host

IN

Matrix multiplication descriptor

algSelection

Host

IN

Algorithm selection descriptor

See cusparseStatus_t for the description of the return status.

cusparseLtMatmulPlanDestroy¶
cusparseStatus_t
cusparseLtMatmulPlanDestroy(const cusparseLtMatmulPlan_t* plan)
The function releases the resources used by an instance of the matrix multiplication plan. This function is the last call with a specific plan instance.
Calling any cusparseLt function which uses cusparseLtMatmulPlan_t after cusparseLtMatmulPlanDestroy() will return an error.
Parameter

Memory

In/Out

Description

plan

Host

IN

Matrix multiplication plan

See cusparseStatus_t for the description of the return status.

cusparseLtMatmul¶
cusparseStatus_t
cusparseLtMatmul(const cusparseLtHandle_t*     handle,
                 const cusparseLtMatmulPlan_t* plan,
                 const void*                   alpha,
                 const void*                   d_A,
                 const void*                   d_B,
                 const void*                   beta,
                 const void*                   d_C,
                 void*                         d_D,
                 void*                         workspace,
                 cudaStream_t*                 streams,
                 int32_t                       numStreams)
The function computes the matrix multiplication of matrices A and B to produce the the output matrix D, according to the following operation:

D = Activation(\alpha op(A) \cdot op(B) + \beta op(C) + bias) \cdot scale

where A, B, and C are input matrices, and \alpha and \beta are input scalars or vectors of scalars (device-side pointers).
Note: The function currently only supports the case where D has the same shape of C
Parameter

Memory

In/Out

Description

handle

Host

IN

cuSPARSELt library handle

plan

Host

IN

Matrix multiplication plan

alpha

Host

IN

\alpha scalar/vector of scalars used for multiplication (float data type)

d_A

Device

IN

Pointer to the structured or dense matrix A

d_B

Device

IN

Pointer to the structured or dense matrix B

beta

Host

IN

\beta scalar/vector of scalars used for multiplication (float data type). It can have a NULL value only if CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING is set without CUSPARSELT_MATMUL_BETA_VECTOR_SCALING

d_C

Device

OUT

Pointer to the dense matrix C

d_D

Device

OUT

Pointer to the dense matrix D

workspace

Device

IN

Pointer to workspace

streams

Host

IN

Pointer to CUDA stream array for the computation

numStreams

Host

IN

Number of CUDA streams in streams

Data types Supported:

Input

Output

Compute

CUDA_R_16F

CUDA_R_16F

CUSPARSE_COMPUTE_16F

CUDA_R_16BF

CUDA_R_16BF

CUSPARSE_COMPUTE_16F

CUDA_R_8I

CUDA_R_8I

CUSPARSE_COMPUTE_32I

CUDA_R_8I

CUDA_R_16F

CUSPARSE_COMPUTE_32I

CUDA_R_32F

CUDA_R_32F

CUSPARSE_COMPUTE_TF32_FAST

CUDA_R_32F

CUDA_R_32F

CUSPARSE_COMPUTE_TF32

CUSPARSE_COMPUTE_TF32 kernels perform the conversion from 32-bit IEEE754 floating-point to TensorFloat-32 by applying round toward plus infinity rounding mode before the computation.

CUSPARSE_COMPUTE_TF32_FAST kernels suppose that the data are already represented in TensorFloat-32 (32-bit per value). If 32-bit IEEE754 floating-point are used as input, the values are truncated to TensorFloat-32 before the computation.

CUSPARSE_COMPUTE_TF32_FAST kernels provide better performance than CUSPARSE_COMPUTE_TF32 but could produce less accurate results.

The structured matrix A or B (before the compression) must respect the following constrains depending on the operation applied on it:

For op = CUSPARSE_NON_TRANSPOSE

CUDA_R_16F, CUDA_R_16BF, CUDA_R_8I each row must have at least two non-zero values every four elements

CUDA_R_32F each row must have at least one non-zero value every two elements

For op = CUSPARSE_TRANSPOSE

CUDA_R_16F, CUDA_R_16BF, CUDA_R_8I each column must have at least two non-zero values every four elements

CUDA_R_32F each column must have at least one non-zero value every two elements

int8 kernels should run at high SM clocks for maximizing the performance.

The correctness of the pruning result (matrix A/B) can be check with the function cusparseLtSpMMAPruneCheck().

Constrains:

All pointers must be aligned to 16 bytes

Properties

The routine requires no extra storage

The routine supports asynchronous execution with respect to streams[0]

Provides deterministic (bit-wise) results for each run

cusparseLtMatmul supports the following optimizations:

CUDA graph capture

Hardware Memory Compression

See cusparseStatus_t for the description of the return status.

cusparseLtMatmulSearch¶
cusparseStatus_t
cusparseLtMatmulSearch(const cusparseLtHandle_t* handle,
                       cusparseLtMatmulPlan_t*   plan,
                       const void*               alpha,
                       const void*               d_A,
                       const void*               d_B,
                       const void*               beta,
                       const void*               d_C,
                       void*                     d_D,
                       void*                     workspace,
                       cudaStream_t*             streams,
                       int32_t                   numStreams)
The function evaluates all available algorithms for the matrix multiplication and automatically updates the plan by selecting the fastest one. The functionality is intended to be used for auto-tuning purposes when the same operation is repeated multiple times over different inputs.
The function behavior is the same of cusparseLtMatmul().
The function is NOT asynchronous with respect to streams[0] (blocking call)

The number of iterations for the evaluation can be set by using cusparseLtMatmulAlgSetAttribute() with CUSPARSELT_MATMUL_SEARCH_ITERATIONS.

The selected algorithm id can be retrieved by using cusparseLtMatmulAlgGetAttribute() with CUSPARSELT_MATMUL_ALG_CONFIG_ID.

The function also searches for optimal combination of Split-K parameters if the value of the attribute is not already set (e.g. CUSPARSELT_MATMUL_SPLIT_K = 1). The selected values can be retrieved by using cusparseLtMatmulAlgGetAttribute().

Helper Functions¶
cusparseLtSpMMAPrune¶
cusparseStatus_t
cusparseLtSpMMAPrune(const cusparseLtHandle_t*           handle,
                     const cusparseLtMatmulDescriptor_t* matmulDescr,
                     const void*                         d_in,
                     void*                               d_out,
                     cusparseLtPruneAlg_t                pruneAlg,
                     cudaStream_t                        stream)
The function prunes a dense matrix d_in according to the specified algorithm pruneAlg.

Parameter

Memory

In/Out

Description

Possible Values

handle

Host

IN

cuSPARSELt library handle

matmulDescr

Host

IN

Matrix multiplication descriptor

d_in

Device

IN

Pointer to the dense matrix

d_out

Device

OUT

Pointer to the pruned matrix

pruneAlg

Device

IN

Pruning algorithm

CUSPARSELT_PRUNE_SPMMA_TILE, CUSPARSELT_PRUNE_SPMMA_STRIP

stream

Host

IN

CUDA stream for the computation

Properties

The routine requires no extra storage

The routine supports asynchronous execution with respect to stream

Provides deterministic (bit-wise) results for each run

cusparseLtSpMMAPrune() supports the following optimizations:

CUDA graph capture

Hardware Memory Compression

See cusparseStatus_t for the description of the return status.

cusparseLtSpMMAPrune2¶
cusparseStatus_t
cusparseLtSpMMAPrune2(const cusparseLtHandle_t*        handle,
                      const cusparseLtMatDescriptor_t* sparseMatDescr,
                      int                              isSparseA,
                      cusparseOperation_t              op,
                      const void*                      d_in,
                      void*                            d_out,
                      cusparseLtPruneAlg_t             pruneAlg,
                      cudaStream_t                     stream);
The function prunes a dense matrix d_in according to the specified algorithm pruneAlg.

Parameter

Memory

In/Out

Description

Possible Values

handle

Host

IN

cuSPARSELt library handle

sparseMatDescr

Host

IN

Structured (sparse) matrix descriptor

isSparseA

Host

IN

Specify if the structured (sparse) matrix is in the first position (matA or matB)

0 false, true otherwise

op

Host

IN

Operation that will be applied to the structured (sparse) matrix in the multiplication

CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE

d_in

Device

IN

Pointer to the dense matrix

d_out

Device

OUT

Pointer to the pruned matrix

pruneAlg

Device

IN

Pruning algorithm

CUSPARSELT_PRUNE_SPMMA_TILE, CUSPARSELT_PRUNE_SPMMA_STRIP

stream

Host

IN

CUDA stream for the computation

If CUSPARSELT_PRUNE_SPMMA_TILE is used, isSparseA and op are not relevant.

The function has the same properties of cusparseLtSpMMAPrune()

cusparseLtSpMMAPruneCheck¶
cusparseStatus_t
cusparseLtSpMMAPruneCheck(const cusparseLtHandle_t*           handle,
                          const cusparseLtMatmulDescriptor_t* matmulDescr,
                          const void*                         d_in,
                          int*                                d_valid,
                          cudaStream_t                        stream)
The function checks the correctness of the pruning structure for a given matrix.

Parameter

Memory

In/Out

Description

handle

Host

IN

cuSPARSELt library handle

matmulDescr

Host

IN

Matrix multiplication descriptor

d_in

Device

IN

Pointer to the matrix to check

d_valid

Device

OUT

Validation results (0 correct, 1 wrong)

stream

Host

IN

CUDA stream for the computation

See cusparseStatus_t for the description of the return status.

cusparseLtSpMMAPruneCheck2¶
cusparseStatus_t
cusparseLtSpMMAPruneCheck2(const cusparseLtHandle_t*        handle,
                           const cusparseLtMatDescriptor_t* sparseMatDescr,
                           int                              isSparseA,
                           cusparseOperation_t              op,
                           const void*                      d_in,
                           int*                             d_valid,
                           cudaStream_t                     stream)
The function checks the correctness of the pruning structure for a given matrix.

Parameter

Memory

In/Out

Description

Possible Values

handle

Host

IN

cuSPARSELt library handle

sparseMatDescr

Host

IN

Structured (sparse) matrix descriptor

isSparseA

Host

IN

Specify if the structured (sparse) matrix is in the first position (matA or matB)

0: false, != 0: true

op

Host

IN

Operation that will be applied to the structured (sparse) matrix in the multiplication

CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE

d_in

Device

IN

Pointer to the matrix to check

d_valid

Device

OUT

Validation results (0 correct, 1 wrong)

stream

Host

IN

CUDA stream for the computation

The function has the same properties of cusparseLtSpMMAPruneCheck()

cusparseLtSpMMACompressedSize¶
cusparseStatus_t
cusparseLtSpMMACompressedSize(const cusparseLtHandle_t*     handle,
                              const cusparseLtMatmulPlan_t* plan,
                              size_t*                       compressedSize,
                              size_t*                       compressBufferSize)
The function provides the size of the compressed matrix to be allocated before calling cusparseLtSpMMACompress().

Parameter

Memory

In/Out

Description

handle

Host

IN

cuSPARSELt library handle

plan

Host

IN

Matrix plan descriptor

compressedSize

Host

OUT

Size in bytes for the compressed matrix

compressBufferSize

Host

OUT

Size in bytes for the buffer needed for the matrix compression

See cusparseStatus_t for the description of the return status.

cusparseLtSpMMACompressedSize2¶
cusparseStatus_t
cusparseLtSpMMACompressedSize2(const cusparseLtHandle_t*        handle,
                               const cusparseLtMatDescriptor_t* sparseMatDescr,
                               size_t*                          compressedSize,
                               size_t*                          compressBufferSize)
The function provides the size of the compressed matrix to be allocated before calling cusparseLtSpMMACompress2().

Parameter

Memory

In/Out

Description

handle

Host

IN

cuSPARSELt library handle

sparseMatDescr

Host

IN

Structured (sparse) matrix descriptor

compressedSize

Host

OUT

Size in bytes of the compressed matrix

compressBufferSize

Host

OUT

Size in bytes for the buffer needed for the matrix compression

The function has the same properties of cusparseLtSpMMACompressedSize()

cusparseLtSpMMACompress¶
cusparseStatus_t
cusparseLtSpMMACompress(const cusparseLtHandle_t*     handle,
                        const cusparseLtMatmulPlan_t* plan,
                        const void*                   d_dense,
                        void*                         d_compressed,
                        void*                         d_compressed_buffer,
                        cudaStream_t                  stream)
The function compresses a dense matrix d_dense. The compressed matrix is intended to be used as the first/second operand A/B in the cusparseLtMatmul() function.

Parameter

Memory

In/Out

Description

handle

Host

IN

cuSPARSELt library handle

plan

Host

IN

Matrix multiplication plan

d_dense

Device

IN

Pointer to the dense matrix

d_compressed

Device

OUT

Pointer to the compressed matrix

d_compressed_buffer

Device

OUT

Pointer to temporary buffer for the compression

stream

Host

IN

CUDA stream for the computation

Properties

The routine supports asynchronous execution with respect to stream

Provides deterministic (bit-wise) results for each run

cusparseLtSpMMACompress() supports the following optimizations:

CUDA graph capture

Hardware Memory Compression

See cusparseStatus_t for the description of the return status.

cusparseLtSpMMACompress2¶
cusparseStatus_t
cusparseLtSpMMACompress2(const cusparseLtHandle_t*        handle,
                         const cusparseLtMatDescriptor_t* sparseMatDescr,
                         int                              isSparseA,
                         cusparseOperation_t              op,
                         const void*                      d_dense,
                         void*                            d_compressed,
                         void*                            d_compressed_buffer,
                         cudaStream_t                     stream)
Parameter

Memory

In/Out

Description

Possible Values

handle

Host

IN

cuSPARSELt library handle

sparseMatDescr

Host

IN

Structured (sparse) matrix descriptor

isSparseA

Host

IN

Specify if the structured (sparse) matrix is in the first position (matA or matB)

0 false, true otherwise

op

Host

IN

Operation that will be applied to the structured (sparse) matrix in the multiplication

CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE

d_dense

Device

IN

Pointer to the dense matrix

d_compressed

Device

OUT

Pointer to the compressed matrix

d_compressed_buffer

Device

OUT

Pointer to temporary buffer for the compression

stream

Host

IN

CUDA stream for the computation

The function has the same properties of cusparseLtSpMMACompress()
