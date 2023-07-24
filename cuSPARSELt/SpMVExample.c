#include <stdio.h>
#include <cusparseLt.h>

int main(void) {
    cusparseLtHandle_t handle;
    cusparseLtInit(&handle);

    // Example matrix and vector sizes
    int m = 10, n = 10;
    int* csrRowPtr = NULL; // Row pointers for CSR format, size = m + 1
    int* csrColInd = NULL; // Column indices for CSR format
    float* csrVal = NULL;  // Nonzero values for CSR format
    float* x = NULL;       // Vector to multiply with, size = n
    float* y = NULL;       // Result vector, size = m

    // Fill csrRowPtr, csrColInd, csrVal, and x with your data...

    // Allocate result vector y on the device
    cudaMallocManaged(&y, m * sizeof(float));

    // Setup matrix descriptor
    cusparseLtMatDescriptor_t matDescr;
    cusparseLtMatDescriptorInit(&matDescr);
    cusparseLtMatDescriptorSetAttribute(&matDescr, CUSPARSELT_MAT_ATTR_TYPE, CUSPARSELT_MAT_TYPE_CSR);

    // Create matrix
    cusparseLtSpMat_t matA;
    cusparseLtSpMatInit(&matA, &matDescr, csrRowPtr, csrColInd, csrVal, m, n);

    // Setup operation descriptor
    cusparseLtSpMVDescriptor_t opDescr;
    cusparseLtSpMVDescriptorInit(&opDescr, CUSPARSELT_SPMV_TYPE_CSR, CUSPARSELT_OPERATION_NON_TRANSPOSE);

    // Perform SpMV operation
    cusparseLtSpMV(handle, &opDescr, &matA, x, y, CUSPARSELT_HINT_ACCURACY);

    // Free memory and destroy handle
    cudaFree(y);
    cusparseLtDestroy(handle);

    return 0;
}
