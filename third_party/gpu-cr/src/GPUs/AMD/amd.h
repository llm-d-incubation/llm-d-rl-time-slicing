#ifndef AMD_H
#define AMD_H

#include "../GPU.h"
#include <hip/hip_runtime.h>
#include <map>

class amd : public GPU {
private:
    std::map<void*, size_t> memory_map_;
    int device_;  // Device ID
    bool hip_initialized_;
    
    void ensureHipInitialized();
    
public:
    amd();
    ~amd() override;
    
    // memory management implementation
    int allocate(void** ptr, size_t size) override;
    int deallocate(void* ptr) override;
    std::map<void*, size_t>& getMemoryMap() override;
    
    // synchronization implementation
    int createStream(GPUStream* stream) override;
    int destroyStream(GPUStream stream) override;
    int createEvent(GPUEvent* event) override;
    int destroyEvent(GPUEvent event) override;
    int recordEvent(GPUEvent event, GPUStream stream) override;
    int synchronizeEvent(GPUEvent event) override;
    int memcpyAsync(void* dst, const void* src, size_t size, 
                   GPUMemcpyKind kind, GPUStream stream) override;
    int synchronizeStream(GPUStream stream) override;
    int syncAllKernels() override;
    int registerHostMemory(void* ptr, size_t size) override;
    
    // Part 3: external tools
    int externalCheckpoint(int pid) override;
    int externalRestore(int pid) override;
    
    // Memory release/remap
    int releasePhysicalMemory(void* ptr) override;
    int remapPhysicalMemory(void* ptr, size_t size) override;
    
    GPUVendor getVendor() const override { return GPUVendor::AMD; }
    std::string getVendorName() const override { return "AMD"; }
};

// hook functions implementation
extern "C" {
    hipError_t hipMalloc(void **devPtr, size_t size);
    hipError_t hipFree(void *devPtr);
}

#endif // AMD_H