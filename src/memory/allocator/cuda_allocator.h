#pragma once
#include <unordered_map>
#include <map>
#include <vector>
#include <iostream>
#include "src/memory/allocator/base_allocator.h"
#include "src/utils/macro.h"

// I use Bytes to printf buffer size msg, because sometime I allocate <1KB buffer, which causes that display 0KB
struct CudaBigBlock {
    void *data;
    size_t size;
    bool is_allocated;

    CudaBigBlock() = default;
    CudaBigBlock(void* data_, int size_, bool is_allocated_):
        data(data_),
        size(size_),
        is_allocated(is_allocated_){}
};

struct CudaSmallBlock {
    void *data;
    size_t size;
    bool is_allocated;

    CudaSmallBlock() = default;
    CudaSmallBlock(void* data_, int size_, bool is_allocated_):
        data(data_),
        size(size_),
        is_allocated(is_allocated_){}
};

class CudaAllocator: public BaseAllocator {
private:
    //{device id: block}
    std::map<int, std::vector<CudaSmallBlock> > cudaSmallBlocksMap;
    std::map<int, std::vector<CudaBigBlock> > CudaBigBlocksMap;
    std::map<int, size_t> FreeSize;
    size_t total_allocated_size = 0;
    int dev_id;
public:
    CudaAllocator() {
        cudaGetDevice(&dev_id);
    }
    ~CudaAllocator() {
        for (auto &it: cudaSmallBlocksMap) {
            auto &cudaBlocks = it.second; // vector
            for (int i = 0; i < cudaBlocks.size(); i++) {
                cudaFree(cudaBlocks[i].data);
            }
            auto &bigBlocks = CudaBigBlocksMap[it.first];
            for (int i = 0; i < bigBlocks.size(); i++) {
                cudaFree(bigBlocks[i].data);
            }
        }
    }

    void* UnifyMalloc(void* ptr, size_t size, bool is_host) {
        // 1.host malloc
        if (is_host) {
            ptr = malloc(size);
            memset(ptr, 0, size);
            return ptr;
        }
        // 2.big buf, 先去bigblocks里面找空闲的（free出来且未归还到OS的）
        if (size > 1024 * 1024) { // > 1M
            auto &BigBlocks = CudaBigBlocksMap[dev_id];
            int blockID = -1;
            for (int i = 0; i < BigBlocks.size(); i++) {
                // the freed bigblock by free method
                if (BigBlocks[i].size >= size && !BigBlocks[i].is_allocated
                    && BigBlocks[i].size - size < 1 * 1024 * 1024) {
                    if (blockID == -1 || BigBlocks[blockID].size > BigBlocks[i].size) {
                            blockID = i;
                    }
                }
            }
            // the allocated big block id
            if (blockID != -1) {
                BigBlocks[blockID].is_allocated = true;
                return BigBlocks[blockID].data;
            }
            // 没找到空闲的再cudamalloc，并插进block pool
            void* new_buffer;
            cudaMalloc(&new_buffer, size);
            total_allocated_size += size;
            BigBlocks.push_back(CudaBigBlock(new_buffer, size, true));
            return new_buffer;
        }
        // 3. small buf，先去smallblocks里面找空闲的（free出来且未归还到OS的）
        // 问题：为什么要分成大小blocks？答：用free size记录碎片
        auto &SmallBlocks = cudaSmallBlocksMap[dev_id];
        for (int i = 0; i < SmallBlocks.size(); i++) {
            if (SmallBlocks[i].size >= size && !SmallBlocks[i].is_allocated) {
                SmallBlocks[i].is_allocated = true;
                // 感觉这里应该是减 原仓库的实现有问题
                FreeSize[i] -= SmallBlocks[i].size; // 小buf size
                return SmallBlocks[i].data;
            }
        }
        // 4.没找到空闲的再cudaMalloc
        void* new_buffer = (void*)ptr;
        CHECK(cudaMalloc(&new_buffer, size));
        CHECK(cudaMemset(new_buffer, 0, size));
        SmallBlocks.push_back(CudaSmallBlock(new_buffer, size, true));
        return new_buffer;
    }

    void UnifyFree(void* ptr, bool is_host) {
        if (ptr == nullptr) {
            return;
        }
        // 1.host free
        if (is_host) {
            free(ptr);
            return;
        }
        // 2.清理碎片：当累计的小buf超出了1G时，清理未分配出去的smallblocks，已分配的还是保留在smallmap
        for (auto &it: cudaSmallBlocksMap) {
            if (FreeSize[it.first] > 1024 * 1024 * 1024) {
                auto &cudaBlocks = it.second;
                std::vector<CudaSmallBlock> temp;
                for (int i = 0; i < cudaBlocks.size(); i++) {
                    if (!cudaBlocks[i].is_allocated) {
                        cudaSetDevice(it.first);
                        cudaFree(cudaBlocks[i].data);
                    } else {
                        temp.push_back(cudaBlocks[i]);
                    }
                }
                cudaBlocks.clear();
                it.second = temp;
                FreeSize[it.first] = 0;
            }
        }
        // 3.找到待free的buffer的位置，设is_allocated = false，大小block都不归还到OS，除非没找到
        for (auto &it:cudaSmallBlocksMap) {
            auto &cudaBlocks = it.second;
            for (int i = 0; i < cudaBlocks.size(); i++) {
                if (cudaBlocks[i].data == ptr) {
                    FreeSize[it.first] += cudaBlocks[i].size;
                    cudaBlocks[i].is_allocated = false;
                    return;
                }
            }
            //若是大block，那不归还到OS
            auto &bigBlocks = CudaBigBlocksMap[it.first];
            for (int i = 0; i < bigBlocks.size(); i++) {
                if (bigBlocks[i].data == ptr) {
                    bigBlocks[i].is_allocated = false;
                    return;
                }
            }
        }

        cudaFree(ptr);
    }
};