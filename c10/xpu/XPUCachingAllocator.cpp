#include <c10/core/CachingDeviceAllocator.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/xpu/XPUCachingAllocator.h>

#include <deque>
#include <mutex>
#include <set>
#include <vector>

namespace c10::xpu::XPUCachingAllocator {

using namespace c10::CachingAllocator;
using namespace c10::CachingDeviceAllocator;

// newly allocated memory with 512-byte alignment.
constexpr size_t kDeviceAlignment = 512;

namespace {
using stream_set = ska::flat_hash_set<xpu::XPUStream>;

struct Block;
typedef bool (*Comparison)(const Block*, const Block*);
bool BlockComparatorSize(const Block* a, const Block* b);

struct PrivatePool;

struct BlockPool {
  BlockPool(bool small, PrivatePool* private_pool = nullptr)
      : blocks(BlockComparatorSize),
        is_small(small),
        owner_PrivatePool(private_pool) {}

  std::set<Block*, Comparison> blocks;
  const bool is_small;
  PrivatePool* owner_PrivatePool;

  // TODO:insert_into_blocks()

  MempoolId_t owner_MempoolId() const;
};

struct Block {
  DeviceIndex device;
  sycl::queue* queue{nullptr}; // underlying queue of the allocation stream
  stream_set stream_uses; // streams on which the block was used
  size_t size; // block size in bytes
  size_t requested_size; // memory originally requested
  BlockPool* pool{nullptr}; // owning memory pool
  void* ptr{nullptr}; // memory address
  bool allocated{false}; // in-use flag
  Block* prev{nullptr}; // prev block if split from a larger allocation
  Block* next{nullptr}; // next block if split from a larger allocation
  int event_count{0}; // number of outstanding XPU events

  Block(
      DeviceIndex device,
      sycl::queue* queue,
      size_t size,
      BlockPool* pool,
      void* ptr)
      : device(device),
        queue(queue),
        stream_uses(),
        size(size),
        requested_size(0),
        pool(pool),
        ptr(ptr) {}

  // constructor for search key
  Block(DeviceIndex device, sycl::queue* queue, size_t size)
      : device(device),
        queue(queue),
        stream_uses(),
        size(size),
        requested_size(0) {}

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }
};

bool BlockComparatorSize(const Block* a, const Block* b) {
  if (a->queue != b->queue) {
    return reinterpret_cast<uintptr_t>(a->queue) <
        reinterpret_cast<uintptr_t>(b->queue);
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return reinterpret_cast<uintptr_t>(a->ptr) <
      reinterpret_cast<uintptr_t>(b->ptr);
}

struct AllocParams {
  AllocParams(
      DeviceIndex device,
      size_t size,
      sycl::queue* queue,
      BlockPool* pool,
      size_t alloc_size)
      : search_key(device, queue, size),
        pool(pool),
        alloc_size(alloc_size),
        block(nullptr) {}

  DeviceIndex device() const {
    return search_key.device;
  }

  sycl::queue* queue() const {
    return search_key.queue;
  }

  size_t size() const {
    return search_key.size;
  }

  Block search_key;
  BlockPool* pool;
  size_t alloc_size;
  Block* block;
  StatTypes stat_types = {};
};

struct PrivatePool {
  PrivatePool(MempoolId_t id, XPUAllocator* allocator = nullptr)
      : id(std::move(id)),
        allocator_(allocator),
        large_blocks(/*small=*/false, this),
        small_blocks(/*small=*/true, this) {}
  PrivatePool(const PrivatePool&) = delete;
  PrivatePool(PrivatePool&&) = delete;
  PrivatePool& operator=(const PrivatePool&) = delete;
  PrivatePool& operator=(PrivatePool&&) = delete;
  ~PrivatePool() = default;

  MempoolId_t id{0, 0};
  // Number of live graphs using this pool
  int use_count{1};
  // Number of unfreed allocations made for this pool. When use_count and
  // allocation_count drop to zero, we can delete this PrivatePool from
  // graph_pools.
  int allocation_count{0};
  XPUAllocator* allocator_;
  BlockPool large_blocks;
  BlockPool small_blocks;

 public:
  XPUAllocator* allocator() {
    return allocator_;
  }
};

MempoolId_t BlockPool::owner_MempoolId() const {
  if (owner_PrivatePool) {
    return owner_PrivatePool->id;
  } else {
    return {0, 0};
  }
}

struct MempoolIdHash {
  std::size_t operator()(const MempoolId_t& mempool_id) const noexcept {
    return mempool_id.first != 0 ? mempool_id.first : mempool_id.second;
  }
};

} // anonymous namespace

class DeviceCachingAllocator {
 private:
  mutable std::recursive_mutex mutex;
  DeviceStats stats;
  BlockPool large_blocks; // unallocated cached blocks larger than 1 MB
  BlockPool small_blocks; // unallocated cached blocks 1 MB or smaller
  ska::flat_hash_set<Block*> active_blocks; // allocated or in use by a stream
  ska::flat_hash_map<xpu::XPUStream, std::deque<std::pair<sycl::event, Block*>>>
      xpu_events;
  DeviceIndex device_index;
  std::vector<std::pair<MempoolId_t, std::function<bool(sycl::queue*)>>>
      captures_underway;
  ska::flat_hash_map<MempoolId_t, std::unique_ptr<PrivatePool>, MempoolIdHash>
      graph_pools;
  // Pools no longer referenced by any graph. Their BlockPools are eligible for
  // free_blocks. Can't be a vector or deque because we might erase entries in
  // any order.
  ska::flat_hash_map<MempoolId_t, PrivatePool*, MempoolIdHash>
      graph_pools_freeable;

  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    if (!src || src->allocated || src->event_count > 0 ||
        !src->stream_uses.empty()) {
      return 0;
    }

    TORCH_INTERNAL_ASSERT(dst->is_split() && src->is_split());
    if (dst->prev == src) { // [src dst]
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else { // [dst src]
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }
    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    auto erased = pool.blocks.erase(src);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
    delete src;

    return subsumed_size;
  }

  void free_block(Block* block) {
    TORCH_INTERNAL_ASSERT(
        !block->allocated && block->event_count == 0 &&
        block->stream_uses.empty());

    size_t original_block_size = block->size;
    size_t requested_size = block->requested_size;
    auto& pool = *block->pool;
    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      try_merge_blocks(block, merge_candidate, pool);
    }

    active_blocks.erase(block);
    bool inserted = pool.blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

    StatTypes stat_types = get_stat_types_for_pool(pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.active_bytes[stat_type].decrease(original_block_size);
      stats.requested_bytes[stat_type].decrease(requested_size);
    });
  }

  void process_events() {
    using namespace sycl::info;
    for (auto it = xpu_events.begin(); it != xpu_events.end();) {
      while (!it->second.empty()) {
        auto& e = it->second.front();
        auto event = e.first;
        auto* block = e.second;
        if (event.get_info<event::command_execution_status>() !=
            event_command_status::complete) {
          break;
        }
        block->event_count--;
        if (block->event_count == 0) {
          free_block(block);
        }
        it->second.pop_front();
      }

      if (it->second.empty()) {
        it = xpu_events.erase(it);
      } else {
        it++;
      }
    }
  }

  static size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
  }

  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  BlockPool& get_pool(size_t size, sycl::queue* queue) {
    if (C10_UNLIKELY(!captures_underway.empty())) {
      for (auto& entry : captures_underway) {
        // lookup for mempool id matching current capture graph
        if (entry.second(queue)) {
          auto it1 = graph_pools.find(entry.first);
          // lookup mempool
          TORCH_INTERNAL_ASSERT(it1 != graph_pools.end());
          if (size <= kSmallSize) {
            return it1->second->small_blocks;
          } else {
            return it1->second->large_blocks;
          }
        }
      }
    }
    if (size < kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  bool get_free_block(AllocParams& p) {
    BlockPool& pool = *p.pool;
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() || (*it)->queue != p.queue()) {
      return false;
    }
    p.block = *it;
    pool.blocks.erase(it);
    return true;
  }

  bool alloc_block(AllocParams& p, bool isRetry) {
    auto size = p.alloc_size;
    auto device = p.device();
    if (isRetry) {
      stats.num_alloc_retries += 1;
    }
    void* ptr = sycl::aligned_alloc_device(
        kDeviceAlignment,
        size,
        xpu::get_raw_device(device),
        xpu::get_device_context());
    if (!ptr) {
      return false;
    }

    if (p.pool->owner_PrivatePool) {
      p.pool->owner_PrivatePool->allocation_count++;
    }
    p.block = new Block(device, p.queue(), size, p.pool, ptr);
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      stats.reserved_bytes[stat_type].increase(size);
    });
    return true;
  }

  void synchronize_and_free_events(PrivatePool* pool = nullptr) {
    for (auto& xe : xpu_events) {
      for (auto& e : xe.second) {
        auto event = e.first;
        auto* block = e.second;
        if (pool && block->pool->owner_PrivatePool != pool) {
          continue;
        }
        event.wait();
        block->event_count--;
        if (block->event_count == 0) {
          free_block(block);
        }
      }
    }
    xpu_events.clear();
  }

  void release_block(Block* block) {
    /*
     * Note [Safe to Free Blocks on BlockPool]
     *
     * Callers must ensure that all accesses to the block, whose raw pointer is
     * allocated by SYCL APIs, have been completed before invoking sycl::free.
     *
     * We have to do a device-level synchronization before free these blocks to
     * guarantee that all kernels can access to the blocks have finished.
     */
    sycl::free(block->ptr, xpu::get_device_context());
    auto* pool = block->pool;
    pool->blocks.erase(block);

    StatTypes stat_types = get_stat_types_for_pool(*pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.reserved_bytes[stat_type].decrease(block->size);
    });

    delete block;
  }

  void release_blocks(BlockPool& pool) {
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block* block = *it;
      ++it;
      if (!block->prev && !block->next) {
        release_block(block);
      }
    }
  }

  bool release_cached_blocks(MempoolId_t mempool_id) {
    if (mempool_id.first == 0 && mempool_id.second == 0 &&
        captures_underway.empty()) {
      synchronize_and_free_events();
      c10::xpu::syncStreamsOnDevice(device_index);

      release_blocks(large_blocks);
      release_blocks(small_blocks);
    }

    for (auto it = graph_pools_freeable.begin();
         it != graph_pools_freeable.end();) {
      if (mempool_id.first != 0 || mempool_id.second != 0) {
        if (it->first == mempool_id) {
          // If there is an active mempool, we sync only the events
          // associated with the pool
          synchronize_and_free_events(it->second);
        } else {
          // otherwise we move on
          ++it;
          continue;
        }
      }
      TORCH_INTERNAL_ASSERT(it->second->use_count == 0);
      release_blocks(it->second->small_blocks);
      release_blocks(it->second->large_blocks);
      if (it->second->allocation_count == 0) {
        auto erase_count = graph_pools.erase(it->first);
        TORCH_INTERNAL_ASSERT(erase_count == 1);
        it = graph_pools_freeable.erase(it);
      } else {
        ++it;
      }
    }
    return true;
  }

  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small) {
      return remaining >= kMinBlockSize;
    } else {
      return remaining > kSmallSize;
    }
  }

  StatTypes get_stat_types_for_pool(const BlockPool& pool) {
    StatTypes stat_types = {};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(
        pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL)] = true;
    return stat_types;
  }

  Block* alloc_found_block(
      AllocParams params,
      size_t orig_size,
      bool split_remainder) {
    auto size = params.size();
    auto device = params.device();
    BlockPool* pool = params.pool;
    sycl::queue* queue = params.queue();

    TORCH_INTERNAL_ASSERT(
        params.block != nullptr && params.block->ptr != nullptr);
    Block* block = params.block;
    Block* remaining = nullptr;

    if (split_remainder) {
      remaining = block;

      block = new Block(device, queue, size, pool, block->ptr);
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
      bool inserted = pool->blocks.insert(remaining).second;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);
    }

    block->allocated = true;
    block->requested_size = orig_size;
    bool inserted = active_blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted)

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
      stats.allocated_bytes[stat_type].increase(block->size);
      stats.active_bytes[stat_type].increase(block->size);
      stats.requested_bytes[stat_type].increase(block->requested_size);
    });

    c10::reportMemoryUsageToProfiler(
        block->ptr,
        static_cast<int64_t>(block->size),
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::XPU, device));

    return block;
  }

  void insert_events(Block* block) {
    stream_set streams(std::move(block->stream_uses));
    TORCH_INTERNAL_ASSERT(block->stream_uses.empty());
    for (auto& stream : streams) {
      block->event_count++;
      xpu_events[stream].emplace_back(
          stream.queue().ext_oneapi_submit_barrier(), block);
    }
  }

  void create_or_incref_pool(
      MempoolId_t mempool_id,
      XPUAllocator* allocator = nullptr) {
    auto it = graph_pools.find(mempool_id);
    if (it == graph_pools.end()) {
      // mempool_id does not reference an existing pool.
      // Make a new pool for XPU graph capture or memory pool usage.
      graph_pools.emplace(
          mempool_id, std::make_unique<PrivatePool>(mempool_id, allocator));
    } else {
      // mempool_id references an existing pool, which the current XPU graph
      // capture will share.
      TORCH_INTERNAL_ASSERT(it->second->use_count > 0);
      TORCH_INTERNAL_ASSERT(allocator == nullptr);
      it->second->use_count++;
    }
  }

  PrivatePool* get_private_pool(MempoolId_t mempool_id) {
    auto it = graph_pools.find(mempool_id);
    TORCH_INTERNAL_ASSERT(it != graph_pools.end());
    return it->second.get();
  }

 public:
  DeviceCachingAllocator(DeviceIndex device_index)
      : large_blocks(/* small */ false),
        small_blocks(/* small */ true),
        device_index(device_index) {}

  Block* malloc(DeviceIndex device, size_t orig_size, sycl::queue& queue) {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    if (C10_LIKELY(captures_underway.empty())) {
      process_events();
    }
    size_t size = round_size(orig_size);
    auto& pool = get_pool(size, &queue);
    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, &queue, &pool, alloc_size);
    params.stat_types = get_stat_types_for_pool(pool);

    // First, try to get a block from the existing pool.
    bool block_found = get_free_block(params);
    // Can't reuse an existing block, try to get a new one.
    if (!block_found) {
      block_found = alloc_block(params, false) ||
          (release_cached_blocks({0, 0}) && alloc_block(params, true));
    }
    if (!block_found) {
      c10::xpu::DeviceProp device_prop;
      c10::xpu::get_device_properties(&device_prop, device);
      auto device_total = device_prop.global_mem_size;
      // Estimate the available device memory when the SYCL runtime does not
      // support the corresponding aspect (ext_intel_free_memory).
      size_t device_free = device_prop.global_mem_size -
          stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      auto& raw_device = c10::xpu::get_raw_device(device);
      // TODO: Remove the aspect check once the SYCL runtime bug is fixed on
      // affected devices.
      if (raw_device.has(sycl::aspect::ext_intel_free_memory)) {
        device_free =
            raw_device.get_info<sycl::ext::intel::info::device::free_memory>();
      }
      auto allocated_bytes =
          stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      auto reserved_bytes =
          stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;

      c10::reportOutOfMemoryToProfiler(
          static_cast<int64_t>(size),
          allocated_bytes,
          reserved_bytes,
          c10::Device(c10::DeviceType::XPU, device));

      TORCH_CHECK_WITH(
          OutOfMemoryError,
          false,
          "XPU out of memory. Tried to allocate ",
          format_size(alloc_size),
          ". GPU ",
          static_cast<int>(device),
          " has a total capacity of ",
          format_size(device_total),
          " of which ",
          format_size(device_free),
          " is free. Of the allocated memory ",
          format_size(allocated_bytes),
          " is allocated by PyTorch, and ",
          format_size(reserved_bytes - allocated_bytes),
          " is reserved by PyTorch but unallocated.",
          " Please use `empty_cache` to release all unoccupied cached memory.");
    }
    bool split_remainder = should_split(params.block, params.size());
    return alloc_found_block(std::move(params), orig_size, split_remainder);
  }

  void free(Block* block) {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    block->allocated = false;

    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      stats.allocated_bytes[stat_type].decrease(block->size);
    });

    if (!block->stream_uses.empty()) {
      insert_events(block);
    } else {
      free_block(block);
    }

    c10::reportMemoryUsageToProfiler(
        orig_block_ptr,
        -static_cast<int64_t>(orig_block_size),
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::XPU, block->device));
  }

  void recordStream(Block* block, xpu::XPUStream stream) {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    if (stream.queue() == *block->queue) {
      return;
    }
    block->stream_uses.insert(stream);
  }

  void emptyCache(MempoolId_t mempool_id) {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    release_cached_blocks(mempool_id);
  }

  DeviceStats getStats() {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    return stats;
  }

  void resetAccumulatedStats() {
    std::scoped_lock<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      stats.allocated_bytes[statType].reset_accumulated();
      stats.reserved_bytes[statType].reset_accumulated();
      stats.active_bytes[statType].reset_accumulated();
      stats.requested_bytes[statType].reset_accumulated();
    }
    stats.num_alloc_retries = 0;
  }

  void resetPeakStats() {
    std::scoped_lock<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      stats.allocated_bytes[statType].reset_peak();
      stats.reserved_bytes[statType].reset_peak();
      stats.active_bytes[statType].reset_peak();
      stats.requested_bytes[statType].reset_peak();
    }
  }

  void createOrIncrefPool(
      MempoolId_t mempool_id,
      XPUAllocator* allocator = nullptr) {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    create_or_incref_pool(mempool_id, allocator);
  }

  int getPoolUseCount(MempoolId_t mempool_id) {
    std::scoped_lock<std::recursive_mutex> lock(mutex);
    auto it = graph_pools.find(mempool_id);
    if (it == graph_pools.end()) {
      return 0;
    }
    return it->second->use_count;
  }

  // Called by XPUGraph::capture_begin
  void beginAllocateToPool(
      MempoolId_t mempool_id,
      std::function<bool(sycl::queue*)> filter) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    create_or_incref_pool(mempool_id);
    for (auto it2 = captures_underway.begin(); it2 != captures_underway.end();
         ++it2) {
      TORCH_CHECK(
          it2->first != mempool_id,
          "beginAllocateToPool: already recording to mempool_id");
    }
    captures_underway.emplace_back(mempool_id, std::move(filter));
  }

  // Called by XPUGraph::capture_end
  void endAllocateToPool(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (auto it = captures_underway.begin(); it != captures_underway.end();
         ++it) {
      if (it->first == mempool_id) {
        captures_underway.erase(it);
        return;
      }
    }
    TORCH_CHECK(
        false, "endAllocatePool: not currently recording to mempool_id");
  }

  // Called by XPUGraph::reset and MemPool::~MemPool()
  void releasePool(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto pp = get_private_pool(mempool_id);
    auto uc = --(pp->use_count);
    TORCH_INTERNAL_ASSERT(uc >= 0);
    if (uc == 0) {
      bool inserted = graph_pools_freeable.insert({mempool_id, pp}).second;
      TORCH_INTERNAL_ASSERT(inserted);
    }
  }
};

static void local_raw_delete(void* ptr);

class XPUAllocator : public DeviceAllocator {
 private:
  alignas(hardware_destructive_interference_size) std::mutex mutex;
  ska::flat_hash_map<void*, Block*> allocated_blocks;

  void add_allocated_block(Block* block) {
    std::lock_guard<std::mutex> lock(mutex);
    allocated_blocks[block->ptr] = block;
  }

  Block* get_allocated_block(void* ptr, bool remove = false) {
    std::scoped_lock<std::mutex> lock(mutex);
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return nullptr;
    }
    Block* block = it->second;
    if (remove) {
      allocated_blocks.erase(it);
    }
    return block;
  }

 public:
  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocators;

  void init(DeviceIndex device_count) {
    const auto size = static_cast<DeviceIndex>(device_allocators.size());
    if (size < device_count) {
      device_allocators.resize(device_count);
      for (const auto i : c10::irange(size, device_count)) {
        device_allocators[i] = std::make_unique<DeviceCachingAllocator>(i);
      }
    }
  }

  bool initialized() override {
    return !device_allocators.empty();
  }

  void malloc(
      void** devPtr,
      DeviceIndex device,
      size_t size,
      sycl::queue& queue) {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocators.size(),
        "Allocator not initialized for device ",
        static_cast<int16_t>(device),
        ": did you call init?");
    Block* block = device_allocators[device]->malloc(device, size, queue);
    add_allocated_block(block);
    *devPtr = block->ptr;
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_allocation(
          c10::kXPU, reinterpret_cast<uintptr_t>(*devPtr));
    }
  }

  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    Block* block = get_allocated_block(ptr, /* remove */ true);
    TORCH_CHECK(block, "invalid device pointer: ", ptr);
    device_allocators[block->device]->free(block);
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_deallocation(
          c10::kXPU, reinterpret_cast<uintptr_t>(block->ptr));
    }
  }

  void emptyCache(MempoolId_t mempool_id) override {
    for (auto& da : device_allocators) {
      da->emptyCache(mempool_id);
    }
  }

  void recordStream(const DataPtr& ptr, c10::Stream stream) override {
    if (!ptr.get()) {
      return;
    }
    if (ptr.get_deleter() != &local_raw_delete) {
      return;
    }

    Block* block = get_allocated_block(ptr.get());
    TORCH_CHECK(block, "No allocated block can be found.");
    c10::xpu::XPUStream xpu_stream{stream};
    device_allocators[block->device]->recordStream(block, xpu_stream);
  }

  DataPtr allocate(size_t size) override {
    auto device = c10::xpu::current_device();
    void* r = nullptr;
    if (size != 0) {
      this->malloc(&r, device, size, xpu::getCurrentXPUStream(device));
    }
    return {r, r, &local_raw_delete, Device(DeviceType::XPU, device)};
  }

  DeleterFnPtr raw_deleter() const override {
    return &local_raw_delete;
  }

  void* raw_alloc(size_t size) {
    if (size == 0) {
      return nullptr;
    }
    auto device = c10::xpu::current_device();
    void* r = nullptr;
    malloc(&r, device, size, xpu::getCurrentXPUStream(device));
    return r;
  }

  void* raw_alloc_with_stream(size_t size, XPUStream stream) {
    if (size == 0) {
      return nullptr;
    }
    auto device = c10::xpu::current_device();
    void* r = nullptr;
    malloc(&r, device, size, stream);
    return r;
  }

  void raw_delete(void* ptr) {
    this->free(ptr);
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    xpu::getCurrentXPUStream().queue().memcpy(dest, src, count);
  }

  void assertValidDevice(DeviceIndex device) {
    const auto device_num = device_allocators.size();
    TORCH_CHECK(
        0 <= device && device < static_cast<int64_t>(device_num),
        "Invalid device argument ",
        device,
        ": did you call init?");
  }

  DeviceStats getDeviceStats(DeviceIndex device) override {
    assertValidDevice(device);
    return device_allocators[device]->getStats();
  }

  void resetPeakStats(DeviceIndex device) override {
    assertValidDevice(device);
    device_allocators[device]->resetPeakStats();
  }

  void resetAccumulatedStats(DeviceIndex device) override {
    assertValidDevice(device);
    device_allocators[device]->resetAccumulatedStats();
  }

  void createOrIncrefPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      XPUAllocator* allocator) {
    assertValidDevice(device);
    device_allocators[device]->createOrIncrefPool(
        std::move(mempool_id), allocator);
  }

  void beginAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::function<bool(sycl::queue*)> filter) {
    assertValidDevice(device);
    device_allocators[device]->beginAllocateToPool(
        std::move(mempool_id), std::move(filter));
  }

  void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) {
    assertValidDevice(device);
    device_allocators[device]->endAllocateToPool(mempool_id);
  }

  void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) {
    assertValidDevice(device);
    device_allocators[device]->releasePool(std::move(mempool_id));
  }

  int getPoolUseCount(c10::DeviceIndex device, MempoolId_t mempool_id) {
    assertValidDevice(device);
    return device_allocators[device]->getPoolUseCount(std::move(mempool_id));
  }
};

static XPUAllocator allocator;

void local_raw_delete(void* ptr) {
  allocator.free(ptr);
}

Allocator* get() {
  return &allocator;
}

void init(DeviceIndex device_count) {
  return allocator.init(device_count);
}

void emptyCache(MempoolId_t mempool_id) {
  return allocator.emptyCache(mempool_id);
}

void resetPeakStats(DeviceIndex device) {
  return allocator.resetPeakStats(device);
}

void resetAccumulatedStats(DeviceIndex device) {
  return allocator.resetAccumulatedStats(device);
}

DeviceStats getDeviceStats(DeviceIndex device) {
  return allocator.getDeviceStats(device);
}

void* raw_alloc(size_t size) {
  return allocator.raw_alloc(size);
}

void raw_delete(void* ptr) {
  return allocator.raw_delete(ptr);
}

void recordStream(const DataPtr& dataPtr, XPUStream stream) {
  return allocator.recordStream(dataPtr, stream);
}

void createOrIncrefPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    XPUAllocator* allocator_ptr) {
  return allocator.createOrIncrefPool(device, mempool_id, allocator_ptr);
}

void beginAllocateToPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::function<bool(sycl::queue*)> filter) {
  return allocator.beginAllocateToPool(device, mempool_id, std::move(filter));
}

void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  return allocator.endAllocateToPool(device, mempool_id);
}

void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  return allocator.releasePool(device, mempool_id);
}

int getPoolUseCount(c10::DeviceIndex device, MempoolId_t mempool_id) {
  return allocator.getPoolUseCount(device, mempool_id);
}

REGISTER_ALLOCATOR(kXPU, &allocator)

} // namespace c10::xpu::XPUCachingAllocator

namespace c10::xpu {

// uid_ is incremented when a user creates a MemPool,
//
// uuid_ is incremented when XPUGraph creates a MemPool
// as a result of a user not providing a pool.

std::atomic<CaptureId_t> MemPool::uid_{1};
std::atomic<CaptureId_t> MemPool::uuid_{1};

MemPool::MemPool(
    XPUCachingAllocator::XPUAllocator* allocator,
    bool is_user_created,
    bool use_on_oom)
    : allocator_(allocator), is_user_created_(is_user_created) {
  if (is_user_created_) {
    id_ = {0, uid_++};
  } else {
    id_ = {uuid_++, 0};
  }
  device_ = c10::xpu::current_device();
  XPUCachingAllocator::createOrIncrefPool(device_, id_, allocator);
  if (use_on_oom) {
    // XPU doesn't support use_on_oom yet, but we keep the interface
    // XPUCachingAllocator::setUseOnOOM(device_, id_);
  }
}

MemPool::~MemPool() {
  TORCH_INTERNAL_ASSERT(use_count() == 1);
  XPUCachingAllocator::releasePool(device_, id_);
  c10::xpu::XPUCachingAllocator::emptyCache(id_); // release cached blocks
}

MempoolId_t MemPool::id() {
  return id_;
}

XPUCachingAllocator::XPUAllocator* MemPool::allocator() {
  return allocator_;
}

int MemPool::use_count() {
  return XPUCachingAllocator::getPoolUseCount(device_, id_);
}

c10::DeviceIndex MemPool::device() {
  return device_;
}

MempoolId_t MemPool::graph_pool_handle(bool is_user_created) {
  if (is_user_created) {
    return {0, uid_++};
  }
  return {uuid_++, 0};
}

} // namespace c10::xpu
