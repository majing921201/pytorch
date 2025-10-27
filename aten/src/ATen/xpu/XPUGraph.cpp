#include <ATen/xpu/XPUGeneratorImpl.h>
#include <ATen/xpu/XPUGraph.h>
#include <ATen/Functions.h>
#include <c10/xpu/XPUFunctions.h>

#include <cstddef>

namespace at::xpu {

using namespace sycl::ext::oneapi::experimental;
static bool _xpu_graphs_debug = false;

MempoolId_t graph_pool_handle() {
  // set the second value by default
  return c10::xpu::MemPool::graph_pool_handle();
}

XPUGraph::XPUGraph(bool keep_graph)
  : capture_stream_(at::xpu::getCurrentXPUStream()),
    keep_graph_(keep_graph) {
}

void XPUGraph::register_generator_state(
    c10::intrusive_ptr<at::XPUGeneratorState> state) {
  captured_generator_states_[std::move(state)] = 0;
}

void XPUGraph::register_generator_state(const at::Generator& generator) {
  c10::intrusive_ptr<XPUGeneratorImpl> xpu_gen =
      dynamic_intrusive_pointer_cast<XPUGeneratorImpl>(
          generator.getIntrusivePtr());
  xpu_gen->register_graph(this);
}

void XPUGraph::capture_begin(MempoolId_t pool) {
  TORCH_CHECK(!has_graph_exec_,
              "This XPUGraph instance already owns a captured graph. "
              "To capture a new graph, create a new instance.");

  // default generator is always registered
  auto* gen = get_generator_or_default<XPUGeneratorImpl>(
      std::nullopt, xpu::detail::getDefaultXPUGenerator());
  gen->register_graph(this);

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->capture_prologue();
  }

  capture_stream_ = at::xpu::getCurrentXPUStream();
  capture_dev_ = c10::xpu::current_device();

  if (pool.first != 0 || pool.second != 0) {
    // Either value being nonzero means the user supplied a pool to share.
    // But only one should be nonzero.
    // If pool was created by another graph's capture_begin, first should be nonzero.
    // If pool was created by graph_pool_handle, second should be nonzero.
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // User did not ask us to share a mempool. Create graph pool handle using is_user_created=false.
    // Sets just the first value, to distinguish it from MempoolId_ts created by graph_pool_handle().
    mempool_id_ = c10::xpu::MemPool::graph_pool_handle(false);
    TORCH_INTERNAL_ASSERT(mempool_id_.first > 0);
  }

  c10::xpu::XPUCachingAllocator::beginAllocateToPool(capture_dev_, mempool_id_, [this](sycl::queue* queue) {
    return queue->ext_oneapi_get_state() == queue_state::executing && queue == &capture_stream_.queue();
  });

  auto graph_impl = xpuGraph_t(capture_stream_.queue());
  graph_ = std::make_unique<xpuGraph_t>(std::move(graph_impl));
  graph_->begin_recording(capture_stream_.queue());

  TORCH_INTERNAL_ASSERT(capture_stream_.queue().ext_oneapi_get_state() == queue_state::recording);
}

void XPUGraph::capture_end() {
  auto stream = at::xpu::getCurrentXPUStream();

  TORCH_CHECK(stream == capture_stream_,
              "Capture must end on the same stream it began on.");

  graph_->end_recording();

  c10::xpu::XPUCachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);

  TORCH_CHECK(graph_ != nullptr, "Invalid capture.");

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    wholegraph_increments = generator_state->capture_epilogue();
  }

  size_t numXPUGraphNodes = graph_->get_nodes().size();
  if (numXPUGraphNodes == 0) {
      TORCH_WARN("The XPU Graph is empty. This usually means that the graph was ",
                 "attempted to be captured on wrong device or stream.");
  }

  capture_ended_ = true;
  has_graph_ = true;
  if (!keep_graph_) {
    instantiate();
    if (!_xpu_graphs_debug) {
      graph_.reset();
    }
    has_graph_ = false;
  }
}

void XPUGraph::instantiate() {
  TORCH_CHECK(capture_ended_, "capture_end() must have been called before calling instantiate");

  if (has_graph_exec_) {
    TORCH_CHECK(keep_graph_, "instantiate() is intended to be called by the user only when keep_graph=true");
    graph_exec_.reset();
  }
  auto graph_exec_impl = graph_->finalize();
  graph_exec_ = std::make_unique<xpuGraphExec_t>(std::move(graph_exec_impl));
  has_graph_exec_ = true;
}

void XPUGraph::replay() {
  TORCH_CHECK(capture_ended_,
              "Called XPUGraph::replay without a preceding successful capture.");

  if (!has_graph_exec_) {
    TORCH_INTERNAL_ASSERT(keep_graph_);
    instantiate();
  }

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->replay_prologue(wholegraph_increments);
  }

  auto& queue = at::xpu::getCurrentXPUStream().queue();
  queue.ext_oneapi_graph(*graph_exec_);
}

void XPUGraph::reset() {
  if (capture_ended_) {
    c10::xpu::XPUCachingAllocator::releasePool(capture_dev_, mempool_id_);
    capture_ended_ = false;
  }
  if (has_graph_) {
    graph_.reset();
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    graph_exec_.reset();
    has_graph_exec_ = false;
  }
}

XPUGraph::~XPUGraph() {
  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->unregister_graph(this);
  }
  reset();
}

} // namespace at::xpu
