#include "generated/tyr_ffi.h"

#include <cmath>
#include <deque>
#include <mutex>
#include <thread>
#include <chrono>
#include <string>
#include <atomic>

// Defined in float_buffer.cpp — creates a FloatBuffer from raw float data.
extern "C" lean_object* lean_float_buffer_from_raw(const float* src, size_t n);

#ifdef __APPLE__
#include <AudioToolbox/AudioToolbox.h>

namespace {

std::mutex g_audio_mu;
std::mutex g_lifecycle_mu;
std::deque<float> g_audio_fifo;
AudioQueueRef g_input_queue = nullptr;
std::atomic<AudioQueueRef> g_active_queue{nullptr};
std::atomic<bool> g_running{false};
std::atomic<uint64_t> g_target_rate{16000};
std::atomic<uint64_t> g_target_channels{1};

static lean_object* mk_io_error(const std::string& msg) {
  return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(msg.c_str())));
}

static void input_callback(
    void* /*inUserData*/,
    AudioQueueRef inAQ,
    AudioQueueBufferRef inBuffer,
    const AudioTimeStamp* /*inStartTime*/,
    UInt32 inNumPackets,
    const AudioStreamPacketDescription* /*inPacketDesc*/) {
  if (inNumPackets > 0 && inBuffer->mAudioData != nullptr) {
    const size_t n = static_cast<size_t>(inBuffer->mAudioDataByteSize / sizeof(float));
    const float* p = static_cast<const float*>(inBuffer->mAudioData);
    std::lock_guard<std::mutex> lock(g_audio_mu);
    for (size_t i = 0; i < n; ++i) {
      g_audio_fifo.push_back(p[i]);
    }
    // Keep ~30s cap to avoid unbounded growth.
    const uint64_t rate = g_target_rate.load(std::memory_order_relaxed);
    const uint64_t channels = g_target_channels.load(std::memory_order_relaxed);
    const size_t cap = static_cast<size_t>(rate * channels * 30);
    while (g_audio_fifo.size() > cap) {
      g_audio_fifo.pop_front();
    }
  }

  if (g_running.load(std::memory_order_acquire) &&
      g_active_queue.load(std::memory_order_acquire) == inAQ) {
    AudioQueueEnqueueBuffer(inAQ, inBuffer, 0, nullptr);
  }
}

static void stop_queue_locked() {
  g_running.store(false, std::memory_order_release);
  g_active_queue.store(nullptr, std::memory_order_release);

  AudioQueueRef queue = g_input_queue;
  g_input_queue = nullptr;
  if (queue != nullptr) {
    AudioQueueStop(queue, true);
    AudioQueueDispose(queue, true);
  }
}

static void stop_queue() {
  std::lock_guard<std::mutex> lifecycle_lock(g_lifecycle_mu);
  stop_queue_locked();
}

} // namespace

extern "C" {

lean_object* lean_tyr_audio_input_start(
    uint64_t sample_rate,
    uint64_t channels,
    uint64_t buffer_ms) {
  if (sample_rate == 0 || channels == 0) {
    return mk_io_error("audio_input_start: sample_rate/channels must be > 0");
  }

  std::lock_guard<std::mutex> lifecycle_lock(g_lifecycle_mu);
  stop_queue_locked();
  {
    std::lock_guard<std::mutex> lock(g_audio_mu);
    g_audio_fifo.clear();
  }

  g_target_rate.store(sample_rate, std::memory_order_relaxed);
  g_target_channels.store(channels, std::memory_order_relaxed);

  AudioStreamBasicDescription fmt{};
  fmt.mSampleRate = static_cast<Float64>(sample_rate);
  fmt.mFormatID = kAudioFormatLinearPCM;
  fmt.mFormatFlags = kLinearPCMFormatFlagIsFloat | kAudioFormatFlagIsPacked;
  fmt.mFramesPerPacket = 1;
  fmt.mChannelsPerFrame = static_cast<UInt32>(channels);
  fmt.mBitsPerChannel = 32;
  fmt.mBytesPerFrame = fmt.mChannelsPerFrame * sizeof(float);
  fmt.mBytesPerPacket = fmt.mFramesPerPacket * fmt.mBytesPerFrame;

  AudioQueueRef queue = nullptr;
  OSStatus st = AudioQueueNewInput(
      &fmt,
      input_callback,
      nullptr,
      nullptr,
      kCFRunLoopCommonModes,
      0,
      &queue);
  if (st != noErr || queue == nullptr) {
    return mk_io_error("audio_input_start: AudioQueueNewInput failed");
  }

  const uint64_t ms = (buffer_ms == 0) ? 100 : buffer_ms;
  uint64_t frames = (sample_rate * ms) / 1000;
  if (frames < 256) frames = 256;
  UInt32 bytes = static_cast<UInt32>(frames * channels * sizeof(float));

  for (int i = 0; i < 3; ++i) {
    AudioQueueBufferRef buf = nullptr;
    st = AudioQueueAllocateBuffer(queue, bytes, &buf);
    if (st != noErr || buf == nullptr) {
      AudioQueueDispose(queue, true);
      return mk_io_error("audio_input_start: AudioQueueAllocateBuffer failed");
    }
    buf->mAudioDataByteSize = bytes;
    st = AudioQueueEnqueueBuffer(queue, buf, 0, nullptr);
    if (st != noErr) {
      AudioQueueDispose(queue, true);
      return mk_io_error("audio_input_start: AudioQueueEnqueueBuffer failed");
    }
  }

  g_input_queue = queue;
  g_active_queue.store(queue, std::memory_order_release);
  g_running.store(true, std::memory_order_release);

  st = AudioQueueStart(queue, nullptr);
  if (st != noErr) {
    stop_queue_locked();
    return mk_io_error("audio_input_start: AudioQueueStart failed (check microphone permission)");
  }

  return lean_io_result_mk_ok(lean_box(0));
}

lean_object* lean_tyr_audio_input_read(
    uint64_t max_samples,
    uint64_t block_ms) {
  if (max_samples == 0) {
    return lean_io_result_mk_ok(lean_mk_empty_array());
  }

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(block_ms);
  while (true) {
    {
      std::lock_guard<std::mutex> lock(g_audio_mu);
      if (!g_audio_fifo.empty() || !g_running.load(std::memory_order_acquire)) break;
    }
    if (block_ms == 0 || std::chrono::steady_clock::now() >= deadline) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  lean_object* out = lean_mk_empty_array();
  {
    std::lock_guard<std::mutex> lock(g_audio_mu);
    size_t n = std::min(static_cast<size_t>(max_samples), g_audio_fifo.size());
    for (size_t i = 0; i < n; ++i) {
      out = lean_array_push(out, lean_box_float(static_cast<double>(g_audio_fifo.front())));
      g_audio_fifo.pop_front();
    }
  }
  return lean_io_result_mk_ok(out);
}

lean_object* lean_tyr_audio_input_read_buffer(
    uint64_t max_samples,
    uint64_t block_ms) {
  if (max_samples == 0) {
    return lean_io_result_mk_ok(lean_float_buffer_from_raw(nullptr, 0));
  }

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(block_ms);
  while (true) {
    {
      std::lock_guard<std::mutex> lock(g_audio_mu);
      if (!g_audio_fifo.empty() || !g_running.load(std::memory_order_acquire)) break;
    }
    if (block_ms == 0 || std::chrono::steady_clock::now() >= deadline) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  // Drain from FIFO into a temporary contiguous float buffer, then wrap as FloatBuffer.
  std::lock_guard<std::mutex> lock(g_audio_mu);
  size_t n = std::min(static_cast<size_t>(max_samples), g_audio_fifo.size());
  if (n == 0) {
    return lean_io_result_mk_ok(lean_float_buffer_from_raw(nullptr, 0));
  }
  auto* tmp = static_cast<float*>(std::malloc(sizeof(float) * n));
  for (size_t i = 0; i < n; ++i) {
    tmp[i] = g_audio_fifo.front();
    g_audio_fifo.pop_front();
  }
  lean_object* fb = lean_float_buffer_from_raw(tmp, n);
  std::free(tmp);
  return lean_io_result_mk_ok(fb);
}

lean_object* lean_tyr_audio_input_stop() {
  stop_queue();
  std::lock_guard<std::mutex> lock(g_audio_mu);
  g_audio_fifo.clear();
  return lean_io_result_mk_ok(lean_box(0));
}

lean_object* lean_tyr_audio_rms(b_lean_obj_arg arr) {
  size_t n = lean_array_size(arr);
  if (n == 0) return lean_io_result_mk_ok(lean_box_float(0.0));
  double sum_sq = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double x = lean_unbox_float(lean_array_uget(arr, i));
    sum_sq += x * x;
  }
  return lean_io_result_mk_ok(lean_box_float(std::sqrt(sum_sq / static_cast<double>(n))));
}

} // extern "C"

#else

namespace {

static lean_object* mk_io_error(const char* msg) {
  return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(msg)));
}

} // namespace

extern "C" {

lean_object* lean_tyr_audio_input_start(uint64_t, uint64_t, uint64_t) {
  return mk_io_error("audio input is only supported on macOS");
}

lean_object* lean_tyr_audio_input_read(uint64_t, uint64_t) {
  return mk_io_error("audio input is only supported on macOS");
}

lean_object* lean_tyr_audio_input_read_buffer(uint64_t, uint64_t) {
  return mk_io_error("audio input is only supported on macOS");
}

lean_object* lean_tyr_audio_input_stop() {
  return mk_io_error("audio input is only supported on macOS");
}

lean_object* lean_tyr_audio_rms(b_lean_obj_arg arr) {
  size_t n = lean_array_size(arr);
  if (n == 0) return lean_io_result_mk_ok(lean_box_float(0.0));
  double sum_sq = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double x = lean_unbox_float(lean_array_uget(arr, i));
    sum_sq += x * x;
  }
  return lean_io_result_mk_ok(lean_box_float(std::sqrt(sum_sq / static_cast<double>(n))));
}

} // extern "C"

#endif
