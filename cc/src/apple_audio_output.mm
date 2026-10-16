#include "generated/tyr_ffi.h"

#include <cmath>
#include <cstring>
#include <mutex>
#include <atomic>
#include <thread>

#ifdef __APPLE__
#include <AudioToolbox/AudioToolbox.h>

namespace {

std::mutex g_output_mu;
std::atomic<bool> g_playing{false};

static lean_object* mk_io_error(const char* msg) {
  return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(msg)));
}

struct ToneBuffer {
  float* samples;
  uint32_t totalFrames;
  uint32_t offset;
};

static void output_callback(
    void* inUserData,
    AudioQueueRef inAQ,
    AudioQueueBufferRef inBuffer) {
  auto* tone = static_cast<ToneBuffer*>(inUserData);
  auto* out = static_cast<float*>(inBuffer->mAudioData);
  uint32_t framesToFill = inBuffer->mAudioDataByteSize / sizeof(float);
  uint32_t remaining = tone->totalFrames - tone->offset;
  uint32_t toWrite = remaining < framesToFill ? remaining : framesToFill;

  if (toWrite > 0) {
    std::memcpy(out, tone->samples + tone->offset, toWrite * sizeof(float));
    tone->offset += toWrite;
  }
  // Zero-fill any leftover.
  for (uint32_t i = toWrite; i < framesToFill; ++i) {
    out[i] = 0.0f;
  }
  inBuffer->mAudioDataByteSize = framesToFill * sizeof(float);

  if (tone->offset >= tone->totalFrames) {
    AudioQueueStop(inAQ, false);
    g_playing.store(false, std::memory_order_release);
  } else {
    AudioQueueEnqueueBuffer(inAQ, inBuffer, 0, nullptr);
  }
}

/// Generate and play a sine tone at `freqHz` for `durationMs` milliseconds
/// at `sampleRate` Hz.  The tone has a short fade-in/out to avoid clicks.
/// Returns true on success.
static bool play_tone_core(
    double freqHz,
    uint64_t durationMs,
    uint64_t sampleRate) {
  if (freqHz <= 0.0 || durationMs == 0 || sampleRate == 0) return true;

  std::lock_guard<std::mutex> lock(g_output_mu);

  uint32_t totalFrames = static_cast<uint32_t>((sampleRate * durationMs) / 1000);
  if (totalFrames == 0) totalFrames = 1;

  // Fade samples (5ms each side).
  uint32_t fadeSamples = static_cast<uint32_t>((sampleRate * 5) / 1000);
  if (fadeSamples > totalFrames / 2) fadeSamples = totalFrames / 2;

  auto* samples = new float[totalFrames];
  double twoPiF = 2.0 * M_PI * freqHz / static_cast<double>(sampleRate);
  for (uint32_t i = 0; i < totalFrames; ++i) {
    float s = static_cast<float>(std::sin(twoPiF * i));
    // Apply fade envelope.
    float env = 1.0f;
    if (i < fadeSamples) {
      env = static_cast<float>(i) / static_cast<float>(fadeSamples);
    } else if (i >= totalFrames - fadeSamples) {
      env = static_cast<float>(totalFrames - 1 - i) / static_cast<float>(fadeSamples);
    }
    samples[i] = s * env * 0.3f;  // 0.3 volume
  }

  ToneBuffer tone{samples, totalFrames, 0};

  AudioStreamBasicDescription fmt{};
  fmt.mSampleRate = static_cast<Float64>(sampleRate);
  fmt.mFormatID = kAudioFormatLinearPCM;
  fmt.mFormatFlags = kLinearPCMFormatFlagIsFloat | kAudioFormatFlagIsPacked;
  fmt.mFramesPerPacket = 1;
  fmt.mChannelsPerFrame = 1;
  fmt.mBitsPerChannel = 32;
  fmt.mBytesPerFrame = sizeof(float);
  fmt.mBytesPerPacket = sizeof(float);

  AudioQueueRef queue = nullptr;
  OSStatus st = AudioQueueNewOutput(
      &fmt, output_callback, &tone, nullptr, kCFRunLoopCommonModes, 0, &queue);
  if (st != noErr || queue == nullptr) {
    delete[] samples;
    return false;
  }

  // Allocate 3 buffers.
  uint32_t bufFrames = sampleRate / 10;  // 100ms buffers
  if (bufFrames > totalFrames) bufFrames = totalFrames;
  uint32_t bufBytes = bufFrames * sizeof(float);

  for (int i = 0; i < 3; ++i) {
    AudioQueueBufferRef buf = nullptr;
    st = AudioQueueAllocateBuffer(queue, bufBytes, &buf);
    if (st != noErr || buf == nullptr) {
      AudioQueueDispose(queue, true);
      delete[] samples;
      return false;
    }
    buf->mAudioDataByteSize = bufBytes;
    output_callback(&tone, queue, buf);
  }

  g_playing.store(true, std::memory_order_release);
  st = AudioQueueStart(queue, nullptr);
  if (st != noErr) {
    g_playing.store(false, std::memory_order_release);
    AudioQueueDispose(queue, true);
    delete[] samples;
    return false;
  }

  // Spin until playback finishes (short tones only — sub-second).
  while (g_playing.load(std::memory_order_acquire)) {
    CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.01, false);
  }

  AudioQueueDispose(queue, true);
  delete[] samples;
  return true;
}

static lean_object* play_tone_sync(
    double freqHz,
    uint64_t durationMs,
    uint64_t sampleRate) {
  if (!play_tone_core(freqHz, durationMs, sampleRate))
    return mk_io_error("audio_output: playback failed");
  return lean_io_result_mk_ok(lean_box(0));
}

} // namespace

extern "C" {

lean_object* lean_tyr_audio_output_beep(
    double freq_hz,
    uint64_t duration_ms,
    uint64_t sample_rate) {
  return play_tone_sync(freq_hz, duration_ms, sample_rate);
}

lean_object* lean_tyr_audio_output_beep_async(
    double freq_hz,
    uint64_t duration_ms,
    uint64_t sample_rate) {
  std::thread([=]() { play_tone_core(freq_hz, duration_ms, sample_rate); }).detach();
  return lean_io_result_mk_ok(lean_box(0));
}

} // extern "C"

#else

namespace {

static lean_object* mk_io_error(const char* msg) {
  return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(msg)));
}

} // namespace

extern "C" {

lean_object* lean_tyr_audio_output_beep(
    double, uint64_t, uint64_t) {
  return mk_io_error("audio output is only supported on macOS");
}

lean_object* lean_tyr_audio_output_beep_async(
    double, uint64_t, uint64_t) {
  return mk_io_error("audio output is only supported on macOS");
}

} // extern "C"

#endif
