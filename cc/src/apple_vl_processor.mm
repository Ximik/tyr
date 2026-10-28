#include "tyr_nat.h"
#include "generated/tyr_ffi.h"
#include "tyr_checked_result.h"
/*
 * apple_vl_processor.mm
 *
 * Apple-only image/video preprocessing for Qwen3.5-VL patch inputs.
 * No fallback path: requires macOS system media frameworks.
 */

#include <lean/lean.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#ifdef __APPLE__
#import <AVFoundation/AVFoundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>

// Defined in tyr.cpp
lean_object *fromTorchTensor(torch::Tensor t);

static lean_object* mk_io_error(const std::string& msg) {
  return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(msg.c_str())));
}

// CoreFoundation Create/Copy ownership and pixel-buffer locks must survive C++
// exceptions from the vectors and tensor allocations in the media pipeline.
class ScopedCFRelease {
  CFTypeRef value_;
public:
  explicit ScopedCFRelease(CFTypeRef value) : value_(value) {}
  ScopedCFRelease(const ScopedCFRelease&) = delete;
  ScopedCFRelease& operator=(const ScopedCFRelease&) = delete;
  ~ScopedCFRelease() { reset(); }
  void reset() {
    if (value_) CFRelease(value_);
    value_ = nullptr;
  }
};

class ScopedPixelBufferLock {
  CVPixelBufferRef value_;
public:
  explicit ScopedPixelBufferLock(CVPixelBufferRef value)
      : value_(CVPixelBufferLockBaseAddress(value, kCVPixelBufferLock_ReadOnly) == kCVReturnSuccess
                   ? value : nullptr) {}
  ScopedPixelBufferLock(const ScopedPixelBufferLock&) = delete;
  ScopedPixelBufferLock& operator=(const ScopedPixelBufferLock&) = delete;
  ~ScopedPixelBufferLock() { reset(); }
  bool locked() const { return value_ != nullptr; }
  void reset() {
    if (value_) CVPixelBufferUnlockBaseAddress(value_, kCVPixelBufferLock_ReadOnly);
    value_ = nullptr;
  }
};

static bool load_image_rgb_f32(
    const std::string& path,
    int& width,
    int& height,
    std::vector<float>& rgb,
    std::string& err) {
  @autoreleasepool {
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    if (url == nil) {
      err = "Invalid image path URL";
      return false;
    }

    CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr);
    ScopedCFRelease src_owner(src);
    if (src == nullptr) {
      err = "Failed to open image with ImageIO";
      return false;
    }

    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
    ScopedCFRelease img_owner(img);
    src_owner.reset();
    if (img == nullptr) {
      err = "Failed to decode image";
      return false;
    }

    width = static_cast<int>(CGImageGetWidth(img));
    height = static_cast<int>(CGImageGetHeight(img));
    if (width <= 0 || height <= 0) {
      err = "Decoded image has invalid dimensions";
      return false;
    }

    std::vector<uint8_t> rgba(static_cast<size_t>(width) * static_cast<size_t>(height) * 4);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    ScopedCFRelease color_space_owner(colorSpace);
    CGContextRef ctx = CGBitmapContextCreate(
      rgba.data(),
      width,
      height,
      8,
      width * 4,
      colorSpace,
      kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big
    );
    ScopedCFRelease context_owner(ctx);
    color_space_owner.reset();
    if (ctx == nullptr) {
      err = "Failed to create bitmap context";
      return false;
    }

    CGContextDrawImage(ctx, CGRectMake(0, 0, width, height), img);
    context_owner.reset();
    img_owner.reset();

    rgb.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
    for (size_t i = 0; i < static_cast<size_t>(width) * static_cast<size_t>(height); ++i) {
      rgb[i * 3 + 0] = static_cast<float>(rgba[i * 4 + 0]) / 255.0f;
      rgb[i * 3 + 1] = static_cast<float>(rgba[i * 4 + 1]) / 255.0f;
      rgb[i * 3 + 2] = static_cast<float>(rgba[i * 4 + 2]) / 255.0f;
    }
  }
  return true;
}

struct PatchGrid {
  int h_eff = 0;
  int w_eff = 0;
  int y_off = 0;
  int x_off = 0;
  uint64_t patches_per_group = 0;
  uint64_t patch_dim = 0;
};

static bool build_patch_grid(
    int width,
    int height,
    uint64_t in_channels,
    uint64_t patch_size,
    uint64_t temporal_patch_size,
    PatchGrid& grid,
    std::string& err) {
  if (in_channels != 3) {
    err = "Apple media preprocessor currently supports in_channels=3 only";
    return false;
  }
  if (patch_size == 0 || temporal_patch_size == 0) {
    err = "patch_size and temporal_patch_size must be > 0";
    return false;
  }

  grid.h_eff = (height / static_cast<int>(patch_size)) * static_cast<int>(patch_size);
  grid.w_eff = (width / static_cast<int>(patch_size)) * static_cast<int>(patch_size);
  if (grid.h_eff <= 0 || grid.w_eff <= 0) {
    err = "Image/video dimensions are smaller than patch_size";
    return false;
  }

  grid.y_off = (height - grid.h_eff) / 2;
  grid.x_off = (width - grid.w_eff) / 2;
  grid.patches_per_group =
      static_cast<uint64_t>(grid.h_eff / static_cast<int>(patch_size)) *
      static_cast<uint64_t>(grid.w_eff / static_cast<int>(patch_size));
  grid.patch_dim = in_channels * temporal_patch_size * patch_size * patch_size;
  return true;
}

static void append_patch_group(
    const std::vector<const std::vector<float>*>& group_frames,
    int width,
    uint64_t in_channels,
    uint64_t patch_size,
    uint64_t temporal_patch_size,
    const PatchGrid& grid,
    std::vector<float>& out) {
  const size_t group_floats =
      static_cast<size_t>(grid.patches_per_group) * static_cast<size_t>(grid.patch_dim);
  const size_t write_base = out.size();
  out.resize(write_base + group_floats);
  size_t write = write_base;

  for (int py = 0; py < grid.h_eff / static_cast<int>(patch_size); ++py) {
    for (int px = 0; px < grid.w_eff / static_cast<int>(patch_size); ++px) {
      for (uint64_t c = 0; c < in_channels; ++c) {
        for (uint64_t t = 0; t < temporal_patch_size; ++t) {
          const auto& frame = *group_frames[static_cast<size_t>(t)];
          for (uint64_t dy = 0; dy < patch_size; ++dy) {
            for (uint64_t dx = 0; dx < patch_size; ++dx) {
              int y = grid.y_off + py * static_cast<int>(patch_size) + static_cast<int>(dy);
              int x = grid.x_off + px * static_cast<int>(patch_size) + static_cast<int>(dx);
              size_t pix =
                  (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3 +
                  static_cast<size_t>(c);
              out[write++] = frame[pix];
            }
          }
        }
      }
    }
  }
}

static bool patchify_rgb_frames(
    const std::vector<std::vector<float>>& frames,
    int width,
    int height,
    uint64_t in_channels,
    uint64_t patch_size,
    uint64_t temporal_patch_size,
    std::vector<float>& out,
    uint64_t& n_patches,
    std::string& err) {
  if (frames.empty()) {
    err = "No frames available for patchify";
    return false;
  }

  PatchGrid grid;
  if (!build_patch_grid(width, height, in_channels, patch_size, temporal_patch_size, grid, err)) {
    return false;
  }

  uint64_t groups =
      (static_cast<uint64_t>(frames.size()) + temporal_patch_size - 1) / temporal_patch_size;
  n_patches = groups * grid.patches_per_group;

  out.clear();
  out.reserve(static_cast<size_t>(n_patches) * static_cast<size_t>(grid.patch_dim));

  for (uint64_t g = 0; g < groups; ++g) {
    std::vector<const std::vector<float>*> group_frames;
    group_frames.reserve(static_cast<size_t>(temporal_patch_size));
    for (uint64_t t = 0; t < temporal_patch_size; ++t) {
      uint64_t frame_idx = g * temporal_patch_size + t;
      if (frame_idx >= frames.size()) {
        frame_idx = static_cast<uint64_t>(frames.size() - 1);
      }
      group_frames.push_back(&frames[static_cast<size_t>(frame_idx)]);
    }
    append_patch_group(group_frames, width, in_channels, patch_size, temporal_patch_size, grid, out);
  }

  return true;
}

static bool compute_gemma4_target_size(
    int width,
    int height,
    uint64_t patch_size,
    uint64_t pooling_kernel_size,
    uint64_t max_soft_tokens,
    int& target_h,
    int& target_w,
    std::string& err) {
  if (width <= 0 || height <= 0) {
    err = "image dimensions must be positive";
    return false;
  }
  if (patch_size == 0 || pooling_kernel_size == 0 || max_soft_tokens == 0) {
    err = "patch_size, pooling_kernel_size, and max_soft_tokens must be > 0";
    return false;
  }

  const double total_px = static_cast<double>(width) * static_cast<double>(height);
  const double target_px =
      static_cast<double>(max_soft_tokens) * static_cast<double>(patch_size) * static_cast<double>(patch_size) *
      static_cast<double>(pooling_kernel_size) * static_cast<double>(pooling_kernel_size);
  const double factor = std::sqrt(target_px / total_px);
  const double ideal_h = factor * static_cast<double>(height);
  const double ideal_w = factor * static_cast<double>(width);
  const uint64_t side_mult = pooling_kernel_size * patch_size;

  target_h = static_cast<int>(std::floor(ideal_h / static_cast<double>(side_mult)) * static_cast<double>(side_mult));
  target_w = static_cast<int>(std::floor(ideal_w / static_cast<double>(side_mult)) * static_cast<double>(side_mult));

  if (target_h == 0 && target_w == 0) {
    err = "attempted to resize to 0x0 image";
    return false;
  }

  const int max_side_length = static_cast<int>(max_soft_tokens * side_mult);
  if (target_h == 0) {
    target_h = static_cast<int>(side_mult);
    target_w = std::min(
        static_cast<int>(std::floor(static_cast<double>(width) / static_cast<double>(height)) *
                         static_cast<double>(side_mult)),
        max_side_length);
  } else if (target_w == 0) {
    target_w = static_cast<int>(side_mult);
    target_h = std::min(
        static_cast<int>(std::floor(static_cast<double>(height) / static_cast<double>(width)) *
                         static_cast<double>(side_mult)),
        max_side_length);
  }

  if (target_h <= 0 || target_w <= 0) {
    err = "computed non-positive target image size";
    return false;
  }
  if (static_cast<uint64_t>(target_h) % side_mult != 0 || static_cast<uint64_t>(target_w) % side_mult != 0) {
    err = "computed target size is not divisible by pooling_kernel_size * patch_size";
    return false;
  }
  if (static_cast<double>(target_h) * static_cast<double>(target_w) > target_px + 1e-6) {
    err = "computed target image size exceeds patch budget";
    return false;
  }
  return true;
}

static void resize_rgb_bilinear(
    const std::vector<float>& src,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h,
    std::vector<float>& dst) {
  if (src_w == dst_w && src_h == dst_h) {
    dst = src;
    return;
  }

  dst.resize(static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h) * 3);
  const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
  const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);

  for (int y = 0; y < dst_h; ++y) {
    float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
    int y0 = static_cast<int>(std::floor(src_y));
    int y1 = y0 + 1;
    float ly = src_y - static_cast<float>(y0);
    y0 = std::max(0, std::min(y0, src_h - 1));
    y1 = std::max(0, std::min(y1, src_h - 1));

    for (int x = 0; x < dst_w; ++x) {
      float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
      int x0 = static_cast<int>(std::floor(src_x));
      int x1 = x0 + 1;
      float lx = src_x - static_cast<float>(x0);
      x0 = std::max(0, std::min(x0, src_w - 1));
      x1 = std::max(0, std::min(x1, src_w - 1));

      for (int c = 0; c < 3; ++c) {
        auto sample = [&](int sy, int sx) -> float {
          size_t idx =
              (static_cast<size_t>(sy) * static_cast<size_t>(src_w) + static_cast<size_t>(sx)) * 3 +
              static_cast<size_t>(c);
          return src[idx];
        };
        float v00 = sample(y0, x0);
        float v01 = sample(y0, x1);
        float v10 = sample(y1, x0);
        float v11 = sample(y1, x1);
        float top = v00 + (v01 - v00) * lx;
        float bot = v10 + (v11 - v10) * lx;
        float out = top + (bot - top) * ly;
        size_t dst_idx =
            (static_cast<size_t>(y) * static_cast<size_t>(dst_w) + static_cast<size_t>(x)) * 3 +
            static_cast<size_t>(c);
        dst[dst_idx] = out;
      }
    }
  }
}

static void patchify_resized_image(
    const std::vector<float>& rgb,
    int width,
    int height,
    uint64_t patch_size,
    std::vector<float>& patches) {
  uint64_t patch_rows = static_cast<uint64_t>(height) / patch_size;
  uint64_t patch_cols = static_cast<uint64_t>(width) / patch_size;
  uint64_t patch_dim = 3 * patch_size * patch_size;
  patches.clear();
  patches.resize(static_cast<size_t>(patch_rows) * static_cast<size_t>(patch_cols) * static_cast<size_t>(patch_dim));
  size_t write = 0;

  for (uint64_t py = 0; py < patch_rows; ++py) {
    for (uint64_t px = 0; px < patch_cols; ++px) {
      for (uint64_t c = 0; c < 3; ++c) {
        for (uint64_t dy = 0; dy < patch_size; ++dy) {
          for (uint64_t dx = 0; dx < patch_size; ++dx) {
            uint64_t y = py * patch_size + dy;
            uint64_t x = px * patch_size + dx;
            size_t pix =
                (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3 +
                static_cast<size_t>(c);
            patches[write++] = rgb[pix];
          }
        }
      }
    }
  }
}

static bool load_video_patchified_streaming(
    const std::string& path,
    uint64_t in_channels,
    uint64_t patch_size,
    uint64_t temporal_patch_size,
    uint64_t max_frames,
    uint64_t frame_stride,
    std::vector<float>& patches,
    uint64_t& n_patches,
    std::string& err) {
  @autoreleasepool {
    if (frame_stride == 0) {
      err = "frame_stride must be > 0";
      return false;
    }
    if (temporal_patch_size == 0) {
      err = "temporal_patch_size must be > 0";
      return false;
    }

    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    if (url == nil) {
      err = "Invalid video path URL";
      return false;
    }

    AVURLAsset* asset = [AVURLAsset URLAssetWithURL:url options:nil];
    if (asset == nil) {
      err = "Failed to open video asset";
      return false;
    }

    NSArray<AVAssetTrack*>* tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
    if (tracks.count == 0) {
      err = "No video track found";
      return false;
    }
    AVAssetTrack* track = tracks.firstObject;

    NSError* nsErr = nil;
    AVAssetReader* reader = [[AVAssetReader alloc] initWithAsset:asset error:&nsErr];
    if (reader == nil) {
      err = nsErr != nil ? std::string([[nsErr localizedDescription] UTF8String]) : "Failed to create AVAssetReader";
      return false;
    }

    NSDictionary* outputSettings = @{
      (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)
    };
    AVAssetReaderTrackOutput* output = [[AVAssetReaderTrackOutput alloc] initWithTrack:track outputSettings:outputSettings];
    output.alwaysCopiesSampleData = NO;
    if (![reader canAddOutput:output]) {
      err = "Cannot add AVAssetReader track output";
      return false;
    }
    [reader addOutput:output];

    if (![reader startReading]) {
      err = "AVAssetReader failed to start";
      return false;
    }

    int width = 0;
    int height = 0;
    bool have_grid = false;
    PatchGrid grid;
    uint64_t decoded_frames = 0;
    uint64_t kept_frames = 0;
    uint64_t groups = 0;
    patches.clear();

    std::vector<std::vector<float>> temporal_group;
    temporal_group.reserve(static_cast<size_t>(temporal_patch_size));
    std::vector<float> last_frame;

    while (reader.status == AVAssetReaderStatusReading) {
      CMSampleBufferRef sample = [output copyNextSampleBuffer];
      ScopedCFRelease sample_owner(sample);
      if (sample == nullptr) {
        break;
      }

      bool keep_frame = (decoded_frames % frame_stride) == 0;
      ++decoded_frames;
      if (!keep_frame) {
        continue;
      }

      CVImageBufferRef imgBuf = CMSampleBufferGetImageBuffer(sample);
      if (imgBuf == nullptr) {
        continue;
      }
      CVPixelBufferRef px = (CVPixelBufferRef)imgBuf;
      ScopedPixelBufferLock pixel_lock(px);
      if (!pixel_lock.locked()) {
        err = "Failed to lock video pixel buffer";
        return false;
      }

      int fw = static_cast<int>(CVPixelBufferGetWidth(px));
      int fh = static_cast<int>(CVPixelBufferGetHeight(px));
      if (fw <= 0 || fh <= 0) {
        continue;
      }

      if (width == 0 && height == 0) {
        width = fw;
        height = fh;
      }
      if (fw != width || fh != height) {
        err = "Video frames have inconsistent dimensions";
        return false;
      }

      uint8_t* base = static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(px));
      size_t bpr = CVPixelBufferGetBytesPerRow(px);
      std::vector<float> rgb(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
      for (int y = 0; y < height; ++y) {
        uint8_t* row = base + static_cast<size_t>(y) * bpr;
        for (int x = 0; x < width; ++x) {
          uint8_t b = row[x * 4 + 0];
          uint8_t g = row[x * 4 + 1];
          uint8_t r = row[x * 4 + 2];
          size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3;
          rgb[idx + 0] = static_cast<float>(r) / 255.0f;
          rgb[idx + 1] = static_cast<float>(g) / 255.0f;
          rgb[idx + 2] = static_cast<float>(b) / 255.0f;
        }
      }

      pixel_lock.reset();
      sample_owner.reset();

      if (!have_grid) {
        if (!build_patch_grid(width, height, in_channels, patch_size, temporal_patch_size, grid, err)) {
          return false;
        }
        have_grid = true;
      }

      temporal_group.push_back(std::move(rgb));
      last_frame = temporal_group.back();
      ++kept_frames;

      if (temporal_group.size() == static_cast<size_t>(temporal_patch_size)) {
        std::vector<const std::vector<float>*> group_frames;
        group_frames.reserve(static_cast<size_t>(temporal_patch_size));
        for (const auto& fr : temporal_group) {
          group_frames.push_back(&fr);
        }
        append_patch_group(group_frames, width, in_channels, patch_size, temporal_patch_size, grid, patches);
        temporal_group.clear();
        ++groups;
      }

      if (max_frames > 0 && kept_frames >= max_frames) {
        break;
      }
    }

    if (reader.status == AVAssetReaderStatusFailed) {
      err = "AVAssetReader failed while decoding video";
      return false;
    }

    if (kept_frames == 0) {
      err = "No decodable frames found in video";
      return false;
    }

    if (!temporal_group.empty()) {
      while (temporal_group.size() < static_cast<size_t>(temporal_patch_size)) {
        temporal_group.push_back(last_frame);
      }
      std::vector<const std::vector<float>*> group_frames;
      group_frames.reserve(static_cast<size_t>(temporal_patch_size));
      for (const auto& fr : temporal_group) {
        group_frames.push_back(&fr);
      }
      append_patch_group(group_frames, width, in_channels, patch_size, temporal_patch_size, grid, patches);
      ++groups;
    }

    n_patches = groups * grid.patches_per_group;
  }
  return true;
}

extern "C" {

lean_object* lean_torch_media_load_image_patchified(
    b_lean_obj_arg path_obj,
    b_lean_obj_arg in_channels_obj,
    b_lean_obj_arg patch_size_obj,
    b_lean_obj_arg temporal_patch_size_obj) {
  return tyr::checkedIoCall("loadImagePatchified", [&] {
  const uint64_t in_channels = tyr::natTo<int>(in_channels_obj, "in_channels");
  const uint64_t patch_size = tyr::natTo<int>(patch_size_obj, "patch_size");
  const uint64_t temporal_patch_size = tyr::natTo<int>(temporal_patch_size_obj, "temporal_patch_size");
  tyr::checkedProduct<int64_t>(
      tyr::checkedProduct<int64_t>(in_channels, temporal_patch_size, "in_channels * temporal_patch_size"),
      tyr::checkedProduct<int64_t>(patch_size, patch_size, "patch_size * patch_size"), "patch_dim");
  const char* path_c = lean_string_cstr(path_obj);
  std::string path(path_c);

  int width = 0;
  int height = 0;
  std::vector<float> rgb;
  std::string err;
  if (!load_image_rgb_f32(path, width, height, rgb, err)) {
    return mk_io_error("loadImagePatchified failed: " + err);
  }

  std::vector<std::vector<float>> frames;
  frames.push_back(std::move(rgb));

  std::vector<float> patches;
  uint64_t n_patches = 0;
  if (!patchify_rgb_frames(
        frames, width, height, in_channels, patch_size, temporal_patch_size,
        patches, n_patches, err)) {
    return mk_io_error("loadImagePatchified failed: " + err);
  }

  uint64_t patch_dim = in_channels * temporal_patch_size * patch_size * patch_size;
  auto t = torch::from_blob(
      patches.data(),
      {static_cast<int64_t>(n_patches), static_cast<int64_t>(patch_dim)},
      torch::TensorOptions().dtype(torch::kFloat32)).clone();
  return lean_io_result_mk_ok(fromTorchTensor(t));
  });
}

lean_object* lean_torch_media_load_video_patchified(
    b_lean_obj_arg path_obj,
    b_lean_obj_arg in_channels_obj,
    b_lean_obj_arg patch_size_obj,
    b_lean_obj_arg temporal_patch_size_obj,
    b_lean_obj_arg max_frames_obj,
    b_lean_obj_arg frame_stride_obj) {
  return tyr::checkedIoCall("loadVideoPatchified", [&] {
  const uint64_t in_channels = tyr::natTo<int>(in_channels_obj, "in_channels");
  const uint64_t patch_size = tyr::natTo<int>(patch_size_obj, "patch_size");
  const uint64_t temporal_patch_size = tyr::natTo<int>(temporal_patch_size_obj, "temporal_patch_size");
  tyr::checkedProduct<int64_t>(
      tyr::checkedProduct<int64_t>(in_channels, temporal_patch_size, "in_channels * temporal_patch_size"),
      tyr::checkedProduct<int64_t>(patch_size, patch_size, "patch_size * patch_size"), "patch_dim");
  const uint64_t max_frames = tyr::natTo<int>(max_frames_obj, "max_frames");
  const uint64_t frame_stride = tyr::natTo<int>(frame_stride_obj, "frame_stride");
  const char* path_c = lean_string_cstr(path_obj);
  std::string path(path_c);

  std::vector<float> patches;
  uint64_t n_patches = 0;
  std::string err;
  if (!load_video_patchified_streaming(
        path, in_channels, patch_size, temporal_patch_size,
        max_frames, frame_stride, patches, n_patches, err)) {
    return mk_io_error("loadVideoPatchified failed: " + err);
  }

  uint64_t patch_dim = in_channels * temporal_patch_size * patch_size * patch_size;
  auto t = torch::from_blob(
      patches.data(),
      {static_cast<int64_t>(n_patches), static_cast<int64_t>(patch_dim)},
      torch::TensorOptions().dtype(torch::kFloat32)).clone();
  return lean_io_result_mk_ok(fromTorchTensor(t));
  });
}

LEAN_EXPORT lean_object* lean_torch_media_load_image_patch_grid_gemma4(
    b_lean_obj_arg path_obj,
    b_lean_obj_arg patch_size_obj,
    b_lean_obj_arg pooling_kernel_size_obj,
    b_lean_obj_arg max_soft_tokens_obj,
    double rescale_factor) {
  return tyr::checkedIoCall("loadGemma4ImagePatchGrid", [&] {
  const uint64_t patch_size = tyr::natTo<int>(patch_size_obj, "patch_size");
  const uint64_t pooling_kernel_size = tyr::natTo<int>(pooling_kernel_size_obj, "pooling_kernel_size");
  const uint64_t max_soft_tokens = tyr::natTo<int>(max_soft_tokens_obj, "max_soft_tokens");
  tyr::checkedProduct<int>(max_soft_tokens,
      tyr::checkedProduct<int>(pooling_kernel_size, patch_size, "pooling_kernel_size * patch_size"),
      "max_soft_tokens * pooling_kernel_size * patch_size");
  tyr::checkedProduct<int64_t>(3, tyr::checkedProduct<int64_t>(patch_size, patch_size, "patch_size * patch_size"),
      "patch_dim");
  const char* path_c = lean_string_cstr(path_obj);
  std::string path(path_c);

  if (patch_size == 0 || pooling_kernel_size == 0 || max_soft_tokens == 0) {
    return mk_io_error("loadGemma4ImagePatchGrid failed: patch_size, pooling_kernel_size, and max_soft_tokens must be > 0");
  }

  int width = 0;
  int height = 0;
  std::vector<float> rgb;
  std::string err;
  if (!load_image_rgb_f32(path, width, height, rgb, err)) {
    return mk_io_error("loadGemma4ImagePatchGrid failed: " + err);
  }

  int target_h = 0;
  int target_w = 0;
  if (!compute_gemma4_target_size(
        width, height, patch_size, pooling_kernel_size, max_soft_tokens, target_h, target_w, err)) {
    return mk_io_error("loadGemma4ImagePatchGrid failed: " + err);
  }

  uint64_t patch_rows = static_cast<uint64_t>(target_h) / patch_size;
  uint64_t patch_cols = static_cast<uint64_t>(target_w) / patch_size;
  if (patch_rows == 0 || patch_cols == 0) {
    return mk_io_error("loadGemma4ImagePatchGrid failed: computed empty patch grid");
  }

  std::vector<float> resized;
  resize_rgb_bilinear(rgb, width, height, target_w, target_h, resized);

  const float rescale = static_cast<float>(rescale_factor * 255.0);
  if (std::abs(rescale - 1.0f) > 1e-6f) {
    for (float& v : resized) {
      v *= rescale;
    }
  }

  std::vector<float> patches;
  patchify_resized_image(resized, target_w, target_h, patch_size, patches);

  uint64_t patch_dim = 3 * patch_size * patch_size;
  auto t = torch::from_blob(
      patches.data(),
      {
        static_cast<int64_t>(patch_rows),
        static_cast<int64_t>(patch_cols),
        static_cast<int64_t>(patch_dim)
      },
      torch::TensorOptions().dtype(torch::kFloat32)).clone();
  return lean_io_result_mk_ok(fromTorchTensor(t));
  });
}

} // extern "C"

#else

extern "C" {

lean_object* lean_torch_media_load_image_patchified(
    b_lean_obj_arg /*path_obj*/,
    b_lean_obj_arg /*in_channels*/,
    b_lean_obj_arg /*patch_size*/,
    b_lean_obj_arg /*temporal_patch_size*/) {
  return lean_io_result_mk_error(lean_mk_io_user_error(
    lean_mk_string("Apple media path requires macOS build")));
}

lean_object* lean_torch_media_load_video_patchified(
    b_lean_obj_arg /*path_obj*/,
    b_lean_obj_arg /*in_channels*/,
    b_lean_obj_arg /*patch_size*/,
    b_lean_obj_arg /*temporal_patch_size*/,
    b_lean_obj_arg /*max_frames*/,
    b_lean_obj_arg /*frame_stride*/) {
  return lean_io_result_mk_error(lean_mk_io_user_error(
    lean_mk_string("Apple media path requires macOS build")));
}

} // extern "C"

#endif
