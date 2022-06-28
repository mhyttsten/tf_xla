/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_FLOW_CACHE_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_FLOW_CACHE_H_
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSflow_cacheDTh {
public:
   std::string _s;
   int _indent = 0;
   std::string _functionName;
   bool _isFile = false;
   std::string _fileName;
   std::string _envMHIndent;
   int _lineNumber;
   bool _filtered = false;
   bool _otherThread = false;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSflow_cacheDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
      _functionName = functionName;
      _lineNumber = lineNumber;

      // Check if tracing is enabled
      const char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }
      // Should we trace of filter?
      const char* env_filter = std::getenv("MHTRACER_FILTER");
      if (env_filter != nullptr) {
         std::string sfilter = std::string(env_filter);
         std::string sLineNumber = std::to_string(lineNumber);
         while (true) {
            std::size_t ioE = sfilter.find(";");
            if (sfilter.size() == 0) {
               break;
            }
            std::string cfs = sfilter.substr(0, ioE);
            std::size_t ioFileName = cfs.find("|");
            std::string fFileName  = cfs.substr(0, ioFileName);
            std::size_t ioFunctionName = cfs.find("|", ioFileName+1);
            std::string fFunctionName  = cfs.substr(ioFileName+1, ioFunctionName-ioFileName-1);
            std::string fLineNumber    = cfs.substr(ioFunctionName+1, cfs.size()-ioFunctionName-1);

            if (  (fFileName == "*" || fFileName == fileName)
               && (fFunctionName == "*" || fFunctionName == functionName)
               && (fLineNumber == "*" || fLineNumber == sLineNumber)) {
              _filtered = true;
               return;
            }

            if (ioE == std::string::npos) {
               sfilter = "";
            } else {
               sfilter = sfilter.substr(ioE+1, sfilter.size()-ioE-1);
            }
         }
      }

      // Create log string
      std::string ostr;

      // Assign indent spaces (tied to PID and TID)
      pid_t pid = getpid();
      std::thread::id tid = std::this_thread::get_id();
      std::stringstream pid_dash_tid_ss;
      pid_dash_tid_ss << pid << "-" << tid;
      std::string pid_dash_tid_str = pid_dash_tid_ss.str();
      _envMHIndent = "MHTRACER_INDENT_";
      char* env_indent = std::getenv(_envMHIndent.c_str());
      if (env_indent != nullptr) {
         _indent = std::stoi(std::string(env_indent));
      }
      _s.assign(_indent, ' ');

      // Check that reporting matches pid/tid
      const char* env_pid_dash_tid = std::getenv("MHTRACER_PID_DASH_TID");
      if (env_pid_dash_tid != nullptr) {
         std::string env_pid_dash_tid_str(env_pid_dash_tid);
         if (env_pid_dash_tid_str != pid_dash_tid_str) {
            _otherThread = true;
         }
      }
      else {  // PID-THREAD not set, set it for the first time (starter thread)
         setenv("MHTRACER_PID_DASH_TID", pid_dash_tid_str.c_str(), 1);
      }

      std::string paramStr;
      for (int i=0; i < params.size(); i++) {
         auto e = params[i];
         while (e.find("\n") != std::string::npos) {
            size_t pos = e.find("\n");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<NL>");
         }
         while (e.find("[") != std::string::npos) {
            size_t pos = e.find("[");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<LB>");
         }
         while (e.find("]") != std::string::npos) {
            size_t pos = e.find("]");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<RB>");
         }
         paramStr += e;
         if ((i+1) < params.size()) {
            paramStr += ", ";
         }
      }

      const char* env_dont_print_pid_dash_tid = std::getenv("MHTRACER_DONT_PRINT_PID_DASH_TID");
      if (env_dont_print_pid_dash_tid != nullptr) {
         pid_dash_tid_str = "";
      }
      if (_otherThread) {
         functionName = "MHOT_" + functionName;
      }
      ostr += _s + functionName + 
         + " [1]"
         + " [" + prefix + "]"
         + " [" + paramStr + "]"
         + " [" + pid_dash_tid_str + " "
         +    std::to_string(lineNumber)
         +    " @ " + fileName + "]\n";

      // Log to file
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_USEFILE") != std::string::npos) {
         _isFile = true;
         _fileName = "/tmp/mhtracer_" + pid_dash_tid_str + ".log";
         std::ofstream os;
         os.open(_fileName, std::ofstream::out | std::ofstream::app);
         os << ostr << "";
         os.close();
      }
      // Log to stdout
      else {
         std::cout << ostr << "";
      }

      // Increment indent spaces
      if (_otherThread) {
         return;
      }
      _indent += 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSflow_cacheDTh() {
      // Check if tracing is enabled
      char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }

      // Don't update indent if tracing was filtered or from another thread
      if (_filtered || _otherThread) {
         return;
      }

      _indent -= 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
};


#include "tensorflow/tools/android/test/jni/object_tracking/config.h"
#include "tensorflow/tools/android/test/jni/object_tracking/geom.h"
#include "tensorflow/tools/android/test/jni/object_tracking/optical_flow.h"
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

namespace tf_tracking {

// Class that helps OpticalFlow to speed up flow computation
// by caching coarse-grained flow.
class FlowCache {
 public:
  explicit FlowCache(const OpticalFlowConfig* const config)
      : config_(config),
        image_size_(config->image_size),
        optical_flow_(config),
        fullframe_matrix_(NULL) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSflow_cacheDTh mht_0(mht_0_v, 203, "", "./tensorflow/tools/android/test/jni/object_tracking/flow_cache.h", "FlowCache");

    for (int i = 0; i < kNumCacheLevels; ++i) {
      const int curr_dims = BlockDimForCacheLevel(i);
      has_cache_[i] = new Image<bool>(curr_dims, curr_dims);
      displacements_[i] = new Image<Point2f>(curr_dims, curr_dims);
    }
  }

  ~FlowCache() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSflow_cacheDTh mht_1(mht_1_v, 214, "", "./tensorflow/tools/android/test/jni/object_tracking/flow_cache.h", "~FlowCache");

    for (int i = 0; i < kNumCacheLevels; ++i) {
      SAFE_DELETE(has_cache_[i]);
      SAFE_DELETE(displacements_[i]);
    }
    delete[](fullframe_matrix_);
    fullframe_matrix_ = NULL;
  }

  void NextFrame(ImageData* const new_frame,
                 const float* const align_matrix23) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSflow_cacheDTh mht_2(mht_2_v, 227, "", "./tensorflow/tools/android/test/jni/object_tracking/flow_cache.h", "NextFrame");

    ClearCache();
    SetFullframeAlignmentMatrix(align_matrix23);
    optical_flow_.NextFrame(new_frame);
  }

  void ClearCache() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSflow_cacheDTh mht_3(mht_3_v, 236, "", "./tensorflow/tools/android/test/jni/object_tracking/flow_cache.h", "ClearCache");

    for (int i = 0; i < kNumCacheLevels; ++i) {
      has_cache_[i]->Clear(false);
    }
    delete[](fullframe_matrix_);
    fullframe_matrix_ = NULL;
  }

  // Finds the flow at a point, using the cache for performance.
  bool FindFlowAtPoint(const float u_x, const float u_y,
                       float* const flow_x, float* const flow_y) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSflow_cacheDTh mht_4(mht_4_v, 249, "", "./tensorflow/tools/android/test/jni/object_tracking/flow_cache.h", "FindFlowAtPoint");

    // Get the best guess from the cache.
    const Point2f guess_from_cache = LookupGuess(u_x, u_y);

    *flow_x = guess_from_cache.x;
    *flow_y = guess_from_cache.y;

    // Now refine the guess using the image pyramid.
    for (int pyramid_level = kMinNumPyramidLevelsToUseForAdjustment - 1;
        pyramid_level >= 0; --pyramid_level) {
      if (!optical_flow_.FindFlowAtPointSingleLevel(
          pyramid_level, u_x, u_y, false, flow_x, flow_y)) {
        return false;
      }
    }

    return true;
  }

  // Determines the displacement of a point, and uses that to calculate a new
  // position.
  // Returns true iff the displacement determination worked and the new position
  // is in the image.
  bool FindNewPositionOfPoint(const float u_x, const float u_y,
                              float* final_x, float* final_y) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSflow_cacheDTh mht_5(mht_5_v, 276, "", "./tensorflow/tools/android/test/jni/object_tracking/flow_cache.h", "FindNewPositionOfPoint");

    float flow_x;
    float flow_y;
    if (!FindFlowAtPoint(u_x, u_y, &flow_x, &flow_y)) {
      return false;
    }

    // Add in the displacement to get the final position.
    *final_x = u_x + flow_x;
    *final_y = u_y + flow_y;

    // Assign the best guess, if we're still in the image.
    if (InRange(*final_x, 0.0f, static_cast<float>(image_size_.width) - 1) &&
        InRange(*final_y, 0.0f, static_cast<float>(image_size_.height) - 1)) {
      return true;
    } else {
      return false;
    }
  }

  // Comparison function for qsort.
  static int Compare(const void* a, const void* b) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSflow_cacheDTh mht_6(mht_6_v, 300, "", "./tensorflow/tools/android/test/jni/object_tracking/flow_cache.h", "Compare");

    return *reinterpret_cast<const float*>(a) -
           *reinterpret_cast<const float*>(b);
  }

  // Returns the median flow within the given bounding box as determined
  // by a grid_width x grid_height grid.
  Point2f GetMedianFlow(const BoundingBox& bounding_box,
                        const bool filter_by_fb_error,
                        const int grid_width,
                        const int grid_height) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSflow_cacheDTh mht_7(mht_7_v, 313, "", "./tensorflow/tools/android/test/jni/object_tracking/flow_cache.h", "GetMedianFlow");

    const int kMaxPoints = 100;
    SCHECK(grid_width * grid_height <= kMaxPoints,
          "Too many points for Median flow!");

    const BoundingBox valid_box = bounding_box.Intersect(
        BoundingBox(0, 0, image_size_.width - 1, image_size_.height - 1));

    if (valid_box.GetArea() <= 0.0f) {
      return Point2f(0, 0);
    }

    float x_deltas[kMaxPoints];
    float y_deltas[kMaxPoints];

    int curr_offset = 0;
    for (int i = 0; i < grid_width; ++i) {
      for (int j = 0; j < grid_height; ++j) {
        const float x_in = valid_box.left_ +
            (valid_box.GetWidth() * i) / (grid_width - 1);

        const float y_in = valid_box.top_ +
            (valid_box.GetHeight() * j) / (grid_height - 1);

        float curr_flow_x;
        float curr_flow_y;
        const bool success = FindNewPositionOfPoint(x_in, y_in,
                                                    &curr_flow_x, &curr_flow_y);

        if (success) {
          x_deltas[curr_offset] = curr_flow_x;
          y_deltas[curr_offset] = curr_flow_y;
          ++curr_offset;
        } else {
          LOGW("Tracking failure!");
        }
      }
    }

    if (curr_offset > 0) {
      qsort(x_deltas, curr_offset, sizeof(*x_deltas), Compare);
      qsort(y_deltas, curr_offset, sizeof(*y_deltas), Compare);

      return Point2f(x_deltas[curr_offset / 2], y_deltas[curr_offset / 2]);
    }

    LOGW("No points were valid!");
    return Point2f(0, 0);
  }

  void SetFullframeAlignmentMatrix(const float* const align_matrix23) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSflow_cacheDTh mht_8(mht_8_v, 366, "", "./tensorflow/tools/android/test/jni/object_tracking/flow_cache.h", "SetFullframeAlignmentMatrix");

    if (align_matrix23 != NULL) {
      if (fullframe_matrix_ == NULL) {
        fullframe_matrix_ = new float[6];
      }

      memcpy(fullframe_matrix_, align_matrix23,
             6 * sizeof(fullframe_matrix_[0]));
    }
  }

 private:
  Point2f LookupGuessFromLevel(
      const int cache_level, const float x, const float y) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSflow_cacheDTh mht_9(mht_9_v, 382, "", "./tensorflow/tools/android/test/jni/object_tracking/flow_cache.h", "LookupGuessFromLevel");

    // LOGE("Looking up guess at %5.2f %5.2f for level %d.", x, y, cache_level);

    // Cutoff at the target level and use the matrix transform instead.
    if (fullframe_matrix_ != NULL && cache_level == kCacheCutoff) {
      const float xnew = x * fullframe_matrix_[0] +
                         y * fullframe_matrix_[1] +
                             fullframe_matrix_[2];
      const float ynew = x * fullframe_matrix_[3] +
                         y * fullframe_matrix_[4] +
                             fullframe_matrix_[5];

      return Point2f(xnew - x, ynew - y);
    }

    const int level_dim = BlockDimForCacheLevel(cache_level);
    const int pixels_per_cache_block_x =
        (image_size_.width + level_dim - 1) / level_dim;
    const int pixels_per_cache_block_y =
        (image_size_.height + level_dim - 1) / level_dim;
    const int index_x = x / pixels_per_cache_block_x;
    const int index_y = y / pixels_per_cache_block_y;

    Point2f displacement;
    if (!(*has_cache_[cache_level])[index_y][index_x]) {
      (*has_cache_[cache_level])[index_y][index_x] = true;

      // Get the lower cache level's best guess, if it exists.
      displacement = cache_level >= kNumCacheLevels - 1 ?
          Point2f(0, 0) : LookupGuessFromLevel(cache_level + 1, x, y);
      // LOGI("Best guess at cache level %d is %5.2f, %5.2f.", cache_level,
      //      best_guess.x, best_guess.y);

      // Find the center of the block.
      const float center_x = (index_x + 0.5f) * pixels_per_cache_block_x;
      const float center_y = (index_y + 0.5f) * pixels_per_cache_block_y;
      const int pyramid_level = PyramidLevelForCacheLevel(cache_level);

      // LOGI("cache level %d: [%d, %d (%5.2f / %d, %5.2f / %d)] "
      //      "Querying %5.2f, %5.2f at pyramid level %d, ",
      //      cache_level, index_x, index_y,
      //      x, pixels_per_cache_block_x, y, pixels_per_cache_block_y,
      //      center_x, center_y, pyramid_level);

      // TODO(andrewharp): Turn on FB error filtering.
      const bool success = optical_flow_.FindFlowAtPointSingleLevel(
          pyramid_level, center_x, center_y, false,
          &displacement.x, &displacement.y);

      if (!success) {
        LOGV("Computation of cached value failed for level %d!", cache_level);
      }

      // Store the value for later use.
      (*displacements_[cache_level])[index_y][index_x] = displacement;
    } else {
      displacement = (*displacements_[cache_level])[index_y][index_x];
    }

    // LOGI("Returning %5.2f, %5.2f for level %d",
    //      displacement.x, displacement.y, cache_level);
    return displacement;
  }

  Point2f LookupGuess(const float x, const float y) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSflow_cacheDTh mht_10(mht_10_v, 449, "", "./tensorflow/tools/android/test/jni/object_tracking/flow_cache.h", "LookupGuess");

    if (x < 0 || x >= image_size_.width || y < 0 || y >= image_size_.height) {
      return Point2f(0, 0);
    }

    // LOGI("Looking up guess at %5.2f %5.2f.", x, y);
    if (kNumCacheLevels > 0) {
      return LookupGuessFromLevel(0, x, y);
    } else {
      return Point2f(0, 0);
    }
  }

  // Returns the number of cache bins in each dimension for a given level
  // of the cache.
  int BlockDimForCacheLevel(const int cache_level) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSflow_cacheDTh mht_11(mht_11_v, 467, "", "./tensorflow/tools/android/test/jni/object_tracking/flow_cache.h", "BlockDimForCacheLevel");

    // The highest (coarsest) cache level has a block dim of kCacheBranchFactor,
    // thus if there are 4 cache levels, requesting level 3 (0-based) should
    // return kCacheBranchFactor, level 2 should return kCacheBranchFactor^2,
    // and so on.
    int block_dim = kNumCacheLevels;
    for (int curr_level = kNumCacheLevels - 1; curr_level > cache_level;
        --curr_level) {
      block_dim *= kCacheBranchFactor;
    }
    return block_dim;
  }

  // Returns the level of the image pyramid that a given cache level maps to.
  int PyramidLevelForCacheLevel(const int cache_level) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSflow_cacheDTh mht_12(mht_12_v, 484, "", "./tensorflow/tools/android/test/jni/object_tracking/flow_cache.h", "PyramidLevelForCacheLevel");

    // Higher cache and pyramid levels have smaller dimensions. The highest
    // cache level should refer to the highest image pyramid level. The
    // lower, finer image pyramid levels are uncached (assuming
    // kNumCacheLevels < kNumPyramidLevels).
    return cache_level + (kNumPyramidLevels - kNumCacheLevels);
  }

  const OpticalFlowConfig* const config_;

  const Size image_size_;
  OpticalFlow optical_flow_;

  float* fullframe_matrix_;

  // Whether this value is currently present in the cache.
  Image<bool>* has_cache_[kNumCacheLevels];

  // The cached displacement values.
  Image<Point2f>* displacements_[kNumCacheLevels];

  TF_DISALLOW_COPY_AND_ASSIGN(FlowCache);
};

}  // namespace tf_tracking

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_FLOW_CACHE_H_
