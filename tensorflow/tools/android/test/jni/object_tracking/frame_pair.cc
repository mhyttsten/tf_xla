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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSframe_pairDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSframe_pairDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSframe_pairDTcc() {
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

#include "tensorflow/tools/android/test/jni/object_tracking/frame_pair.h"

#include <float.h>

#include "tensorflow/tools/android/test/jni/object_tracking/config.h"

namespace tf_tracking {

void FramePair::Init(const int64_t start_time, const int64_t end_time) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSframe_pairDTcc mht_0(mht_0_v, 193, "", "./tensorflow/tools/android/test/jni/object_tracking/frame_pair.cc", "FramePair::Init");

  start_time_ = start_time;
  end_time_ = end_time;
  memset(optical_flow_found_keypoint_, false,
         sizeof(*optical_flow_found_keypoint_) * kMaxKeypoints);
  number_of_keypoints_ = 0;
}

void FramePair::AdjustBox(const BoundingBox box,
                          float* const translation_x,
                          float* const translation_y,
                          float* const scale_x,
                          float* const scale_y) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSframe_pairDTcc mht_1(mht_1_v, 208, "", "./tensorflow/tools/android/test/jni/object_tracking/frame_pair.cc", "FramePair::AdjustBox");

  static float weights[kMaxKeypoints];
  static Point2f deltas[kMaxKeypoints];
  memset(weights, 0.0f, sizeof(*weights) * kMaxKeypoints);

  BoundingBox resized_box(box);
  resized_box.Scale(0.4f, 0.4f);
  FillWeights(resized_box, weights);
  FillTranslations(deltas);

  const Point2f translation = GetWeightedMedian(weights, deltas);

  *translation_x = translation.x;
  *translation_y = translation.y;

  const Point2f old_center = box.GetCenter();
  const int good_scale_points =
      FillScales(old_center, translation, weights, deltas);

  // Default scale factor is 1 for x and y.
  *scale_x = 1.0f;
  *scale_y = 1.0f;

  // The assumption is that all deltas that make it to this stage with a
  // corresponding optical_flow_found_keypoint_[i] == true are not in
  // themselves degenerate.
  //
  // The degeneracy with scale arose because if the points are too close to the
  // center of the objects, the scale ratio determination might be incalculable.
  //
  // The check for kMinNumInRange is not a degeneracy check, but merely an
  // attempt to ensure some sort of stability. The actual degeneracy check is in
  // the comparison to EPSILON in FillScales (which I've updated to return the
  // number good remaining as well).
  static const int kMinNumInRange = 5;
  if (good_scale_points >= kMinNumInRange) {
    const float scale_factor = GetWeightedMedianScale(weights, deltas);

    if (scale_factor > 0.0f) {
      *scale_x = scale_factor;
      *scale_y = scale_factor;
    }
  }
}

int FramePair::FillWeights(const BoundingBox& box,
                           float* const weights) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSframe_pairDTcc mht_2(mht_2_v, 257, "", "./tensorflow/tools/android/test/jni/object_tracking/frame_pair.cc", "FramePair::FillWeights");

  // Compute the max score.
  float max_score = -FLT_MAX;
  float min_score = FLT_MAX;
  for (int i = 0; i < kMaxKeypoints; ++i) {
    if (optical_flow_found_keypoint_[i]) {
      max_score = MAX(max_score, frame1_keypoints_[i].score_);
      min_score = MIN(min_score, frame1_keypoints_[i].score_);
    }
  }

  int num_in_range = 0;
  for (int i = 0; i < kMaxKeypoints; ++i) {
    if (!optical_flow_found_keypoint_[i]) {
      weights[i] = 0.0f;
      continue;
    }

    const bool in_box = box.Contains(frame1_keypoints_[i].pos_);
    if (in_box) {
      ++num_in_range;
    }

    // The weighting based off distance.  Anything within the bounding box
    // has a weight of 1, and everything outside of that is within the range
    // [0, kOutOfBoxMultiplier), falling off with the squared distance ratio.
    float distance_score = 1.0f;
    if (!in_box) {
      const Point2f initial = box.GetCenter();
      const float sq_x_dist =
          Square(initial.x - frame1_keypoints_[i].pos_.x);
      const float sq_y_dist =
          Square(initial.y - frame1_keypoints_[i].pos_.y);
      const float squared_half_width = Square(box.GetWidth() / 2.0f);
      const float squared_half_height = Square(box.GetHeight() / 2.0f);

      static const float kOutOfBoxMultiplier = 0.5f;
      distance_score = kOutOfBoxMultiplier *
          MIN(squared_half_height / sq_y_dist, squared_half_width / sq_x_dist);
    }

    // The weighting based on relative score strength. kBaseScore - 1.0f.
    float intrinsic_score =  1.0f;
    if (max_score > min_score) {
      static const float kBaseScore = 0.5f;
      intrinsic_score = ((frame1_keypoints_[i].score_ - min_score) /
         (max_score - min_score)) * (1.0f - kBaseScore) + kBaseScore;
    }

    // The final score will be in the range [0, 1].
    weights[i] = distance_score * intrinsic_score;
  }

  return num_in_range;
}

void FramePair::FillTranslations(Point2f* const translations) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSframe_pairDTcc mht_3(mht_3_v, 316, "", "./tensorflow/tools/android/test/jni/object_tracking/frame_pair.cc", "FramePair::FillTranslations");

  for (int i = 0; i < kMaxKeypoints; ++i) {
    if (!optical_flow_found_keypoint_[i]) {
      continue;
    }
    translations[i].x =
        frame2_keypoints_[i].pos_.x - frame1_keypoints_[i].pos_.x;
    translations[i].y =
        frame2_keypoints_[i].pos_.y - frame1_keypoints_[i].pos_.y;
  }
}

int FramePair::FillScales(const Point2f& old_center,
                          const Point2f& translation,
                          float* const weights,
                          Point2f* const scales) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSframe_pairDTcc mht_4(mht_4_v, 334, "", "./tensorflow/tools/android/test/jni/object_tracking/frame_pair.cc", "FramePair::FillScales");

  int num_good = 0;
  for (int i = 0; i < kMaxKeypoints; ++i) {
    if (!optical_flow_found_keypoint_[i]) {
      continue;
    }

    const Keypoint keypoint1 = frame1_keypoints_[i];
    const Keypoint keypoint2 = frame2_keypoints_[i];

    const float dist1_x = keypoint1.pos_.x - old_center.x;
    const float dist1_y = keypoint1.pos_.y - old_center.y;

    const float dist2_x = (keypoint2.pos_.x - translation.x) - old_center.x;
    const float dist2_y = (keypoint2.pos_.y - translation.y) - old_center.y;

    // Make sure that the scale makes sense; points too close to the center
    // will result in either NaNs or infinite results for scale due to
    // limited tracking and floating point resolution.
    // Also check that the parity of the points is the same with respect to
    // x and y, as we can't really make sense of data that has flipped.
    if (((dist2_x > EPSILON && dist1_x > EPSILON) ||
         (dist2_x < -EPSILON && dist1_x < -EPSILON)) &&
         ((dist2_y > EPSILON && dist1_y > EPSILON) ||
          (dist2_y < -EPSILON && dist1_y < -EPSILON))) {
      scales[i].x = dist2_x / dist1_x;
      scales[i].y = dist2_y / dist1_y;
      ++num_good;
    } else {
      weights[i] = 0.0f;
      scales[i].x = 1.0f;
      scales[i].y = 1.0f;
    }
  }
  return num_good;
}

struct WeightedDelta {
  float weight;
  float delta;
};

// Sort by delta, not by weight.
inline int WeightedDeltaCompare(const void* const a, const void* const b) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSframe_pairDTcc mht_5(mht_5_v, 380, "", "./tensorflow/tools/android/test/jni/object_tracking/frame_pair.cc", "WeightedDeltaCompare");

  return (reinterpret_cast<const WeightedDelta*>(a)->delta -
          reinterpret_cast<const WeightedDelta*>(b)->delta) <= 0 ? 1 : -1;
}

// Returns the median delta from a sorted set of weighted deltas.
static float GetMedian(const int num_items,
                       const WeightedDelta* const weighted_deltas,
                       const float sum) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSframe_pairDTcc mht_6(mht_6_v, 391, "", "./tensorflow/tools/android/test/jni/object_tracking/frame_pair.cc", "GetMedian");

  if (num_items == 0 || sum < EPSILON) {
    return 0.0f;
  }

  float current_weight = 0.0f;
  const float target_weight = sum / 2.0f;
  for (int i = 0; i < num_items; ++i) {
    if (weighted_deltas[i].weight > 0.0f) {
      current_weight += weighted_deltas[i].weight;
      if (current_weight >= target_weight) {
        return weighted_deltas[i].delta;
      }
    }
  }
  LOGW("Median not found! %d points, sum of %.2f", num_items, sum);
  return 0.0f;
}

Point2f FramePair::GetWeightedMedian(
    const float* const weights, const Point2f* const deltas) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSframe_pairDTcc mht_7(mht_7_v, 414, "", "./tensorflow/tools/android/test/jni/object_tracking/frame_pair.cc", "FramePair::GetWeightedMedian");

  Point2f median_delta;

  // TODO(andrewharp): only sort deltas that could possibly have an effect.
  static WeightedDelta weighted_deltas[kMaxKeypoints];

  // Compute median X value.
  {
    float total_weight = 0.0f;

    // Compute weighted mean and deltas.
    for (int i = 0; i < kMaxKeypoints; ++i) {
      weighted_deltas[i].delta = deltas[i].x;
      const float weight = weights[i];
      weighted_deltas[i].weight = weight;
      if (weight > 0.0f) {
        total_weight += weight;
      }
    }
    qsort(weighted_deltas, kMaxKeypoints, sizeof(WeightedDelta),
          WeightedDeltaCompare);
    median_delta.x = GetMedian(kMaxKeypoints, weighted_deltas, total_weight);
  }

  // Compute median Y value.
  {
    float total_weight = 0.0f;

    // Compute weighted mean and deltas.
    for (int i = 0; i < kMaxKeypoints; ++i) {
      const float weight = weights[i];
      weighted_deltas[i].weight = weight;
      weighted_deltas[i].delta = deltas[i].y;
      if (weight > 0.0f) {
        total_weight += weight;
      }
    }
    qsort(weighted_deltas, kMaxKeypoints, sizeof(WeightedDelta),
          WeightedDeltaCompare);
    median_delta.y = GetMedian(kMaxKeypoints, weighted_deltas, total_weight);
  }

  return median_delta;
}

float FramePair::GetWeightedMedianScale(
    const float* const weights, const Point2f* const deltas) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSframe_pairDTcc mht_8(mht_8_v, 463, "", "./tensorflow/tools/android/test/jni/object_tracking/frame_pair.cc", "FramePair::GetWeightedMedianScale");

  float median_delta;

  // TODO(andrewharp): only sort deltas that could possibly have an effect.
  static WeightedDelta weighted_deltas[kMaxKeypoints * 2];

  // Compute median scale value across x and y.
  {
    float total_weight = 0.0f;

    // Add X values.
    for (int i = 0; i < kMaxKeypoints; ++i) {
      weighted_deltas[i].delta = deltas[i].x;
      const float weight = weights[i];
      weighted_deltas[i].weight = weight;
      if (weight > 0.0f) {
        total_weight += weight;
      }
    }

    // Add Y values.
    for (int i = 0; i < kMaxKeypoints; ++i) {
      weighted_deltas[i + kMaxKeypoints].delta = deltas[i].y;
      const float weight = weights[i];
      weighted_deltas[i + kMaxKeypoints].weight = weight;
      if (weight > 0.0f) {
        total_weight += weight;
      }
    }

    qsort(weighted_deltas, kMaxKeypoints * 2, sizeof(WeightedDelta),
          WeightedDeltaCompare);

    median_delta = GetMedian(kMaxKeypoints * 2, weighted_deltas, total_weight);
  }

  return median_delta;
}

}  // namespace tf_tracking
