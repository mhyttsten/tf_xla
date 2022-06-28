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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTcc() {
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

// Various keypoint detecting functions.

#include "tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.h"

#include <float.h>

#include "tensorflow/tools/android/test/jni/object_tracking/config.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image-inl.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image.h"
#include "tensorflow/tools/android/test/jni/object_tracking/keypoint.h"
#include "tensorflow/tools/android/test/jni/object_tracking/time_log.h"
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

namespace tf_tracking {

static inline int GetDistSquaredBetween(const int* vec1, const int* vec2) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTcc mht_0(mht_0_v, 200, "", "./tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.cc", "GetDistSquaredBetween");

  return Square(vec1[0] - vec2[0]) + Square(vec1[1] - vec2[1]);
}

void KeypointDetector::ScoreKeypoints(const ImageData& image_data,
                                      const int num_candidates,
                                      Keypoint* const candidate_keypoints) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTcc mht_1(mht_1_v, 209, "", "./tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.cc", "KeypointDetector::ScoreKeypoints");

  const Image<int>& I_x = *image_data.GetSpatialX(0);
  const Image<int>& I_y = *image_data.GetSpatialY(0);

  if (config_->detect_skin) {
    const Image<uint8_t>& u_data = *image_data.GetU();
    const Image<uint8_t>& v_data = *image_data.GetV();

    static const int reference[] = {111, 155};

    // Score all the keypoints.
    for (int i = 0; i < num_candidates; ++i) {
      Keypoint* const keypoint = candidate_keypoints + i;

      const int x_pos = keypoint->pos_.x * 2;
      const int y_pos = keypoint->pos_.y * 2;

      const int curr_color[] = {u_data[y_pos][x_pos], v_data[y_pos][x_pos]};
      keypoint->score_ =
          HarrisFilter(I_x, I_y, keypoint->pos_.x, keypoint->pos_.y) /
          GetDistSquaredBetween(reference, curr_color);
    }
  } else {
    // Score all the keypoints.
    for (int i = 0; i < num_candidates; ++i) {
      Keypoint* const keypoint = candidate_keypoints + i;
      keypoint->score_ =
          HarrisFilter(I_x, I_y, keypoint->pos_.x, keypoint->pos_.y);
    }
  }
}


inline int KeypointCompare(const void* const a, const void* const b) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTcc mht_2(mht_2_v, 245, "", "./tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.cc", "KeypointCompare");

  return (reinterpret_cast<const Keypoint*>(a)->score_ -
          reinterpret_cast<const Keypoint*>(b)->score_) <= 0 ? 1 : -1;
}


// Quicksorts detected keypoints by score.
void KeypointDetector::SortKeypoints(const int num_candidates,
                                   Keypoint* const candidate_keypoints) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTcc mht_3(mht_3_v, 256, "", "./tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.cc", "KeypointDetector::SortKeypoints");

  qsort(candidate_keypoints, num_candidates, sizeof(Keypoint), KeypointCompare);

#ifdef SANITY_CHECKS
  // Verify that the array got sorted.
  float last_score = FLT_MAX;
  for (int i = 0; i < num_candidates; ++i) {
    const float curr_score = candidate_keypoints[i].score_;

    // Scores should be monotonically increasing.
    SCHECK(last_score >= curr_score,
          "Quicksort failure! %d: %.5f > %d: %.5f (%d total)",
          i - 1, last_score, i, curr_score, num_candidates);

    last_score = curr_score;
  }
#endif
}


int KeypointDetector::SelectKeypointsInBox(
    const BoundingBox& box,
    const Keypoint* const candidate_keypoints,
    const int num_candidates,
    const int max_keypoints,
    const int num_existing_keypoints,
    const Keypoint* const existing_keypoints,
    Keypoint* const final_keypoints) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTcc mht_4(mht_4_v, 286, "", "./tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.cc", "KeypointDetector::SelectKeypointsInBox");

  if (max_keypoints <= 0) {
    return 0;
  }

  // This is the distance within which keypoints may be placed to each other
  // within this box, roughly based on the box dimensions.
  const int distance =
      MAX(1, MIN(box.GetWidth(), box.GetHeight()) * kClosestPercent / 2.0f);

  // First, mark keypoints that already happen to be inside this region. Ignore
  // keypoints that are outside it, however close they might be.
  interest_map_->Clear(false);
  for (int i = 0; i < num_existing_keypoints; ++i) {
    const Keypoint& candidate = existing_keypoints[i];

    const int x_pos = candidate.pos_.x;
    const int y_pos = candidate.pos_.y;
    if (box.Contains(candidate.pos_)) {
      MarkImage(x_pos, y_pos, distance, interest_map_.get());
    }
  }

  // Now, go through and check which keypoints will still fit in the box.
  int num_keypoints_selected = 0;
  for (int i = 0; i < num_candidates; ++i) {
    const Keypoint& candidate = candidate_keypoints[i];

    const int x_pos = candidate.pos_.x;
    const int y_pos = candidate.pos_.y;

    if (!box.Contains(candidate.pos_) ||
        !interest_map_->ValidPixel(x_pos, y_pos)) {
      continue;
    }

    if (!(*interest_map_)[y_pos][x_pos]) {
      final_keypoints[num_keypoints_selected++] = candidate;
      if (num_keypoints_selected >= max_keypoints) {
        break;
      }
      MarkImage(x_pos, y_pos, distance, interest_map_.get());
    }
  }
  return num_keypoints_selected;
}


void KeypointDetector::SelectKeypoints(
    const std::vector<BoundingBox>& boxes,
    const Keypoint* const candidate_keypoints,
    const int num_candidates,
    FramePair* const curr_change) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTcc mht_5(mht_5_v, 341, "", "./tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.cc", "KeypointDetector::SelectKeypoints");

  // Now select all the interesting keypoints that fall insider our boxes.
  curr_change->number_of_keypoints_ = 0;
  for (std::vector<BoundingBox>::const_iterator iter = boxes.begin();
      iter != boxes.end(); ++iter) {
    const BoundingBox bounding_box = *iter;

    // Count up keypoints that have already been selected, and fall within our
    // box.
    int num_keypoints_already_in_box = 0;
    for (int i = 0; i < curr_change->number_of_keypoints_; ++i) {
      if (bounding_box.Contains(curr_change->frame1_keypoints_[i].pos_)) {
        ++num_keypoints_already_in_box;
      }
    }

    const int max_keypoints_to_find_in_box =
        MIN(kMaxKeypointsForObject - num_keypoints_already_in_box,
            kMaxKeypoints - curr_change->number_of_keypoints_);

    const int num_new_keypoints_in_box = SelectKeypointsInBox(
        bounding_box,
        candidate_keypoints,
        num_candidates,
        max_keypoints_to_find_in_box,
        curr_change->number_of_keypoints_,
        curr_change->frame1_keypoints_,
        curr_change->frame1_keypoints_ + curr_change->number_of_keypoints_);

    curr_change->number_of_keypoints_ += num_new_keypoints_in_box;

    LOGV("Selected %d keypoints!", curr_change->number_of_keypoints_);
  }
}


// Walks along the given circle checking for pixels above or below the center.
// Returns a score, or 0 if the keypoint did not pass the criteria.
//
// Parameters:
//  circle_perimeter: the circumference in pixels of the circle.
//  threshold: the minimum number of contiguous pixels that must be above or
//             below the center value.
//  center_ptr: the location of the center pixel in memory
//  offsets: the relative offsets from the center pixel of the edge pixels.
inline int TestCircle(const int circle_perimeter, const int threshold,
                      const uint8_t* const center_ptr, const int* offsets) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTcc mht_6(mht_6_v, 390, "", "./tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.cc", "TestCircle");

  // Get the actual value of the center pixel for easier reference later on.
  const int center_value = static_cast<int>(*center_ptr);

  // Number of total pixels to check.  Have to wrap around some in case
  // the contiguous section is split by the array edges.
  const int num_total = circle_perimeter + threshold - 1;

  int num_above = 0;
  int above_diff = 0;

  int num_below = 0;
  int below_diff = 0;

  // Used to tell when this is definitely not going to meet the threshold so we
  // can early abort.
  int minimum_by_now = threshold - num_total + 1;

  // Go through every pixel along the perimeter of the circle, and then around
  // again a little bit.
  for (int i = 0; i < num_total; ++i) {
    // This should be faster than mod.
    const int perim_index = i < circle_perimeter ? i : i - circle_perimeter;

    // This gets the value of the current pixel along the perimeter by using
    // a precomputed offset.
    const int curr_value =
        static_cast<int>(center_ptr[offsets[perim_index]]);

    const int difference = curr_value - center_value;

    if (difference > kFastDiffAmount) {
      above_diff += difference;
      ++num_above;

      num_below = 0;
      below_diff = 0;

      if (num_above >= threshold) {
        return above_diff;
      }
    } else if (difference < -kFastDiffAmount) {
      below_diff += difference;
      ++num_below;

      num_above = 0;
      above_diff = 0;

      if (num_below >= threshold) {
        return below_diff;
      }
    } else {
      num_above = 0;
      num_below = 0;
      above_diff = 0;
      below_diff = 0;
    }

    // See if there's any chance of making the threshold.
    if (MAX(num_above, num_below) < minimum_by_now) {
      // Didn't pass.
      return 0;
    }
    ++minimum_by_now;
  }

  // Didn't pass.
  return 0;
}


// Returns a score in the range [0.0, positive infinity) which represents the
// relative likelihood of a point being a corner.
float KeypointDetector::HarrisFilter(const Image<int32_t>& I_x,
                                     const Image<int32_t>& I_y, const float x,
                                     const float y) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTcc mht_7(mht_7_v, 468, "", "./tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.cc", "KeypointDetector::HarrisFilter");

  if (I_x.ValidInterpPixel(x - kHarrisWindowSize, y - kHarrisWindowSize) &&
      I_x.ValidInterpPixel(x + kHarrisWindowSize, y + kHarrisWindowSize)) {
    // Image gradient matrix.
    float G[] = { 0, 0, 0, 0 };
    CalculateG(kHarrisWindowSize, x, y, I_x, I_y, G);

    const float dx = G[0];
    const float dy = G[3];
    const float dxy = G[1];

    // Harris-Nobel corner score.
    return (dx * dy - Square(dxy)) / (dx + dy + FLT_MIN);
  }

  return 0.0f;
}


int KeypointDetector::AddExtraCandidatesForBoxes(
    const std::vector<BoundingBox>& boxes,
    const int max_num_keypoints,
    Keypoint* const keypoints) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTcc mht_8(mht_8_v, 493, "", "./tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.cc", "KeypointDetector::AddExtraCandidatesForBoxes");

  int num_keypoints_added = 0;

  for (std::vector<BoundingBox>::const_iterator iter = boxes.begin();
      iter != boxes.end(); ++iter) {
    const BoundingBox box = *iter;

    for (int i = 0; i < kNumToAddAsCandidates; ++i) {
      for (int j = 0; j < kNumToAddAsCandidates; ++j) {
        if (num_keypoints_added >= max_num_keypoints) {
          LOGW("Hit cap of %d for temporary keypoints!", max_num_keypoints);
          return num_keypoints_added;
        }

        Keypoint& curr_keypoint = keypoints[num_keypoints_added++];
        curr_keypoint.pos_ = Point2f(
            box.left_ + box.GetWidth() * (i + 0.5f) / kNumToAddAsCandidates,
            box.top_ + box.GetHeight() * (j + 0.5f) / kNumToAddAsCandidates);
        curr_keypoint.type_ = KEYPOINT_TYPE_INTEREST;
      }
    }
  }

  return num_keypoints_added;
}


void KeypointDetector::FindKeypoints(const ImageData& image_data,
                                   const std::vector<BoundingBox>& rois,
                                   const FramePair& prev_change,
                                   FramePair* const curr_change) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTcc mht_9(mht_9_v, 526, "", "./tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.cc", "KeypointDetector::FindKeypoints");

  // Copy keypoints from second frame of last pass to temp keypoints of this
  // pass.
  int number_of_tmp_keypoints = CopyKeypoints(prev_change, tmp_keypoints_);

  const int max_num_fast = kMaxTempKeypoints - number_of_tmp_keypoints;
  number_of_tmp_keypoints +=
      FindFastKeypoints(image_data, max_num_fast,
                       tmp_keypoints_ + number_of_tmp_keypoints);

  TimeLog("Found FAST keypoints");

  if (number_of_tmp_keypoints >= kMaxTempKeypoints) {
    LOGW("Hit cap of %d for temporary keypoints (FAST)! %d keypoints",
         kMaxTempKeypoints, number_of_tmp_keypoints);
  }

  if (kAddArbitraryKeypoints) {
    // Add some for each object prior to scoring.
    const int max_num_box_keypoints =
        kMaxTempKeypoints - number_of_tmp_keypoints;
    number_of_tmp_keypoints +=
        AddExtraCandidatesForBoxes(rois, max_num_box_keypoints,
                                   tmp_keypoints_ + number_of_tmp_keypoints);
    TimeLog("Added box keypoints");

    if (number_of_tmp_keypoints >= kMaxTempKeypoints) {
      LOGW("Hit cap of %d for temporary keypoints (boxes)! %d keypoints",
           kMaxTempKeypoints, number_of_tmp_keypoints);
    }
  }

  // Score them...
  LOGV("Scoring %d keypoints!", number_of_tmp_keypoints);
  ScoreKeypoints(image_data, number_of_tmp_keypoints, tmp_keypoints_);
  TimeLog("Scored keypoints");

  // Now pare it down a bit.
  SortKeypoints(number_of_tmp_keypoints, tmp_keypoints_);
  TimeLog("Sorted keypoints");

  LOGV("%d keypoints to select from!", number_of_tmp_keypoints);

  SelectKeypoints(rois, tmp_keypoints_, number_of_tmp_keypoints, curr_change);
  TimeLog("Selected keypoints");

  LOGV("Picked %d (%d max) final keypoints out of %d potential.",
       curr_change->number_of_keypoints_,
       kMaxKeypoints, number_of_tmp_keypoints);
}


int KeypointDetector::CopyKeypoints(const FramePair& prev_change,
                                  Keypoint* const new_keypoints) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTcc mht_10(mht_10_v, 582, "", "./tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.cc", "KeypointDetector::CopyKeypoints");

  int number_of_keypoints = 0;

  // Caching values from last pass, just copy and compact.
  for (int i = 0; i < prev_change.number_of_keypoints_; ++i) {
    if (prev_change.optical_flow_found_keypoint_[i]) {
      new_keypoints[number_of_keypoints] =
          prev_change.frame2_keypoints_[i];

      new_keypoints[number_of_keypoints].score_ =
          prev_change.frame1_keypoints_[i].score_;

      ++number_of_keypoints;
    }
  }

  TimeLog("Copied keypoints");
  return number_of_keypoints;
}


// FAST keypoint detector.
int KeypointDetector::FindFastKeypoints(const Image<uint8_t>& frame,
                                        const int quadrant,
                                        const int downsample_factor,
                                        const int max_num_keypoints,
                                        Keypoint* const keypoints) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTcc mht_11(mht_11_v, 611, "", "./tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.cc", "KeypointDetector::FindFastKeypoints");

  /*
   // Reference for a circle of diameter 7.
   const int circle[] = {0, 0, 1, 1, 1, 0, 0,
                         0, 1, 0, 0, 0, 1, 0,
                         1, 0, 0, 0, 0, 0, 1,
                         1, 0, 0, 0, 0, 0, 1,
                         1, 0, 0, 0, 0, 0, 1,
                         0, 1, 0, 0, 0, 1, 0,
                         0, 0, 1, 1, 1, 0, 0};
   const int circle_offset[] =
       {2, 3, 4, 8, 12, 14, 20, 21, 27, 28, 34, 36, 40, 44, 45, 46};
   */

  // Quick test of compass directions.  Any length 16 circle with a break of up
  // to 4 pixels will have at least 3 of these 4 pixels active.
  static const int short_circle_perimeter = 4;
  static const int short_threshold = 3;
  static const int short_circle_x[] = { -3,  0, +3,  0 };
  static const int short_circle_y[] = {  0, -3,  0, +3 };

  // Precompute image offsets.
  int short_offsets[short_circle_perimeter];
  for (int i = 0; i < short_circle_perimeter; ++i) {
    short_offsets[i] = short_circle_x[i] + short_circle_y[i] * frame.GetWidth();
  }

  // Large circle values.
  static const int full_circle_perimeter = 16;
  static const int full_threshold = 12;
  static const int full_circle_x[] =
      { -1,  0, +1, +2, +3, +3, +3, +2, +1, +0, -1, -2, -3, -3, -3, -2 };
  static const int full_circle_y[] =
      { -3, -3, -3, -2, -1,  0, +1, +2, +3, +3, +3, +2, +1, +0, -1, -2 };

  // Precompute image offsets.
  int full_offsets[full_circle_perimeter];
  for (int i = 0; i < full_circle_perimeter; ++i) {
    full_offsets[i] = full_circle_x[i] + full_circle_y[i] * frame.GetWidth();
  }

  const int scratch_stride = frame.stride();

  keypoint_scratch_->Clear(0);

  // Set up the bounds on the region to test based on the passed-in quadrant.
  const int quadrant_width = (frame.GetWidth() / 2) - kFastBorderBuffer;
  const int quadrant_height = (frame.GetHeight() / 2) - kFastBorderBuffer;
  const int start_x =
      kFastBorderBuffer + ((quadrant % 2 == 0) ? 0 : quadrant_width);
  const int start_y =
      kFastBorderBuffer + ((quadrant < 2) ? 0 : quadrant_height);
  const int end_x = start_x + quadrant_width;
  const int end_y = start_y + quadrant_height;

  // Loop through once to find FAST keypoint clumps.
  for (int img_y = start_y; img_y < end_y; ++img_y) {
    const uint8_t* curr_pixel_ptr = frame[img_y] + start_x;

    for (int img_x = start_x; img_x < end_x; ++img_x) {
      // Only insert it if it meets the quick minimum requirements test.
      if (TestCircle(short_circle_perimeter, short_threshold,
                     curr_pixel_ptr, short_offsets) != 0) {
        // Longer test for actual keypoint score..
        const int fast_score = TestCircle(full_circle_perimeter,
                                          full_threshold,
                                          curr_pixel_ptr,
                                          full_offsets);

        // Non-zero score means the keypoint was found.
        if (fast_score != 0) {
          uint8_t* const center_ptr = (*keypoint_scratch_)[img_y] + img_x;

          // Increase the keypoint count on this pixel and the pixels in all
          // 4 cardinal directions.
          *center_ptr += 5;
          *(center_ptr - 1) += 1;
          *(center_ptr + 1) += 1;
          *(center_ptr - scratch_stride) += 1;
          *(center_ptr + scratch_stride) += 1;
        }
      }

      ++curr_pixel_ptr;
    }  // x
  }  // y

  TimeLog("Found FAST keypoints.");

  int num_keypoints = 0;
  // Loop through again and Harris filter pixels in the center of clumps.
  // We can shrink the window by 1 pixel on every side.
  for (int img_y = start_y + 1; img_y < end_y - 1; ++img_y) {
    const uint8_t* curr_pixel_ptr = (*keypoint_scratch_)[img_y] + start_x;

    for (int img_x = start_x + 1; img_x < end_x - 1; ++img_x) {
      if (*curr_pixel_ptr >= kMinNumConnectedForFastKeypoint) {
       Keypoint* const keypoint = keypoints + num_keypoints;
        keypoint->pos_ = Point2f(
            img_x * downsample_factor, img_y * downsample_factor);
        keypoint->score_ = 0;
        keypoint->type_ = KEYPOINT_TYPE_FAST;

        ++num_keypoints;
        if (num_keypoints >= max_num_keypoints) {
          return num_keypoints;
        }
      }

      ++curr_pixel_ptr;
    }  // x
  }  // y

  TimeLog("Picked FAST keypoints.");

  return num_keypoints;
}

int KeypointDetector::FindFastKeypoints(const ImageData& image_data,
                                        const int max_num_keypoints,
                                        Keypoint* const keypoints) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTcc mht_12(mht_12_v, 734, "", "./tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.cc", "KeypointDetector::FindFastKeypoints");

  int downsample_factor = 1;
  int num_found = 0;

  // TODO(andrewharp): Get this working for multiple image scales.
  for (int i = 0; i < 1; ++i) {
    const Image<uint8_t>& frame = *image_data.GetPyramidSqrt2Level(i);
    num_found += FindFastKeypoints(
        frame, fast_quadrant_,
        downsample_factor, max_num_keypoints, keypoints + num_found);
    downsample_factor *= 2;
  }

  // Increment the current quadrant.
  fast_quadrant_ = (fast_quadrant_ + 1) % 4;

  return num_found;
}

}  // namespace tf_tracking
