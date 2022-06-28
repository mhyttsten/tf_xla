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

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_KEYPOINT_DETECTOR_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_KEYPOINT_DETECTOR_H_
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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTh {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTh() {
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


#include <stdint.h>

#include <vector>

#include "tensorflow/tools/android/test/jni/object_tracking/image-inl.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image_data.h"
#include "tensorflow/tools/android/test/jni/object_tracking/optical_flow.h"

namespace tf_tracking {

struct Keypoint;

class KeypointDetector {
 public:
  explicit KeypointDetector(const KeypointDetectorConfig* const config)
      : config_(config),
        keypoint_scratch_(new Image<uint8_t>(config_->image_size)),
        interest_map_(new Image<bool>(config_->image_size)),
        fast_quadrant_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTh mht_0(mht_0_v, 207, "", "./tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.h", "KeypointDetector");

    interest_map_->Clear(false);
  }

  ~KeypointDetector() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSkeypoint_detectorDTh mht_1(mht_1_v, 214, "", "./tensorflow/tools/android/test/jni/object_tracking/keypoint_detector.h", "~KeypointDetector");
}

  // Finds a new set of keypoints for the current frame, picked from the current
  // set of keypoints and also from a set discovered via a keypoint detector.
  // Special attention is applied to make sure that keypoints are distributed
  // within the supplied ROIs.
  void FindKeypoints(const ImageData& image_data,
                     const std::vector<BoundingBox>& rois,
                     const FramePair& prev_change,
                     FramePair* const curr_change);

 private:
  // Compute the corneriness of a point in the image.
  float HarrisFilter(const Image<int32_t>& I_x, const Image<int32_t>& I_y,
                     const float x, const float y) const;

  // Adds a grid of candidate keypoints to the given box, up to
  // max_num_keypoints or kNumToAddAsCandidates^2, whichever is lower.
  int AddExtraCandidatesForBoxes(
      const std::vector<BoundingBox>& boxes,
      const int max_num_keypoints,
      Keypoint* const keypoints) const;

  // Scan the frame for potential keypoints using the FAST keypoint detector.
  // Quadrant is an argument 0-3 which refers to the quadrant of the image in
  // which to detect keypoints.
  int FindFastKeypoints(const Image<uint8_t>& frame, const int quadrant,
                        const int downsample_factor,
                        const int max_num_keypoints, Keypoint* const keypoints);

  int FindFastKeypoints(const ImageData& image_data,
                        const int max_num_keypoints,
                        Keypoint* const keypoints);

  // Score a bunch of candidate keypoints.  Assigns the scores to the input
  // candidate_keypoints array entries.
  void ScoreKeypoints(const ImageData& image_data,
                      const int num_candidates,
                      Keypoint* const candidate_keypoints);

  void SortKeypoints(const int num_candidates,
                    Keypoint* const candidate_keypoints) const;

  // Selects a set of keypoints falling within the supplied box such that the
  // most highly rated keypoints are picked first, and so that none of them are
  // too close together.
  int SelectKeypointsInBox(
      const BoundingBox& box,
      const Keypoint* const candidate_keypoints,
      const int num_candidates,
      const int max_keypoints,
      const int num_existing_keypoints,
      const Keypoint* const existing_keypoints,
      Keypoint* const final_keypoints) const;

  // Selects from the supplied sorted keypoint pool a set of keypoints that will
  // best cover the given set of boxes, such that each box is covered at a
  // resolution proportional to its size.
  void SelectKeypoints(
      const std::vector<BoundingBox>& boxes,
      const Keypoint* const candidate_keypoints,
      const int num_candidates,
      FramePair* const frame_change) const;

  // Copies and compacts the found keypoints in the second frame of prev_change
  // into the array at new_keypoints.
  static int CopyKeypoints(const FramePair& prev_change,
                          Keypoint* const new_keypoints);

  const KeypointDetectorConfig* const config_;

  // Scratch memory for keypoint candidacy detection and non-max suppression.
  std::unique_ptr<Image<uint8_t> > keypoint_scratch_;

  // Regions of the image to pay special attention to.
  std::unique_ptr<Image<bool> > interest_map_;

  // The current quadrant of the image to detect FAST keypoints in.
  // Keypoint detection is staggered for performance reasons. Every four frames
  // a full scan of the frame will have been performed.
  int fast_quadrant_;

  Keypoint tmp_keypoints_[kMaxTempKeypoints];
};

}  // namespace tf_tracking

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_KEYPOINT_DETECTOR_H_
