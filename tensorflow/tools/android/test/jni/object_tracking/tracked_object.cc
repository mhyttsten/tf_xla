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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTcc() {
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

#include "tensorflow/tools/android/test/jni/object_tracking/tracked_object.h"

namespace tf_tracking {

static const float kInitialDistance = 20.0f;

static void InitNormalized(const Image<uint8_t>& src_image,
                           const BoundingBox& position,
                           Image<float>* const dst_image) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTcc mht_0(mht_0_v, 193, "", "./tensorflow/tools/android/test/jni/object_tracking/tracked_object.cc", "InitNormalized");

  BoundingBox scaled_box(position);
  CopyArea(src_image, scaled_box, dst_image);
  NormalizeImage(dst_image);
}

TrackedObject::TrackedObject(const std::string& id, const Image<uint8_t>& image,
                             const BoundingBox& bounding_box,
                             ObjectModelBase* const model)
    : id_(id),
      last_known_position_(bounding_box),
      last_detection_position_(bounding_box),
      position_last_computed_time_(-1),
      object_model_(model),
      last_detection_thumbnail_(kNormalizedThumbnailSize,
                                kNormalizedThumbnailSize),
      last_frame_thumbnail_(kNormalizedThumbnailSize, kNormalizedThumbnailSize),
      tracked_correlation_(0.0f),
      tracked_match_score_(0.0),
      num_consecutive_frames_below_threshold_(0),
      allowable_detection_distance_(Square(kInitialDistance)) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("id: \"" + id + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTcc mht_1(mht_1_v, 217, "", "./tensorflow/tools/android/test/jni/object_tracking/tracked_object.cc", "TrackedObject::TrackedObject");

  InitNormalized(image, bounding_box, &last_detection_thumbnail_);
}

TrackedObject::~TrackedObject() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTcc mht_2(mht_2_v, 224, "", "./tensorflow/tools/android/test/jni/object_tracking/tracked_object.cc", "TrackedObject::~TrackedObject");
}

void TrackedObject::UpdatePosition(const BoundingBox& new_position,
                                   const int64_t timestamp,
                                   const ImageData& image_data,
                                   const bool authoritative) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTcc mht_3(mht_3_v, 232, "", "./tensorflow/tools/android/test/jni/object_tracking/tracked_object.cc", "TrackedObject::UpdatePosition");

  last_known_position_ = new_position;
  position_last_computed_time_ = timestamp;

  InitNormalized(*image_data.GetImage(), new_position, &last_frame_thumbnail_);

  const float last_localization_correlation = ComputeCrossCorrelation(
      last_detection_thumbnail_.data(),
      last_frame_thumbnail_.data(),
      last_frame_thumbnail_.data_size_);
  LOGV("Tracked correlation to last localization:   %.6f",
       last_localization_correlation);

  // Correlation to object model, if it exists.
  if (object_model_ != NULL) {
    tracked_correlation_ =
        object_model_->GetMaxCorrelation(last_frame_thumbnail_);
    LOGV("Tracked correlation to model:               %.6f",
         tracked_correlation_);

    tracked_match_score_ =
        object_model_->GetMatchScore(new_position, image_data);
    LOGV("Tracked match score with model:             %.6f",
         tracked_match_score_.value);
  } else {
    // If there's no model to check against, set the tracked correlation to
    // simply be the correlation to the last set position.
    tracked_correlation_ = last_localization_correlation;
    tracked_match_score_ = MatchScore(0.0f);
  }

  // Determine if it's still being tracked.
  if (tracked_correlation_ >= kMinimumCorrelationForTracking &&
      tracked_match_score_ >= kMinimumMatchScore) {
    num_consecutive_frames_below_threshold_ = 0;

    if (object_model_ != NULL) {
      object_model_->TrackStep(last_known_position_, *image_data.GetImage(),
                               *image_data.GetIntegralImage(), authoritative);
    }
  } else if (tracked_match_score_ < kMatchScoreForImmediateTermination) {
    if (num_consecutive_frames_below_threshold_ < 1000) {
      LOGD("Tracked match score is way too low (%.6f), aborting track.",
           tracked_match_score_.value);
    }

    // Add an absurd amount of missed frames so that all heuristics will
    // consider it a lost track.
    num_consecutive_frames_below_threshold_ += 1000;

    if (object_model_ != NULL) {
      object_model_->TrackLost();
    }
  } else {
    ++num_consecutive_frames_below_threshold_;
    allowable_detection_distance_ *= 1.1f;
  }
}

void TrackedObject::OnDetection(ObjectModelBase* const model,
                                const BoundingBox& detection_position,
                                const MatchScore match_score,
                                const int64_t timestamp,
                                const ImageData& image_data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTcc mht_4(mht_4_v, 298, "", "./tensorflow/tools/android/test/jni/object_tracking/tracked_object.cc", "TrackedObject::OnDetection");

  const float overlap = detection_position.PascalScore(last_known_position_);
  if (overlap > kPositionOverlapThreshold) {
    // If the position agreement with the current tracked position is good
    // enough, lock all the current unlocked examples.
    object_model_->TrackConfirmed();
    num_consecutive_frames_below_threshold_ = 0;
  }

  // Before relocalizing, make sure the new proposed position is better than
  // the existing position by a small amount to prevent thrashing.
  if (match_score <= tracked_match_score_ + kMatchScoreBuffer) {
    LOGI("Not relocalizing since new match is worse: %.6f < %.6f + %.6f",
         match_score.value, tracked_match_score_.value,
         kMatchScoreBuffer.value);
    return;
  }

  LOGI("Relocalizing! From (%.1f, %.1f)[%.1fx%.1f] to "
       "(%.1f, %.1f)[%.1fx%.1f]:   %.6f > %.6f",
       last_known_position_.left_, last_known_position_.top_,
       last_known_position_.GetWidth(), last_known_position_.GetHeight(),
       detection_position.left_, detection_position.top_,
       detection_position.GetWidth(), detection_position.GetHeight(),
       match_score.value, tracked_match_score_.value);

  if (overlap < kPositionOverlapThreshold) {
    // The path might be good, it might be bad, but it's no longer a path
    // since we're moving the box to a new position, so just nuke it from
    // orbit to be safe.
    object_model_->TrackLost();
  }

  object_model_ = model;

  // Reset the last detected appearance.
  InitNormalized(
      *image_data.GetImage(), detection_position, &last_detection_thumbnail_);

  num_consecutive_frames_below_threshold_ = 0;
  last_detection_position_ = detection_position;

  UpdatePosition(detection_position, timestamp, image_data, false);
  allowable_detection_distance_ = Square(kInitialDistance);
}

}  // namespace tf_tracking
