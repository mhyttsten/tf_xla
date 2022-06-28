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

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_TRACKED_OBJECT_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_TRACKED_OBJECT_H_
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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTh {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTh() {
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


#ifdef __RENDER_OPENGL__
#include "tensorflow/tools/android/test/jni/object_tracking/gl_utils.h"
#endif
#include "tensorflow/tools/android/test/jni/object_tracking/object_detector.h"

namespace tf_tracking {

// A TrackedObject is a specific instance of an ObjectModel, with a known
// position in the world.
// It provides the last known position and number of recent detection failures,
// in addition to the more general appearance data associated with the object
// class (which is in ObjectModel).
// TODO(andrewharp): Make getters/setters follow styleguide.
class TrackedObject {
 public:
  TrackedObject(const std::string& id, const Image<uint8_t>& image,
                const BoundingBox& bounding_box, ObjectModelBase* const model);

  ~TrackedObject();

  void UpdatePosition(const BoundingBox& new_position, const int64_t timestamp,
                      const ImageData& image_data, const bool authoritative);

  // This method is called when the tracked object is detected at a
  // given position, and allows the associated Model to grow and/or prune
  // itself based on where the detection occurred.
  void OnDetection(ObjectModelBase* const model,
                   const BoundingBox& detection_position,
                   const MatchScore match_score, const int64_t timestamp,
                   const ImageData& image_data);

  // Called when there's no detection of the tracked object. This will cause
  // a tracking failure after enough consecutive failures if the area under
  // the current bounding box also doesn't meet a minimum correlation threshold
  // with the model.
  void OnDetectionFailure() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTh mht_0(mht_0_v, 223, "", "./tensorflow/tools/android/test/jni/object_tracking/tracked_object.h", "OnDetectionFailure");
}

  inline bool IsVisible() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTh mht_1(mht_1_v, 228, "", "./tensorflow/tools/android/test/jni/object_tracking/tracked_object.h", "IsVisible");

    return tracked_correlation_ >= kMinimumCorrelationForTracking ||
        num_consecutive_frames_below_threshold_ < kMaxNumDetectionFailures;
  }

  inline float GetCorrelation() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTh mht_2(mht_2_v, 236, "", "./tensorflow/tools/android/test/jni/object_tracking/tracked_object.h", "GetCorrelation");

    return tracked_correlation_;
  }

  inline MatchScore GetMatchScore() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTh mht_3(mht_3_v, 243, "", "./tensorflow/tools/android/test/jni/object_tracking/tracked_object.h", "GetMatchScore");

    return tracked_match_score_;
  }

  inline BoundingBox GetPosition() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTh mht_4(mht_4_v, 250, "", "./tensorflow/tools/android/test/jni/object_tracking/tracked_object.h", "GetPosition");

    return last_known_position_;
  }

  inline BoundingBox GetLastDetectionPosition() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTh mht_5(mht_5_v, 257, "", "./tensorflow/tools/android/test/jni/object_tracking/tracked_object.h", "GetLastDetectionPosition");

    return last_detection_position_;
  }

  inline const ObjectModelBase* GetModel() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTh mht_6(mht_6_v, 264, "", "./tensorflow/tools/android/test/jni/object_tracking/tracked_object.h", "GetModel");

    return object_model_;
  }

  inline const std::string& GetName() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTh mht_7(mht_7_v, 271, "", "./tensorflow/tools/android/test/jni/object_tracking/tracked_object.h", "GetName");

    return id_;
  }

  inline void Draw() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTh mht_8(mht_8_v, 278, "", "./tensorflow/tools/android/test/jni/object_tracking/tracked_object.h", "Draw");

#ifdef __RENDER_OPENGL__
    if (tracked_correlation_ < kMinimumCorrelationForTracking) {
      glColor4f(MAX(0.0f, -tracked_correlation_),
                MAX(0.0f, tracked_correlation_),
                0.0f,
                1.0f);
    } else {
      glColor4f(MAX(0.0f, -tracked_correlation_),
                MAX(0.0f, tracked_correlation_),
                1.0f,
                1.0f);
    }

    // Render the box itself.
    BoundingBox temp_box(last_known_position_);
    DrawBox(temp_box);

    // Render a box inside this one (in case the actual box is hidden).
    const float kBufferSize = 1.0f;
    temp_box.left_ -= kBufferSize;
    temp_box.top_ -= kBufferSize;
    temp_box.right_ += kBufferSize;
    temp_box.bottom_ += kBufferSize;
    DrawBox(temp_box);

    // Render one outside as well.
    temp_box.left_ -= -2.0f * kBufferSize;
    temp_box.top_ -= -2.0f * kBufferSize;
    temp_box.right_ += -2.0f * kBufferSize;
    temp_box.bottom_ += -2.0f * kBufferSize;
    DrawBox(temp_box);
#endif
  }

  // Get current object's num_consecutive_frames_below_threshold_.
  inline int64_t GetNumConsecutiveFramesBelowThreshold() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTh mht_9(mht_9_v, 317, "", "./tensorflow/tools/android/test/jni/object_tracking/tracked_object.h", "GetNumConsecutiveFramesBelowThreshold");

    return num_consecutive_frames_below_threshold_;
  }

  // Reset num_consecutive_frames_below_threshold_ to 0.
  inline void resetNumConsecutiveFramesBelowThreshold() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTh mht_10(mht_10_v, 325, "", "./tensorflow/tools/android/test/jni/object_tracking/tracked_object.h", "resetNumConsecutiveFramesBelowThreshold");

    num_consecutive_frames_below_threshold_ = 0;
  }

  inline float GetAllowableDistanceSquared() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTh mht_11(mht_11_v, 332, "", "./tensorflow/tools/android/test/jni/object_tracking/tracked_object.h", "GetAllowableDistanceSquared");

    return allowable_detection_distance_;
  }

 private:
  // The unique id used throughout the system to identify this
  // tracked object.
  const std::string id_;

  // The last known position of the object.
  BoundingBox last_known_position_;

  // The last known position of the object.
  BoundingBox last_detection_position_;

  // When the position was last computed.
  int64_t position_last_computed_time_;

  // The object model this tracked object is representative of.
  ObjectModelBase* object_model_;

  Image<float> last_detection_thumbnail_;

  Image<float> last_frame_thumbnail_;

  // The correlation of the object model with the preview frame at its last
  // tracked position.
  float tracked_correlation_;

  MatchScore tracked_match_score_;

  // The number of consecutive frames that the tracked position for this object
  // has been under the correlation threshold.
  int num_consecutive_frames_below_threshold_;

  float allowable_detection_distance_;

  friend std::ostream& operator<<(std::ostream& stream,
                                  const TrackedObject& tracked_object);

  TF_DISALLOW_COPY_AND_ASSIGN(TrackedObject);
};

inline std::ostream& operator<<(std::ostream& stream,
                                const TrackedObject& tracked_object) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPStracked_objectDTh mht_12(mht_12_v, 379, "", "./tensorflow/tools/android/test/jni/object_tracking/tracked_object.h", "operator<<");

  stream << tracked_object.id_
      << " " << tracked_object.last_known_position_
      << " " << tracked_object.position_last_computed_time_
      << " " << tracked_object.num_consecutive_frames_below_threshold_
      << " " << tracked_object.object_model_
      << " " << tracked_object.tracked_correlation_;
  return stream;
}

}  // namespace tf_tracking

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_TRACKED_OBJECT_H_
