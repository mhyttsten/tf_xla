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

// NOTE: no native object detectors are currently provided or used by the code
// in this directory. This class remains mainly for historical reasons.
// Detection in the TF demo is done through TensorFlowMultiBoxDetector.java.

// Defines the ObjectDetector class that is the main interface for detecting
// ObjectModelBases in frames.

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_OBJECT_DETECTOR_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_OBJECT_DETECTOR_H_
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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh() {
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


#include <float.h>

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/tools/android/test/jni/object_tracking/geom.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image-inl.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image.h"
#include "tensorflow/tools/android/test/jni/object_tracking/integral_image.h"
#ifdef __RENDER_OPENGL__
#include "tensorflow/tools/android/test/jni/object_tracking/sprite.h"
#endif
#include "tensorflow/tools/android/test/jni/object_tracking/config.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image_data.h"
#include "tensorflow/tools/android/test/jni/object_tracking/object_model.h"
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

namespace tf_tracking {

// Adds BoundingSquares to a vector such that the first square added is centered
// in the position given and of square_size, and the remaining squares are added
// concentrentically, scaling down by scale_factor until the minimum threshold
// size is passed.
// Squares that do not fall completely within image_bounds will not be added.
static inline void FillWithSquares(
    const BoundingBox& image_bounds,
    const BoundingBox& position,
    const float starting_square_size,
    const float smallest_square_size,
    const float scale_factor,
    std::vector<BoundingSquare>* const squares) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh mht_0(mht_0_v, 228, "", "./tensorflow/tools/android/test/jni/object_tracking/object_detector.h", "FillWithSquares");

  BoundingSquare descriptor_area =
      GetCenteredSquare(position, starting_square_size);

  SCHECK(scale_factor < 1.0f, "Scale factor too large at %.2f!", scale_factor);

  // Use a do/while loop to ensure that at least one descriptor is created.
  do {
    if (image_bounds.Contains(descriptor_area.ToBoundingBox())) {
      squares->push_back(descriptor_area);
    }
    descriptor_area.Scale(scale_factor);
  } while (descriptor_area.size_ >= smallest_square_size - EPSILON);
  LOGV("Created %zu squares starting from size %.2f to min size %.2f "
       "using scale factor: %.2f",
       squares->size(), starting_square_size, smallest_square_size,
       scale_factor);
}


// Represents a potential detection of a specific ObjectExemplar and Descriptor
// at a specific position in the image.
class Detection {
 public:
  explicit Detection(const ObjectModelBase* const object_model,
                     const MatchScore match_score,
                     const BoundingBox& bounding_box)
      : object_model_(object_model),
        match_score_(match_score),
        bounding_box_(bounding_box) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh mht_1(mht_1_v, 260, "", "./tensorflow/tools/android/test/jni/object_tracking/object_detector.h", "Detection");
}

  Detection(const Detection& other)
      : object_model_(other.object_model_),
        match_score_(other.match_score_),
        bounding_box_(other.bounding_box_) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh mht_2(mht_2_v, 268, "", "./tensorflow/tools/android/test/jni/object_tracking/object_detector.h", "Detection");
}

  virtual ~Detection() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh mht_3(mht_3_v, 273, "", "./tensorflow/tools/android/test/jni/object_tracking/object_detector.h", "~Detection");
}

  inline BoundingBox GetObjectBoundingBox() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh mht_4(mht_4_v, 278, "", "./tensorflow/tools/android/test/jni/object_tracking/object_detector.h", "GetObjectBoundingBox");

    return bounding_box_;
  }

  inline MatchScore GetMatchScore() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh mht_5(mht_5_v, 285, "", "./tensorflow/tools/android/test/jni/object_tracking/object_detector.h", "GetMatchScore");

    return match_score_;
  }

  inline const ObjectModelBase* GetObjectModel() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh mht_6(mht_6_v, 292, "", "./tensorflow/tools/android/test/jni/object_tracking/object_detector.h", "GetObjectModel");

    return object_model_;
  }

  inline bool Intersects(const Detection& other) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh mht_7(mht_7_v, 299, "", "./tensorflow/tools/android/test/jni/object_tracking/object_detector.h", "Intersects");

    // Check if any of the four axes separates us, there must be at least one.
    return bounding_box_.Intersects(other.bounding_box_);
  }

  struct Comp {
    inline bool operator()(const Detection& a, const Detection& b) const {
      return a.match_score_ > b.match_score_;
    }
  };

  // TODO(andrewharp): add accessors to update these instead.
  const ObjectModelBase* object_model_;
  MatchScore match_score_;
  BoundingBox bounding_box_;
};

inline std::ostream& operator<<(std::ostream& stream,
                                const Detection& detection) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh mht_8(mht_8_v, 320, "", "./tensorflow/tools/android/test/jni/object_tracking/object_detector.h", "operator<<");

  const BoundingBox actual_area = detection.GetObjectBoundingBox();
  stream << actual_area;
  return stream;
}

class ObjectDetectorBase {
 public:
  explicit ObjectDetectorBase(const ObjectDetectorConfig* const config)
      : config_(config),
        image_data_(NULL) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh mht_9(mht_9_v, 333, "", "./tensorflow/tools/android/test/jni/object_tracking/object_detector.h", "ObjectDetectorBase");
}

  virtual ~ObjectDetectorBase();

  // Sets the current image data. All calls to ObjectDetector other than
  // FillDescriptors use the image data last set.
  inline void SetImageData(const ImageData* const image_data) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh mht_10(mht_10_v, 342, "", "./tensorflow/tools/android/test/jni/object_tracking/object_detector.h", "SetImageData");

    image_data_ = image_data;
  }

  // Main entry point into the detection algorithm.
  // Scans the frame for candidates, tweaks them, and fills in the
  // given std::vector of Detection objects with acceptable matches.
  virtual void Detect(const std::vector<BoundingSquare>& positions,
                      std::vector<Detection>* const detections) const = 0;

  virtual ObjectModelBase* CreateObjectModel(const std::string& name) = 0;

  virtual void DeleteObjectModel(const std::string& name) = 0;

  virtual void GetObjectModels(
      std::vector<const ObjectModelBase*>* models) const = 0;

  // Creates a new ObjectExemplar from the given position in the context of
  // the last frame passed to NextFrame.
  // Will return null in the case that there's no room for a descriptor to be
  // created in the example area, or the example area is not completely
  // contained within the frame.
  virtual void UpdateModel(const Image<uint8_t>& base_image,
                           const IntegralImage& integral_image,
                           const BoundingBox& bounding_box, const bool locked,
                           ObjectModelBase* model) const = 0;

  virtual void Draw() const = 0;

  virtual bool AllowSpontaneousDetections() = 0;

 protected:
  const std::unique_ptr<const ObjectDetectorConfig> config_;

  // The latest frame data, upon which all detections will be performed.
  // Not owned by this object, just provided for reference by ObjectTracker
  // via SetImageData().
  const ImageData* image_data_;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ObjectDetectorBase);
};

template <typename ModelType>
class ObjectDetector : public ObjectDetectorBase {
 public:
  explicit ObjectDetector(const ObjectDetectorConfig* const config)
      : ObjectDetectorBase(config) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh mht_11(mht_11_v, 392, "", "./tensorflow/tools/android/test/jni/object_tracking/object_detector.h", "ObjectDetector");
}

  virtual ~ObjectDetector() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh mht_12(mht_12_v, 397, "", "./tensorflow/tools/android/test/jni/object_tracking/object_detector.h", "~ObjectDetector");

    typename std::map<std::string, ModelType*>::const_iterator it =
        object_models_.begin();
    for (; it != object_models_.end(); ++it) {
      ModelType* model = it->second;
      delete model;
    }
  }

  virtual void DeleteObjectModel(const std::string& name) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh mht_13(mht_13_v, 410, "", "./tensorflow/tools/android/test/jni/object_tracking/object_detector.h", "DeleteObjectModel");

    ModelType* model = object_models_[name];
    CHECK_ALWAYS(model != NULL, "Model was null!");
    object_models_.erase(name);
    SAFE_DELETE(model);
  }

  virtual void GetObjectModels(
      std::vector<const ObjectModelBase*>* models) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh mht_14(mht_14_v, 421, "", "./tensorflow/tools/android/test/jni/object_tracking/object_detector.h", "GetObjectModels");

    typename std::map<std::string, ModelType*>::const_iterator it =
        object_models_.begin();
    for (; it != object_models_.end(); ++it) {
      models->push_back(it->second);
    }
  }

  virtual bool AllowSpontaneousDetections() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSobject_detectorDTh mht_15(mht_15_v, 432, "", "./tensorflow/tools/android/test/jni/object_tracking/object_detector.h", "AllowSpontaneousDetections");

    return false;
  }

 protected:
  std::map<std::string, ModelType*> object_models_;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ObjectDetector);
};

}  // namespace tf_tracking

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_OBJECT_DETECTOR_H_
