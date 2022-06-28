/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_UTILS_IMAGE_METRICS_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_UTILS_IMAGE_METRICS_H_
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
class MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTh {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTh() {
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

namespace tflite {
namespace evaluation {
namespace image {

struct Box2D {
  struct Interval {
    float min = 0;
    float max = 0;
    Interval(float x, float y) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTh mht_0(mht_0_v, 199, "", "./tensorflow/lite/tools/evaluation/stages/utils/image_metrics.h", "Interval");

      min = x;
      max = y;
    }
    Interval() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTh mht_1(mht_1_v, 206, "", "./tensorflow/lite/tools/evaluation/stages/utils/image_metrics.h", "Interval");
}
  };

  Interval x;
  Interval y;
  static float Length(const Interval& a);
  static float Intersection(const Interval& a, const Interval& b);
  float Area() const;
  float Intersection(const Box2D& other) const;
  float Union(const Box2D& other) const;
  // Intersection of this box and the given box normalized over the union of
  // this box and the given box.
  float IoU(const Box2D& other) const;
  // Intersection of this box and the given box normalized over the area of
  // this box.
  float Overlap(const Box2D& other) const;
};

// If the value is:
//   - kDontIgnore: The object is included in this evaluation.
//   - kIgnoreOneMatch: the first matched prediction bbox will be ignored. This
//      is useful when this groundtruth object is not intended to be evaluated.
//   - kIgnoreAllMatches: all matched prediction bbox will be ignored. Typically
//      it is used to mark an area that has not been labeled.
enum IgnoreType {
  kDontIgnore = 0,
  kIgnoreOneMatch = 1,
  kIgnoreAllMatches = 2,
};

struct Detection {
 public:
  bool difficult = false;
  int64_t imgid = 0;
  float score = 0;
  Box2D box;
  IgnoreType ignore = IgnoreType::kDontIgnore;

  Detection() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTh mht_2(mht_2_v, 247, "", "./tensorflow/lite/tools/evaluation/stages/utils/image_metrics.h", "Detection");
}
  Detection(bool d, int64_t id, float s, Box2D b)
      : difficult(d), imgid(id), score(s), box(b) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTh mht_3(mht_3_v, 252, "", "./tensorflow/lite/tools/evaluation/stages/utils/image_metrics.h", "Detection");
}
  Detection(bool d, int64_t id, float s, Box2D b, IgnoreType i)
      : difficult(d), imgid(id), score(s), box(b), ignore(i) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTh mht_4(mht_4_v, 257, "", "./tensorflow/lite/tools/evaluation/stages/utils/image_metrics.h", "Detection");
}
};

// Precision and recall.
struct PR {
  float p = 0;
  float r = 0;
  PR(const float p_, const float r_) : p(p_), r(r_) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTh mht_5(mht_5_v, 267, "", "./tensorflow/lite/tools/evaluation/stages/utils/image_metrics.h", "PR");
}
};

class AveragePrecision {
 public:
  // iou_threshold: A predicted box matches a ground truth box if and only if
  //   IoU between these two are larger than this iou_threshold. Default: 0.5.
  // num_recall_points: AP is computed as the average of maximum precision at (1
  //   + num_recall_points) recall levels. E.g., if num_recall_points is 10,
  //   recall levels are 0., 0.1, 0.2, ..., 0.9, 1.0.
  // Default: 100. If num_recall_points < 0, AveragePrecision of 0 is returned.
  struct Options {
    float iou_threshold = 0.5;
    int num_recall_points = 100;
  };
  AveragePrecision() : AveragePrecision(Options()) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTh mht_6(mht_6_v, 285, "", "./tensorflow/lite/tools/evaluation/stages/utils/image_metrics.h", "AveragePrecision");
}
  explicit AveragePrecision(const Options& opts) : opts_(opts) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSutilsPSimage_metricsDTh mht_7(mht_7_v, 289, "", "./tensorflow/lite/tools/evaluation/stages/utils/image_metrics.h", "AveragePrecision");
}

  // Given a sequence of precision-recall points ordered by the recall in
  // non-increasing order, returns the average of maximum precisions at
  // different recall values (0.0, 0.1, 0.2, ..., 0.9, 1.0).
  // The p-r pairs at these fixed recall points will be written to pr_out, if
  // it is not null_ptr.
  float FromPRCurve(const std::vector<PR>& pr,
                    std::vector<PR>* pr_out = nullptr);

  // An axis aligned bounding box for an image with id 'imageid'.  Score
  // indicates its confidence.
  //
  // 'difficult' is a special bit specific to Pascal VOC dataset and tasks using
  // the data. If 'difficult' is true, by convention, the box is often ignored
  // during the AP calculation. I.e., if a predicted box matches a 'difficult'
  // ground box, this predicted box is ignored as if the model does not make
  // such a prediction.

  // Given the set of ground truth boxes and a set of predicted boxes, returns
  // the average of the maximum precisions at different recall values.
  float FromBoxes(const std::vector<Detection>& groundtruth,
                  const std::vector<Detection>& prediction,
                  std::vector<PR>* pr_out = nullptr);

 private:
  Options opts_;
};

}  // namespace image
}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_UTILS_IMAGE_METRICS_H_
