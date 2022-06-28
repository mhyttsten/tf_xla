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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutils_neonDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutils_neonDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutils_neonDTcc() {
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

// NEON implementations of Image methods for compatible devices.  Control
// should never enter this compilation unit on incompatible devices.

#ifdef __ARM_NEON

#include <arm_neon.h>

#include "tensorflow/tools/android/test/jni/object_tracking/geom.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image-inl.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image.h"
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

namespace tf_tracking {

inline static float GetSum(const float32x4_t& values) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutils_neonDTcc mht_0(mht_0_v, 199, "", "./tensorflow/tools/android/test/jni/object_tracking/utils_neon.cc", "GetSum");

  static float32_t summed_values[4];
  vst1q_f32(summed_values, values);
  return summed_values[0]
       + summed_values[1]
       + summed_values[2]
       + summed_values[3];
}


float ComputeMeanNeon(const float* const values, const int num_vals) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutils_neonDTcc mht_1(mht_1_v, 212, "", "./tensorflow/tools/android/test/jni/object_tracking/utils_neon.cc", "ComputeMeanNeon");

  SCHECK(num_vals >= 8, "Not enough values to merit NEON: %d", num_vals);

  const float32_t* const arm_vals = (const float32_t* const) values;
  float32x4_t accum = vdupq_n_f32(0.0f);

  int offset = 0;
  for (; offset <= num_vals - 4; offset += 4) {
    accum = vaddq_f32(accum, vld1q_f32(&arm_vals[offset]));
  }

  // Pull the accumulated values into a single variable.
  float sum = GetSum(accum);

  // Get the remaining 1 to 3 values.
  for (; offset < num_vals; ++offset) {
    sum += values[offset];
  }

  const float mean_neon = sum / static_cast<float>(num_vals);

#ifdef SANITY_CHECKS
  const float mean_cpu = ComputeMeanCpu(values, num_vals);
  SCHECK(NearlyEqual(mean_neon, mean_cpu, EPSILON * num_vals),
        "Neon mismatch with CPU mean! %.10f vs %.10f",
        mean_neon, mean_cpu);
#endif

  return mean_neon;
}


float ComputeStdDevNeon(const float* const values,
                        const int num_vals, const float mean) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutils_neonDTcc mht_2(mht_2_v, 248, "", "./tensorflow/tools/android/test/jni/object_tracking/utils_neon.cc", "ComputeStdDevNeon");

  SCHECK(num_vals >= 8, "Not enough values to merit NEON: %d", num_vals);

  const float32_t* const arm_vals = (const float32_t* const) values;
  const float32x4_t mean_vec = vdupq_n_f32(-mean);

  float32x4_t accum = vdupq_n_f32(0.0f);

  int offset = 0;
  for (; offset <= num_vals - 4; offset += 4) {
    const float32x4_t deltas =
        vaddq_f32(mean_vec, vld1q_f32(&arm_vals[offset]));

    accum = vmlaq_f32(accum, deltas, deltas);
  }

  // Pull the accumulated values into a single variable.
  float squared_sum = GetSum(accum);

  // Get the remaining 1 to 3 values.
  for (; offset < num_vals; ++offset) {
    squared_sum += Square(values[offset] - mean);
  }

  const float std_dev_neon = sqrt(squared_sum / static_cast<float>(num_vals));

#ifdef SANITY_CHECKS
  const float std_dev_cpu = ComputeStdDevCpu(values, num_vals, mean);
  SCHECK(NearlyEqual(std_dev_neon, std_dev_cpu, EPSILON * num_vals),
        "Neon mismatch with CPU std dev! %.10f vs %.10f",
        std_dev_neon, std_dev_cpu);
#endif

  return std_dev_neon;
}


float ComputeCrossCorrelationNeon(const float* const values1,
                                  const float* const values2,
                                  const int num_vals) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutils_neonDTcc mht_3(mht_3_v, 290, "", "./tensorflow/tools/android/test/jni/object_tracking/utils_neon.cc", "ComputeCrossCorrelationNeon");

  SCHECK(num_vals >= 8, "Not enough values to merit NEON: %d", num_vals);

  const float32_t* const arm_vals1 = (const float32_t* const) values1;
  const float32_t* const arm_vals2 = (const float32_t* const) values2;

  float32x4_t accum = vdupq_n_f32(0.0f);

  int offset = 0;
  for (; offset <= num_vals - 4; offset += 4) {
    accum = vmlaq_f32(accum,
                      vld1q_f32(&arm_vals1[offset]),
                      vld1q_f32(&arm_vals2[offset]));
  }

  // Pull the accumulated values into a single variable.
  float sxy = GetSum(accum);

  // Get the remaining 1 to 3 values.
  for (; offset < num_vals; ++offset) {
    sxy += values1[offset] * values2[offset];
  }

  const float cross_correlation_neon = sxy / num_vals;

#ifdef SANITY_CHECKS
  const float cross_correlation_cpu =
      ComputeCrossCorrelationCpu(values1, values2, num_vals);
  SCHECK(NearlyEqual(cross_correlation_neon, cross_correlation_cpu,
                    EPSILON * num_vals),
        "Neon mismatch with CPU cross correlation! %.10f vs %.10f",
        cross_correlation_neon, cross_correlation_cpu);
#endif

  return cross_correlation_neon;
}

}  // namespace tf_tracking

#endif  // __ARM_NEON
