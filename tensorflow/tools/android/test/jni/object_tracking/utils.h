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

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_UTILS_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_UTILS_H_
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
class MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh {
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
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh() {
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


#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include <cmath>  // for std::abs(float)

#ifndef HAVE_CLOCK_GETTIME
// Use gettimeofday() instead of clock_gettime().
#include <sys/time.h>
#endif  // ifdef HAVE_CLOCK_GETTIME

#include "tensorflow/tools/android/test/jni/object_tracking/logging.h"

// TODO(andrewharp): clean up these macros to use the codebase statndard.

// A very small number, generally used as the tolerance for accumulated
// floating point errors in bounds-checks.
#define EPSILON 0.00001f

#define SAFE_DELETE(pointer) {\
  if ((pointer) != NULL) {\
    LOGV("Safe deleting pointer: %s", #pointer);\
    delete (pointer);\
    (pointer) = NULL;\
  } else {\
    LOGV("Pointer already null: %s", #pointer);\
  }\
}


#ifdef __GOOGLE__

#define CHECK_ALWAYS(condition, format, ...) {\
  CHECK(condition) << StringPrintf(format, ##__VA_ARGS__);\
}

#define SCHECK(condition, format, ...) {\
  DCHECK(condition) << StringPrintf(format, ##__VA_ARGS__);\
}

#else

#define CHECK_ALWAYS(condition, format, ...) {\
  if (!(condition)) {\
    LOGE("CHECK FAILED (%s): " format, #condition, ##__VA_ARGS__);\
    abort();\
  }\
}

#ifdef SANITY_CHECKS
#define SCHECK(condition, format, ...) {\
  CHECK_ALWAYS(condition, format, ##__VA_ARGS__);\
}
#else
#define SCHECK(condition, format, ...) {}
#endif  // SANITY_CHECKS

#endif  // __GOOGLE__


#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) (((a) > (b)) ? (b) : (a))
#endif

inline static int64_t CurrentThreadTimeNanos() {
#ifdef HAVE_CLOCK_GETTIME
  struct timespec tm;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &tm);
  return tm.tv_sec * 1000000000LL + tm.tv_nsec;
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000000 + tv.tv_usec * 1000;
#endif
}

inline static int64_t CurrentRealTimeMillis() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_0(mht_0_v, 268, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "CurrentRealTimeMillis");

#ifdef HAVE_CLOCK_GETTIME
  struct timespec tm;
  clock_gettime(CLOCK_MONOTONIC, &tm);
  return tm.tv_sec * 1000LL + tm.tv_nsec / 1000000LL;
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
#endif
}


template<typename T>
inline static T Square(const T a) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_1(mht_1_v, 285, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "Square");

  return a * a;
}


template<typename T>
inline static T Clip(const T a, const T floor, const T ceil) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_2(mht_2_v, 294, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "Clip");

  SCHECK(ceil >= floor, "Bounds mismatch!");
  return (a <= floor) ? floor : ((a >= ceil) ? ceil : a);
}


template<typename T>
inline static int Floor(const T a) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_3(mht_3_v, 304, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "Floor");

  return static_cast<int>(a);
}


template<typename T>
inline static int Ceil(const T a) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_4(mht_4_v, 313, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "Ceil");

  return Floor(a) + 1;
}


template<typename T>
inline static bool InRange(const T a, const T min, const T max) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_5(mht_5_v, 322, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "InRange");

  return (a >= min) && (a <= max);
}


inline static bool ValidIndex(const int a, const int max) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_6(mht_6_v, 330, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "ValidIndex");

  return (a >= 0) && (a < max);
}


inline bool NearlyEqual(const float a, const float b, const float tolerance) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_7(mht_7_v, 338, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "NearlyEqual");

  return std::abs(a - b) < tolerance;
}


inline bool NearlyEqual(const float a, const float b) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_8(mht_8_v, 346, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "NearlyEqual");

  return NearlyEqual(a, b, EPSILON);
}


template<typename T>
inline static int Round(const float a) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_9(mht_9_v, 355, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "Round");

  return (a - static_cast<float>(floor(a) > 0.5f) ? ceil(a) : floor(a));
}


template<typename T>
inline static void Swap(T* const a, T* const b) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_10(mht_10_v, 364, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "Swap");

  // Cache out the VALUE of what's at a.
  T tmp = *a;
  *a = *b;

  *b = tmp;
}


static inline float randf() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_11(mht_11_v, 376, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "randf");

  return rand() / static_cast<float>(RAND_MAX);
}

static inline float randf(const float min_value, const float max_value) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_12(mht_12_v, 383, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "randf");

  return randf() * (max_value - min_value) + min_value;
}

static inline uint16_t RealToFixed115(const float real_number) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_13(mht_13_v, 390, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "RealToFixed115");

  SCHECK(InRange(real_number, 0.0f, 2048.0f),
        "Value out of range! %.2f", real_number);

  static const float kMult = 32.0f;
  const float round_add = (real_number > 0.0f) ? 0.5f : -0.5f;
  return static_cast<uint16_t>(real_number * kMult + round_add);
}

static inline float FixedToFloat115(const uint16_t fp_number) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_14(mht_14_v, 402, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "FixedToFloat115");

  const float kDiv = 32.0f;
  return (static_cast<float>(fp_number) / kDiv);
}

static inline int RealToFixed1616(const float real_number) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_15(mht_15_v, 410, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "RealToFixed1616");

  static const float kMult = 65536.0f;
  SCHECK(InRange(real_number, -kMult, kMult),
        "Value out of range! %.2f", real_number);

  const float round_add = (real_number > 0.0f) ? 0.5f : -0.5f;
  return static_cast<int>(real_number * kMult + round_add);
}

static inline float FixedToFloat1616(const int fp_number) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_16(mht_16_v, 422, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "FixedToFloat1616");

  const float kDiv = 65536.0f;
  return (static_cast<float>(fp_number) / kDiv);
}

template<typename T>
// produces numbers in range [0,2*M_PI] (rather than -PI,PI)
inline T FastAtan2(const T y, const T x) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_17(mht_17_v, 432, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "FastAtan2");

  static const T coeff_1 = (T)(M_PI / 4.0);
  static const T coeff_2 = (T)(3.0 * coeff_1);
  const T abs_y = fabs(y);
  T angle;
  if (x >= 0) {
    T r = (x - abs_y) / (x + abs_y);
    angle = coeff_1 - coeff_1 * r;
  } else {
    T r = (x + abs_y) / (abs_y - x);
    angle = coeff_2 - coeff_1 * r;
  }
  static const T PI_2 = 2.0 * M_PI;
  return y < 0 ? PI_2 - angle : angle;
}

#define NELEMS(X) (sizeof(X) / sizeof(X[0]))

namespace tf_tracking {

#ifdef __ARM_NEON
float ComputeMeanNeon(const float* const values, const int num_vals);

float ComputeStdDevNeon(const float* const values, const int num_vals,
                        const float mean);

float ComputeWeightedMeanNeon(const float* const values,
                              const float* const weights, const int num_vals);

float ComputeCrossCorrelationNeon(const float* const values1,
                                  const float* const values2,
                                  const int num_vals);
#endif

inline float ComputeMeanCpu(const float* const values, const int num_vals) {
  // Get mean.
  float sum = values[0];
  for (int i = 1; i < num_vals; ++i) {
    sum += values[i];
  }
  return sum / static_cast<float>(num_vals);
}


inline float ComputeMean(const float* const values, const int num_vals) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_18(mht_18_v, 479, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "ComputeMean");

  return
#ifdef __ARM_NEON
      (num_vals >= 8) ? ComputeMeanNeon(values, num_vals) :
#endif
                      ComputeMeanCpu(values, num_vals);
}


inline float ComputeStdDevCpu(const float* const values,
                              const int num_vals,
                              const float mean) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_19(mht_19_v, 493, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "ComputeStdDevCpu");

  // Get Std dev.
  float squared_sum = 0.0f;
  for (int i = 0; i < num_vals; ++i) {
    squared_sum += Square(values[i] - mean);
  }
  return sqrt(squared_sum / static_cast<float>(num_vals));
}


inline float ComputeStdDev(const float* const values,
                           const int num_vals,
                           const float mean) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_20(mht_20_v, 508, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "ComputeStdDev");

  return
#ifdef __ARM_NEON
      (num_vals >= 8) ? ComputeStdDevNeon(values, num_vals, mean) :
#endif
                      ComputeStdDevCpu(values, num_vals, mean);
}


// TODO(andrewharp): Accelerate with NEON.
inline float ComputeWeightedMean(const float* const values,
                                 const float* const weights,
                                 const int num_vals) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_21(mht_21_v, 523, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "ComputeWeightedMean");

  float sum = 0.0f;
  float total_weight = 0.0f;
  for (int i = 0; i < num_vals; ++i) {
    sum += values[i] * weights[i];
    total_weight += weights[i];
  }
  return sum / num_vals;
}


inline float ComputeCrossCorrelationCpu(const float* const values1,
                                        const float* const values2,
                                        const int num_vals) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_22(mht_22_v, 539, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "ComputeCrossCorrelationCpu");

  float sxy = 0.0f;
  for (int offset = 0; offset < num_vals; ++offset) {
    sxy += values1[offset] * values2[offset];
  }

  const float cross_correlation = sxy / num_vals;

  return cross_correlation;
}


inline float ComputeCrossCorrelation(const float* const values1,
                                     const float* const values2,
                                     const int num_vals) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_23(mht_23_v, 556, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "ComputeCrossCorrelation");

  return
#ifdef __ARM_NEON
      (num_vals >= 8) ? ComputeCrossCorrelationNeon(values1, values2, num_vals)
                      :
#endif
                      ComputeCrossCorrelationCpu(values1, values2, num_vals);
}


inline void NormalizeNumbers(float* const values, const int num_vals) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_24(mht_24_v, 569, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "NormalizeNumbers");

  // Find the mean and then subtract so that the new mean is 0.0.
  const float mean = ComputeMean(values, num_vals);
  VLOG(2) << "Mean is " << mean;
  float* curr_data = values;
  for (int i = 0; i < num_vals; ++i) {
    *curr_data -= mean;
    curr_data++;
  }

  // Now divide by the std deviation so the new standard deviation is 1.0.
  // The numbers might all be identical (and thus shifted to 0.0 now),
  // so only scale by the standard deviation if this is not the case.
  const float std_dev = ComputeStdDev(values, num_vals, 0.0f);
  if (std_dev > 0.0f) {
    VLOG(2) << "Std dev is " << std_dev;
    curr_data = values;
    for (int i = 0; i < num_vals; ++i) {
      *curr_data /= std_dev;
      curr_data++;
    }
  }
}


// Returns the determinant of a 2x2 matrix.
template<class T>
inline T FindDeterminant2x2(const T* const a) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_25(mht_25_v, 599, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "FindDeterminant2x2");

  // Determinant: (ad - bc)
  return a[0] * a[3] - a[1] * a[2];
}


// Finds the inverse of a 2x2 matrix.
// Returns true upon success, false if the matrix is not invertible.
template<class T>
inline bool Invert2x2(const T* const a, float* const a_inv) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPStoolsPSandroidPStestPSjniPSobject_trackingPSutilsDTh mht_26(mht_26_v, 611, "", "./tensorflow/tools/android/test/jni/object_tracking/utils.h", "Invert2x2");

  const float det = static_cast<float>(FindDeterminant2x2(a));
  if (fabs(det) < EPSILON) {
    return false;
  }
  const float inv_det = 1.0f / det;

  a_inv[0] = inv_det * static_cast<float>(a[3]);   // d
  a_inv[1] = inv_det * static_cast<float>(-a[1]);  // -b
  a_inv[2] = inv_det * static_cast<float>(-a[2]);  // -c
  a_inv[3] = inv_det * static_cast<float>(a[0]);   // a

  return true;
}

}  // namespace tf_tracking

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_UTILS_H_
