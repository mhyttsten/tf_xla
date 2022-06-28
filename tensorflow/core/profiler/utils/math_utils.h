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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_MATH_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_MATH_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh() {
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


#include <cstdint>

namespace tensorflow {
namespace profiler {

// Converts among different SI units.
// https://en.wikipedia.org/wiki/International_System_of_Units
// NOTE: We use uint64 for picos and nanos, which are used in
// storage, and double for other units that are used in the UI.
inline double PicoToNano(uint64_t p) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_0(mht_0_v, 197, "", "./tensorflow/core/profiler/utils/math_utils.h", "PicoToNano");
 return p / 1E3; }
inline double PicoToMicro(uint64_t p) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_1(mht_1_v, 201, "", "./tensorflow/core/profiler/utils/math_utils.h", "PicoToMicro");
 return p / 1E6; }
inline double PicoToMilli(uint64_t p) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_2(mht_2_v, 205, "", "./tensorflow/core/profiler/utils/math_utils.h", "PicoToMilli");
 return p / 1E9; }
inline double PicoToUni(uint64_t p) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_3(mht_3_v, 209, "", "./tensorflow/core/profiler/utils/math_utils.h", "PicoToUni");
 return p / 1E12; }
inline uint64_t NanoToPico(uint64_t n) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_4(mht_4_v, 213, "", "./tensorflow/core/profiler/utils/math_utils.h", "NanoToPico");
 return n * 1000; }
inline double NanoToMicro(uint64_t n) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_5(mht_5_v, 217, "", "./tensorflow/core/profiler/utils/math_utils.h", "NanoToMicro");
 return n / 1E3; }
inline double NanoToMilli(uint64_t n) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_6(mht_6_v, 221, "", "./tensorflow/core/profiler/utils/math_utils.h", "NanoToMilli");
 return n / 1E6; }
inline double MicroToNano(double u) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_7(mht_7_v, 225, "", "./tensorflow/core/profiler/utils/math_utils.h", "MicroToNano");
 return u * 1E3; }
inline double MicroToMilli(double u) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_8(mht_8_v, 229, "", "./tensorflow/core/profiler/utils/math_utils.h", "MicroToMilli");
 return u / 1E3; }
inline uint64_t MilliToPico(double m) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_9(mht_9_v, 233, "", "./tensorflow/core/profiler/utils/math_utils.h", "MilliToPico");
 return m * 1E9; }
inline uint64_t MilliToNano(double m) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_10(mht_10_v, 237, "", "./tensorflow/core/profiler/utils/math_utils.h", "MilliToNano");
 return m * 1E6; }
inline double MilliToUni(double m) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_11(mht_11_v, 241, "", "./tensorflow/core/profiler/utils/math_utils.h", "MilliToUni");
 return m / 1E3; }
inline uint64_t UniToPico(double uni) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_12(mht_12_v, 245, "", "./tensorflow/core/profiler/utils/math_utils.h", "UniToPico");
 return uni * 1E12; }
inline uint64_t UniToNano(double uni) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_13(mht_13_v, 249, "", "./tensorflow/core/profiler/utils/math_utils.h", "UniToNano");
 return uni * 1E9; }
inline double UniToMicro(double uni) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_14(mht_14_v, 253, "", "./tensorflow/core/profiler/utils/math_utils.h", "UniToMicro");
 return uni * 1E6; }
inline double GigaToUni(double giga) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_15(mht_15_v, 257, "", "./tensorflow/core/profiler/utils/math_utils.h", "GigaToUni");
 return giga * 1E9; }
inline double GigaToTera(double giga) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_16(mht_16_v, 261, "", "./tensorflow/core/profiler/utils/math_utils.h", "GigaToTera");
 return giga / 1E3; }
inline double TeraToGiga(double tera) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_17(mht_17_v, 265, "", "./tensorflow/core/profiler/utils/math_utils.h", "TeraToGiga");
 return tera * 1E3; }

// Convert from clock cycles to seconds.
inline double CyclesToSeconds(double cycles, double frequency_hz) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_18(mht_18_v, 271, "", "./tensorflow/core/profiler/utils/math_utils.h", "CyclesToSeconds");

  // cycles / (cycles/s) = s.
  return cycles / frequency_hz;
}

// Checks the divisor and returns 0 to avoid divide by zero.
inline double SafeDivide(double dividend, double divisor) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_19(mht_19_v, 280, "", "./tensorflow/core/profiler/utils/math_utils.h", "SafeDivide");

  constexpr double kEpsilon = 1.0E-10;
  if ((-kEpsilon < divisor) && (divisor < kEpsilon)) return 0.0;
  return dividend / divisor;
}

inline double GibiToGiga(double gibi) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_20(mht_20_v, 289, "", "./tensorflow/core/profiler/utils/math_utils.h", "GibiToGiga");
 return gibi * ((1 << 30) / 1.0e9); }
inline double GigaToGibi(double giga) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_21(mht_21_v, 293, "", "./tensorflow/core/profiler/utils/math_utils.h", "GigaToGibi");
 return giga / ((1 << 30) / 1.0e9); }

// Calculates GiB/s.
inline double GibibytesPerSecond(double gigabytes, double ns) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPSmath_utilsDTh mht_22(mht_22_v, 299, "", "./tensorflow/core/profiler/utils/math_utils.h", "GibibytesPerSecond");

  return GigaToGibi(SafeDivide(gigabytes, ns));
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_MATH_UTILS_H_
