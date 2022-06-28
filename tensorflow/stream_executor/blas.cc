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
class MHTracer_DTPStensorflowPSstream_executorPSblasDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPSblasDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSblasDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/blas.h"

#include "absl/strings/str_cat.h"

namespace stream_executor {
namespace blas {

std::string TransposeString(Transpose t) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSblasDTcc mht_0(mht_0_v, 192, "", "./tensorflow/stream_executor/blas.cc", "TransposeString");

  switch (t) {
    case Transpose::kNoTranspose:
      return "NoTranspose";
    case Transpose::kTranspose:
      return "Transpose";
    case Transpose::kConjugateTranspose:
      return "ConjugateTranspose";
    default:
      LOG(FATAL) << "Unknown transpose " << static_cast<int32>(t);
  }
}

std::string UpperLowerString(UpperLower ul) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSblasDTcc mht_1(mht_1_v, 208, "", "./tensorflow/stream_executor/blas.cc", "UpperLowerString");

  switch (ul) {
    case UpperLower::kUpper:
      return "Upper";
    case UpperLower::kLower:
      return "Lower";
    default:
      LOG(FATAL) << "Unknown upperlower " << static_cast<int32>(ul);
  }
}

std::string DiagonalString(Diagonal d) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSblasDTcc mht_2(mht_2_v, 222, "", "./tensorflow/stream_executor/blas.cc", "DiagonalString");

  switch (d) {
    case Diagonal::kUnit:
      return "Unit";
    case Diagonal::kNonUnit:
      return "NonUnit";
    default:
      LOG(FATAL) << "Unknown diagonal " << static_cast<int32>(d);
  }
}

std::string SideString(Side s) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSblasDTcc mht_3(mht_3_v, 236, "", "./tensorflow/stream_executor/blas.cc", "SideString");

  switch (s) {
    case Side::kLeft:
      return "Left";
    case Side::kRight:
      return "Right";
    default:
      LOG(FATAL) << "Unknown side " << static_cast<int32>(s);
  }
}

// -- AlgorithmConfig

std::string AlgorithmConfig::ToString() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSblasDTcc mht_4(mht_4_v, 252, "", "./tensorflow/stream_executor/blas.cc", "AlgorithmConfig::ToString");

  return absl::StrCat(algorithm_);
}

std::string ComputationTypeString(ComputationType ty) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSblasDTcc mht_5(mht_5_v, 259, "", "./tensorflow/stream_executor/blas.cc", "ComputationTypeString");

  switch (ty) {
    case ComputationType::kF16:
      return "f16";
    case ComputationType::kF32:
      return "f32";
    case ComputationType::kF64:
      return "f64";
    case ComputationType::kI32:
      return "i32";
    case ComputationType::kComplexF32:
      return "complex f32";
    case ComputationType::kComplexF64:
      return "complex f64";
    default:
      LOG(FATAL) << "Unknown ComputationType " << static_cast<int32>(ty);
  }
}

std::ostream& operator<<(std::ostream& os, ComputationType ty) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSblasDTcc mht_6(mht_6_v, 281, "", "./tensorflow/stream_executor/blas.cc", "operator<<");

  return os << ComputationTypeString(ty);
}

std::string DataTypeString(DataType ty) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSblasDTcc mht_7(mht_7_v, 288, "", "./tensorflow/stream_executor/blas.cc", "DataTypeString");

  switch (ty) {
    case DataType::kBF16:
      return "bf16";
    case DataType::kHalf:
      return "f16";
    case DataType::kFloat:
      return "f32";
    case DataType::kDouble:
      return "f64";
    case DataType::kInt8:
      return "i8";
    case DataType::kInt32:
      return "i32";
    case DataType::kComplexFloat:
      return "complex f32";
    case DataType::kComplexDouble:
      return "complex f64";
    default:
      LOG(FATAL) << "Unknown DataType " << static_cast<int32>(ty);
  }
}

std::ostream& operator<<(std::ostream& os, DataType ty) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSblasDTcc mht_8(mht_8_v, 314, "", "./tensorflow/stream_executor/blas.cc", "operator<<");

  return os << DataTypeString(ty);
}

}  // namespace blas
}  // namespace stream_executor
