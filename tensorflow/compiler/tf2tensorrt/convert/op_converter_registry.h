/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OP_CONVERTER_REGISTRY_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OP_CONVERTER_REGISTRY_H_
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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_registryDTh {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_registryDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_registryDTh() {
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


#include <initializer_list>
#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include <array>
#include <type_traits>
#include <vector>

#include "tensorflow/compiler/tf2tensorrt/convert/op_converter.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

class OpConverterRegistry {
 public:
  OpConverterRegistry();
  ~OpConverterRegistry() = default;

  InitOnStartupMarker Register(const string& name, const int priority,
                               OpConverter converter);

  InitOnStartupMarker Register(const std::initializer_list<std::string>& names,
                               const int priority, OpConverter converter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_registryDTh mht_0(mht_0_v, 210, "", "./tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h", "Register");

    for (const auto& name : names) {
      Register(name, priority, converter);
    }
    return {};
  }

  template <typename T,
            typename std::enable_if<std::is_convertible<
                typename T::value_type, std::string>::value>::type* = nullptr>
  InitOnStartupMarker Register(const T& names, const int priority,
                               OpConverter converter) {
    for (const auto& name : names) {
      Register(name, priority, converter);
    }
    return {};
  }

  // Clear all registered converters for the given Tensorflow operation name.
  void Clear(const std::string& name);

  StatusOr<OpConverter> LookUp(const string& name);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

OpConverterRegistry* GetOpConverterRegistry();

class RegisterOpConverter {
 public:
  RegisterOpConverter(const string& name, const int priority,
                      OpConverter converter) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_registryDTh mht_1(mht_1_v, 247, "", "./tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h", "RegisterOpConverter");

    GetOpConverterRegistry()->Register(name, priority, converter);
  }
};

constexpr int kDefaultConverterPriority = 1;

}  // namespace convert
}  // namespace tensorrt

#define REGISTER_TRT_OP_CONVERTER_IMPL(ctr, func, priority, ...)    \
  static ::tensorflow::InitOnStartupMarker const                    \
      register_trt_op_converter##ctr TF_ATTRIBUTE_UNUSED =          \
          TF_INIT_ON_STARTUP_IF(true)                               \
          << tensorrt::convert::GetOpConverterRegistry()->Register( \
                 __VA_ARGS__, priority, func)

#define REGISTER_TRT_OP_CONVERTER(func, priority, ...)               \
  TF_NEW_ID_FOR_INIT(REGISTER_TRT_OP_CONVERTER_IMPL, func, priority, \
                     __VA_ARGS__)

#define REGISTER_DEFAULT_TRT_OP_CONVERTER(func, ...) \
  REGISTER_TRT_OP_CONVERTER(                         \
      func, tensorrt::convert::kDefaultConverterPriority, __VA_ARGS__)

}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OP_CONVERTER_REGISTRY_H_
