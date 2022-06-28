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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_registryDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_registryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_registryDTcc() {
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
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"

#include <set>
#include <utility>

#include "tensorflow/core/platform/mutex.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
namespace convert {

struct OpConverterRegistration {
  OpConverter converter;
  int priority;
};
class OpConverterRegistry::Impl {
 public:
  ~Impl() = default;

  InitOnStartupMarker Register(const string& name, const int priority,
                               OpConverter converter) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_registryDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.cc", "Register");

    mutex_lock lock(mu_);
    auto item = registry_.find(name);
    if (item != registry_.end()) {
      const int existing_priority = item->second.priority;
      if (priority <= existing_priority) {
        LOG(WARNING) << absl::StrCat(
            "Ignoring TF->TRT ", name, " op converter with priority ",
            existing_priority, " due to another converter with priority ",
            priority);
        return {};
      } else {
        LOG(WARNING) << absl::StrCat(
            "Overwriting TF->TRT ", name, " op converter with priority ",
            existing_priority, " using another converter with priority ",
            priority);
        registry_.erase(item);
      }
    }
    registry_.insert({name, OpConverterRegistration{converter, priority}});
    return {};
  }

  StatusOr<OpConverter> LookUp(const string& name) {
    mutex_lock lock(mu_);
    auto found = registry_.find(name);
    if (found != registry_.end()) {
      return found->second.converter;
    }
    return errors::NotFound("No converter for op ", name);
  }

  void Clear(const std::string& name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_registryDTcc mht_1(mht_1_v, 243, "", "./tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.cc", "Clear");

    mutex_lock lock(mu_);
    auto itr = registry_.find(name);
    if (itr == registry_.end()) {
      return;
    }
    registry_.erase(itr);
  }

 private:
  mutable mutex mu_;
  mutable std::unordered_map<std::string, OpConverterRegistration> registry_
      TF_GUARDED_BY(mu_);
};

OpConverterRegistry::OpConverterRegistry() : impl_(std::make_unique<Impl>()) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_registryDTcc mht_2(mht_2_v, 261, "", "./tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.cc", "OpConverterRegistry::OpConverterRegistry");
}

StatusOr<OpConverter> OpConverterRegistry::LookUp(const string& name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_registryDTcc mht_3(mht_3_v, 267, "", "./tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.cc", "OpConverterRegistry::LookUp");

  return impl_->LookUp(name);
}

InitOnStartupMarker OpConverterRegistry::Register(const string& name,
                                                  const int priority,
                                                  OpConverter converter) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_registryDTcc mht_4(mht_4_v, 277, "", "./tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.cc", "OpConverterRegistry::Register");

  return impl_->Register(name, priority, converter);
}

void OpConverterRegistry::Clear(const std::string& name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_registryDTcc mht_5(mht_5_v, 285, "", "./tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.cc", "OpConverterRegistry::Clear");
 impl_->Clear(name); }

OpConverterRegistry* GetOpConverterRegistry() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converter_registryDTcc mht_6(mht_6_v, 290, "", "./tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.cc", "GetOpConverterRegistry");

  static OpConverterRegistry* registry = new OpConverterRegistry();
  return registry;
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
