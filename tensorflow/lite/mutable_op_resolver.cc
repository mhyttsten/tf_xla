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
class MHTracer_DTPStensorflowPSlitePSmutable_op_resolverDTcc {
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
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolverDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSmutable_op_resolverDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/mutable_op_resolver.h"

#include <string>
#include <unordered_map>
#include <utility>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/op_resolver_internal.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

const TfLiteRegistration* MutableOpResolver::FindOp(tflite::BuiltinOperator op,
                                                    int version) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolverDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/mutable_op_resolver.cc", "MutableOpResolver::FindOp");

  auto it = builtins_.find(std::make_pair(op, version));
  if (it != builtins_.end()) {
    return &it->second;
  }
  for (const OpResolver* other : other_op_resolvers_) {
    const TfLiteRegistration* result = other->FindOp(op, version);
    if (result != nullptr) {
      return result;
    }
  }
  return nullptr;
}

const TfLiteRegistration* MutableOpResolver::FindOp(const char* op,
                                                    int version) const {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolverDTcc mht_1(mht_1_v, 217, "", "./tensorflow/lite/mutable_op_resolver.cc", "MutableOpResolver::FindOp");

  auto it = custom_ops_.find(std::make_pair(op, version));
  if (it != custom_ops_.end()) {
    return &it->second;
  }
  for (const OpResolver* other : other_op_resolvers_) {
    const TfLiteRegistration* result = other->FindOp(op, version);
    if (result != nullptr) {
      return result;
    }
  }
  return nullptr;
}

void MutableOpResolver::AddBuiltin(tflite::BuiltinOperator op,
                                   const TfLiteRegistration* registration,
                                   int version) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolverDTcc mht_2(mht_2_v, 236, "", "./tensorflow/lite/mutable_op_resolver.cc", "MutableOpResolver::AddBuiltin");

  if (registration == nullptr) {
    // Under certain conditions, builtin TfLiteRegistration factory methods may
    // return null in the client library. This is generally benign, and we
    // silently suppress resulting AddBuiltin calls here.
    return;
  }
  TfLiteRegistration new_registration = *registration;
  new_registration.custom_name = nullptr;
  new_registration.builtin_code = op;
  new_registration.version = version;
  auto op_key = std::make_pair(op, version);
  builtins_[op_key] = new_registration;
  // The builtin op that is being added may be one that is not supported by
  // tflite::ops::builtin::BuiltinOpResolver. Or the TfLiteRegistration for this
  // builtin may be different than the one that BuiltinOpResolver would use,
  // which could lead to different semantics. Both of those cases are considered
  // "user defined ops".
  may_directly_contain_user_defined_ops_ = true;
}

void MutableOpResolver::AddBuiltin(tflite::BuiltinOperator op,
                                   const TfLiteRegistration* registration,
                                   int min_version, int max_version) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolverDTcc mht_3(mht_3_v, 262, "", "./tensorflow/lite/mutable_op_resolver.cc", "MutableOpResolver::AddBuiltin");

  for (int version = min_version; version <= max_version; ++version) {
    AddBuiltin(op, registration, version);
  }
}

void MutableOpResolver::AddCustom(const char* name,
                                  const TfLiteRegistration* registration,
                                  int version) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolverDTcc mht_4(mht_4_v, 274, "", "./tensorflow/lite/mutable_op_resolver.cc", "MutableOpResolver::AddCustom");

  TfLiteRegistration new_registration = *registration;
  new_registration.builtin_code = BuiltinOperator_CUSTOM;
  new_registration.custom_name = name;
  new_registration.version = version;
  auto op_key = std::make_pair(name, version);
  custom_ops_[op_key] = new_registration;
  may_directly_contain_user_defined_ops_ = true;
}

void MutableOpResolver::AddCustom(const char* name,
                                  const TfLiteRegistration* registration,
                                  int min_version, int max_version) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolverDTcc mht_5(mht_5_v, 290, "", "./tensorflow/lite/mutable_op_resolver.cc", "MutableOpResolver::AddCustom");

  for (int version = min_version; version <= max_version; ++version) {
    AddCustom(name, registration, version);
  }
}

void MutableOpResolver::AddAll(const MutableOpResolver& other) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolverDTcc mht_6(mht_6_v, 299, "", "./tensorflow/lite/mutable_op_resolver.cc", "MutableOpResolver::AddAll");

  // map::insert does not replace existing elements, and map::insert_or_assign
  // wasn't added until C++17.
  for (const auto& other_builtin : other.builtins_) {
    builtins_[other_builtin.first] = other_builtin.second;
  }
  for (const auto& other_custom_op : other.custom_ops_) {
    custom_ops_[other_custom_op.first] = other_custom_op.second;
  }
  other_op_resolvers_.insert(other_op_resolvers_.begin(),
                             other.other_op_resolvers_.begin(),
                             other.other_op_resolvers_.end());
}

void MutableOpResolver::ChainOpResolver(const OpResolver* other) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolverDTcc mht_7(mht_7_v, 316, "", "./tensorflow/lite/mutable_op_resolver.cc", "MutableOpResolver::ChainOpResolver");

  other_op_resolvers_.push_back(other);
}

bool MutableOpResolver::MayContainUserDefinedOps() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolverDTcc mht_8(mht_8_v, 323, "", "./tensorflow/lite/mutable_op_resolver.cc", "MutableOpResolver::MayContainUserDefinedOps");

  if (may_directly_contain_user_defined_ops_) {
    return true;
  }
  for (const OpResolver* other : other_op_resolvers_) {
    if (OpResolverInternal::MayContainUserDefinedOps(*other)) {
      return true;
    }
  }
  return false;
}

}  // namespace tflite
