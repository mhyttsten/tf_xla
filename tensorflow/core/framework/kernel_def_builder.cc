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
class MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc() {
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

#include "tensorflow/core/framework/kernel_def_builder.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/kernel_def.pb.h"

namespace tensorflow {

KernelDefBuilder::KernelDefBuilder(const char* op_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc mht_0(mht_0_v, 193, "", "./tensorflow/core/framework/kernel_def_builder.cc", "KernelDefBuilder::KernelDefBuilder");

  kernel_def_ = new KernelDef;
  kernel_def_->set_op(op_name);
}

KernelDefBuilder::~KernelDefBuilder() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc mht_1(mht_1_v, 201, "", "./tensorflow/core/framework/kernel_def_builder.cc", "KernelDefBuilder::~KernelDefBuilder");

  DCHECK(kernel_def_ == nullptr) << "Did not call Build()";
}

KernelDefBuilder& KernelDefBuilder::Device(const char* device_type) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("device_type: \"" + (device_type == nullptr ? std::string("nullptr") : std::string((char*)device_type)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc mht_2(mht_2_v, 209, "", "./tensorflow/core/framework/kernel_def_builder.cc", "KernelDefBuilder::Device");

  kernel_def_->set_device_type(device_type);
  return *this;
}

template <>
KernelDefBuilder& KernelDefBuilder::AttrConstraint<int64_t>(
    const char* attr_name, gtl::ArraySlice<int64_t> allowed) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc mht_3(mht_3_v, 220, "", "./tensorflow/core/framework/kernel_def_builder.cc", "KernelDefBuilder::AttrConstraint<int64_t>");

  auto* constraint = kernel_def_->add_constraint();
  constraint->set_name(attr_name);
  auto* allowed_values = constraint->mutable_allowed_values()->mutable_list();
  for (const int64_t integer : allowed) {
    allowed_values->add_i(integer);
  }
  return *this;
}

template <>
KernelDefBuilder& KernelDefBuilder::AttrConstraint<int64_t>(
    const char* attr_name, int64_t allowed) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc mht_4(mht_4_v, 236, "", "./tensorflow/core/framework/kernel_def_builder.cc", "KernelDefBuilder::AttrConstraint<int64_t>");

  return AttrConstraint(
      attr_name,
      gtl::ArraySlice<int64_t>(std::initializer_list<int64_t>({allowed})));
}

template <>
KernelDefBuilder& KernelDefBuilder::AttrConstraint<string>(
    const char* attr_name, gtl::ArraySlice<string> allowed) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc mht_5(mht_5_v, 248, "", "./tensorflow/core/framework/kernel_def_builder.cc", "KernelDefBuilder::AttrConstraint<string>");

  auto* constraint = kernel_def_->add_constraint();
  constraint->set_name(attr_name);
  auto* allowed_values = constraint->mutable_allowed_values()->mutable_list();
  for (const auto& str : allowed) {
    allowed_values->add_s(str);
  }
  return *this;
}

template <>
KernelDefBuilder& KernelDefBuilder::AttrConstraint<string>(
    const char* attr_name, string allowed) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_6_v.push_back("allowed: \"" + allowed + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc mht_6(mht_6_v, 265, "", "./tensorflow/core/framework/kernel_def_builder.cc", "KernelDefBuilder::AttrConstraint<string>");

  return AttrConstraint(
      attr_name,
      gtl::ArraySlice<string>(std::initializer_list<string>({allowed})));
}

template <>
KernelDefBuilder& KernelDefBuilder::AttrConstraint<const char*>(
    const char* attr_name, gtl::ArraySlice<const char*> allowed) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc mht_7(mht_7_v, 277, "", "./tensorflow/core/framework/kernel_def_builder.cc", ">");

  auto* constraint = kernel_def_->add_constraint();
  constraint->set_name(attr_name);
  auto* allowed_values = constraint->mutable_allowed_values()->mutable_list();
  for (const auto& str : allowed) {
    allowed_values->add_s(str);
  }
  return *this;
}

template <>
KernelDefBuilder& KernelDefBuilder::AttrConstraint<const char*>(
    const char* attr_name, const char* allowed) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_8_v.push_back("allowed: \"" + (allowed == nullptr ? std::string("nullptr") : std::string((char*)allowed)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc mht_8(mht_8_v, 294, "", "./tensorflow/core/framework/kernel_def_builder.cc", ">");

  return AttrConstraint(attr_name,
                        gtl::ArraySlice<const char*>(
                            std::initializer_list<const char*>({allowed})));
}

template <>
KernelDefBuilder& KernelDefBuilder::AttrConstraint<bool>(const char* attr_name,
                                                         bool allowed) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc mht_9(mht_9_v, 306, "", "./tensorflow/core/framework/kernel_def_builder.cc", "KernelDefBuilder::AttrConstraint<bool>");

  auto* constraint = kernel_def_->add_constraint();
  constraint->set_name(attr_name);
  auto* allowed_values = constraint->mutable_allowed_values()->mutable_list();
  allowed_values->add_b(allowed);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(
    const char* attr_name, gtl::ArraySlice<DataType> allowed) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc mht_10(mht_10_v, 319, "", "./tensorflow/core/framework/kernel_def_builder.cc", "KernelDefBuilder::TypeConstraint");

  auto* constraint = kernel_def_->add_constraint();
  constraint->set_name(attr_name);
  auto* allowed_values = constraint->mutable_allowed_values()->mutable_list();
  for (DataType dt : allowed) {
    allowed_values->add_type(dt);
  }
  return *this;
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(const char* attr_name,
                                                   DataType allowed) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc mht_11(mht_11_v, 334, "", "./tensorflow/core/framework/kernel_def_builder.cc", "KernelDefBuilder::TypeConstraint");

  auto* constraint = kernel_def_->add_constraint();
  constraint->set_name(attr_name);
  constraint->mutable_allowed_values()->mutable_list()->add_type(allowed);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::HostMemory(const char* arg_name) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("arg_name: \"" + (arg_name == nullptr ? std::string("nullptr") : std::string((char*)arg_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc mht_12(mht_12_v, 345, "", "./tensorflow/core/framework/kernel_def_builder.cc", "KernelDefBuilder::HostMemory");

  kernel_def_->add_host_memory_arg(arg_name);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::Label(const char* label) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("label: \"" + (label == nullptr ? std::string("nullptr") : std::string((char*)label)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc mht_13(mht_13_v, 354, "", "./tensorflow/core/framework/kernel_def_builder.cc", "KernelDefBuilder::Label");

  CHECK_EQ(kernel_def_->label(), "")
      << "Trying to set a kernel's label a second time: '" << label
      << "' in: " << kernel_def_->DebugString();
  kernel_def_->set_label(label);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::Priority(int32_t priority) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc mht_14(mht_14_v, 365, "", "./tensorflow/core/framework/kernel_def_builder.cc", "KernelDefBuilder::Priority");

  kernel_def_->set_priority(priority);
  return *this;
}

const KernelDef* KernelDefBuilder::Build() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSkernel_def_builderDTcc mht_15(mht_15_v, 373, "", "./tensorflow/core/framework/kernel_def_builder.cc", "KernelDefBuilder::Build");

  KernelDef* r = kernel_def_;
  kernel_def_ = nullptr;
  return r;
}

}  // namespace tensorflow
