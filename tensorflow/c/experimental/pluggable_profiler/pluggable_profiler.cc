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
class MHTracer_DTPStensorflowPScPSexperimentalPSpluggable_profilerPSpluggable_profilerDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSpluggable_profilerPSpluggable_profilerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSpluggable_profilerPSpluggable_profilerDTcc() {
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
#include <vector>

#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/c_api_macros_internal.h"
#include "tensorflow/c/experimental/pluggable_profiler/pluggable_profiler_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/common_runtime/device/device_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/lib/profiler_factory.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tensorflow/core/profiler/profiler_options.pb.h"

namespace tensorflow {
namespace profiler {

namespace {

Status ValidateTPProfilerRegistrationParams(
    const TF_ProfilerRegistrationParams& params) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSpluggable_profilerPSpluggable_profilerDTcc mht_0(mht_0_v, 205, "", "./tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.cc", "ValidateTPProfilerRegistrationParams");

  TF_VALIDATE_STRUCT_SIZE(TF_ProfilerRegistrationParams, params,
                          TF_PROFILER_REGISTRATION_PARAMS_STRUCT_SIZE);
  TF_VALIDATE_NOT_NULL(TF_ProfilerRegistrationParams, params, destroy_profiler);
  TF_VALIDATE_NOT_NULL(TF_ProfilerRegistrationParams, params,
                       destroy_profiler_fns);
  return Status::OK();
}

Status ValidateTPProfiler(const TP_Profiler& profiler) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSpluggable_profilerPSpluggable_profilerDTcc mht_1(mht_1_v, 217, "", "./tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.cc", "ValidateTPProfiler");

  TF_VALIDATE_STRUCT_SIZE(TP_Profiler, profiler, TP_PROFILER_STRUCT_SIZE);
  TF_VALIDATE_NOT_NULL(TP_Profiler, profiler, device_type);
  TF_RETURN_IF_ERROR(
      tensorflow::device_utils::ValidateDeviceType(profiler.device_type));
  return Status::OK();
}

Status ValidateTPProfilerFns(const TP_ProfilerFns& profiler_fns) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSpluggable_profilerPSpluggable_profilerDTcc mht_2(mht_2_v, 228, "", "./tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.cc", "ValidateTPProfilerFns");

  TF_VALIDATE_STRUCT_SIZE(TP_ProfilerFns, profiler_fns,
                          TF_PROFILER_FNS_STRUCT_SIZE);
  TF_VALIDATE_NOT_NULL(TP_ProfilerFns, profiler_fns, start);
  TF_VALIDATE_NOT_NULL(TP_ProfilerFns, profiler_fns, stop);
  TF_VALIDATE_NOT_NULL(TP_ProfilerFns, profiler_fns, collect_data_xspace);
  return Status::OK();
}

class PluggableProfiler : public tensorflow::profiler::ProfilerInterface {
 public:
  // The caller must have validated profiler_fns and profiler.
  static std::unique_ptr<tensorflow::profiler::ProfilerInterface>
  CreatePluggableProfiler(const ProfileOptions& options, TP_Profiler profiler,
                          TP_ProfilerFns profiler_fns) {
    if (options.device_tracer_level() == 0) {
      return nullptr;
    }
    if (options.device_type() != ProfileOptions::PLUGGABLE_DEVICE &&
        options.device_type() != ProfileOptions::UNSPECIFIED) {
      return nullptr;
    }
    return absl::WrapUnique(new PluggableProfiler(profiler_fns, profiler));
  }

  Status Start() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSpluggable_profilerPSpluggable_profilerDTcc mht_3(mht_3_v, 256, "", "./tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.cc", "Start");

    tensorflow::TF_StatusPtr status(TF_NewStatus());
    profiler_fns_.start(&profiler_, status.get());
    return tensorflow::StatusFromTF_Status(status.get());
  }

  Status Stop() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSpluggable_profilerPSpluggable_profilerDTcc mht_4(mht_4_v, 265, "", "./tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.cc", "Stop");

    tensorflow::TF_StatusPtr status(TF_NewStatus());
    profiler_fns_.stop(&profiler_, status.get());
    return tensorflow::StatusFromTF_Status(status.get());
  }

  Status CollectData(XSpace* space) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSpluggable_profilerPSpluggable_profilerDTcc mht_5(mht_5_v, 274, "", "./tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.cc", "CollectData");

    tensorflow::TF_StatusPtr status(TF_NewStatus());
    // Get size of buffer required for Plugin to serialize XSpace into it.
    size_t size_in_bytes;
    profiler_fns_.collect_data_xspace(&profiler_, /*buffer=*/nullptr,
                                      &size_in_bytes, status.get());

    if (size_in_bytes <= 0)
      return tensorflow::StatusFromTF_Status(status.get());

    // Prepare an appropriately sized buffer.
    std::vector<uint8_t> buffer(size_in_bytes);
    profiler_fns_.collect_data_xspace(&profiler_, buffer.data(), &size_in_bytes,
                                      status.get());
    // Deserialize XSpace from the buffer and return it.
    XSpace plugin_space;
    plugin_space.ParseFromArray(buffer.data(), buffer.size());
    for (XPlane& plugin_plane : *plugin_space.mutable_planes()) {
      XPlane* plane = space->add_planes();
      plane->Swap(&plugin_plane);
    }
    return tensorflow::StatusFromTF_Status(status.get());
  }

 private:
  PluggableProfiler(TP_ProfilerFns profiler_fns, TP_Profiler profiler)
      : profiler_fns_(profiler_fns), profiler_(profiler) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSpluggable_profilerPSpluggable_profilerDTcc mht_6(mht_6_v, 303, "", "./tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.cc", "PluggableProfiler");
}
  TP_ProfilerFns profiler_fns_;
  TP_Profiler profiler_;
};

class PluggableProfilerFactory {
 public:
  PluggableProfilerFactory(TP_Profiler profiler,
                           void (*destroy_profiler)(TP_Profiler*),
                           TP_ProfilerFns profiler_fns,
                           void (*destroy_profiler_fns)(TP_ProfilerFns*))
      : profiler_(std::move(profiler)),
        destroy_profiler_(destroy_profiler),
        profiler_fns_(std::move(profiler_fns)),
        destroy_profiler_fns_(destroy_profiler_fns) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSpluggable_profilerPSpluggable_profilerDTcc mht_7(mht_7_v, 320, "", "./tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.cc", "PluggableProfilerFactory");
}

  ~PluggableProfilerFactory() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSpluggable_profilerPSpluggable_profilerDTcc mht_8(mht_8_v, 325, "", "./tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.cc", "~PluggableProfilerFactory");

    destroy_profiler_(&profiler_);
    destroy_profiler_fns_(&profiler_fns_);
  }

  std::unique_ptr<tensorflow::profiler::ProfilerInterface>
  CreatePluggableProfiler(const ProfileOptions& options) {
    return PluggableProfiler::CreatePluggableProfiler(options, profiler_,
                                                      profiler_fns_);
  }

 private:
  TP_Profiler profiler_{TP_PROFILER_STRUCT_SIZE};
  void (*destroy_profiler_)(TP_Profiler*);
  TP_ProfilerFns profiler_fns_{TP_PROFILER_FNS_STRUCT_SIZE};
  void (*destroy_profiler_fns_)(TP_ProfilerFns*);
};

}  // namespace

Status InitPluginProfiler(TFInitProfilerFn init_fn) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSpluggable_profilerPSpluggable_profilerDTcc mht_9(mht_9_v, 348, "", "./tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.cc", "InitPluginProfiler");

  TF_ProfilerRegistrationParams params{
      TF_PROFILER_REGISTRATION_PARAMS_STRUCT_SIZE};
  TP_Profiler profiler{TP_PROFILER_STRUCT_SIZE};
  TP_ProfilerFns profiler_fns{TP_PROFILER_FNS_STRUCT_SIZE};
  params.major_version = TP_MAJOR;
  params.minor_version = TP_MINOR;
  params.patch_version = TP_PATCH;
  params.profiler = &profiler;
  params.profiler_fns = &profiler_fns;
  tensorflow::TF_StatusPtr status(TF_NewStatus());
  init_fn(&params, status.get());
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(status.get()));
  TF_RETURN_IF_ERROR(ValidateTPProfilerRegistrationParams(params));
  TF_RETURN_IF_ERROR(ValidateTPProfiler(profiler));
  TF_RETURN_IF_ERROR(ValidateTPProfilerFns(profiler_fns));

  PluggableProfilerFactory factory(std::move(profiler), params.destroy_profiler,
                                   std::move(profiler_fns),
                                   params.destroy_profiler_fns);
  std::function<std::unique_ptr<ProfilerInterface>(const ProfileOptions&)>
      create_func = [factory = std::move(factory)](
                        const ProfileOptions& options) mutable {
        return factory.CreatePluggableProfiler(options);
      };

  tensorflow::profiler::RegisterProfilerFactory(std::move(create_func));
  return Status::OK();
}

}  // namespace profiler
}  // namespace tensorflow
