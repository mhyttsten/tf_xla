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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_entrypointDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_entrypointDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_entrypointDTcc() {
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
#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <thread>  // NOLINT: only used on Android, where std::thread is allowed

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/runner.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/set_big_core_affinity.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner.h"
#include "tensorflow/lite/nnapi/sl/include/SupportLibrary.h"

namespace tflite {
namespace acceleration {

extern "C" {
int Java_org_tensorflow_lite_acceleration_validation_entrypoint(int argc,
                                                                char** argv) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSvalidator_runner_entrypointDTcc mht_0(mht_0_v, 209, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/validator_runner_entrypoint.cc", "Java_org_tensorflow_lite_acceleration_validation_entrypoint");

  if (argc < 6) return 1;
  // argv[1] is the helper binary name
  // argv[2] is the function name
  std::string model_path = argv[3];
  std::string storage_path = argv[4];
  // argv[5] is data directory path.
  // argv[6] if present is the NNAPI SL path
  std::string nnapi_sl_path = argc > 6 ? argv[6] : "";
  FileLock lock(storage_path + ".child_lock");
  if (!lock.TryLock()) {
    return kMinibenchmarkChildProcessAlreadyRunning;
  }
  FlatbufferStorage<BenchmarkEvent> storage(storage_path);
  MinibenchmarkStatus status = storage.Read();
  if (status != kMinibenchmarkSuccess) {
    return status;
  }
  status = kMinibenchmarkNoValidationRequestFound;
  TFLiteSettingsT tflite_settings;

  int32_t set_big_core_affinity_errno = SetBigCoresAffinity();
  if (set_big_core_affinity_errno != 0) {
    flatbuffers::FlatBufferBuilder fbb;
    storage.Append(
        &fbb,
        CreateBenchmarkEvent(
            fbb, CreateTFLiteSettings(fbb, &tflite_settings),
            BenchmarkEventType_RECOVERED_ERROR, /* result */ 0,
            CreateBenchmarkError(
                fbb, BenchmarkStage_UNKNOWN,
                kMinibenchmarkUnableToSetCpuAffinity, /*signal=*/0,
                /*error_code=*/0,
                /*mini_benchmark_error_code=*/set_big_core_affinity_errno),
            Validator::BootTimeMicros(), Validator::WallTimeMicros()));
  }

  for (int i = storage.Count() - 1; i >= 0; i--) {
    const BenchmarkEvent* event = storage.Get(i);
    if (event->event_type() == BenchmarkEventType_START) {
      event->tflite_settings()->UnPackTo(&tflite_settings);

      std::unique_ptr<const ::tflite::nnapi::NnApiSupportLibrary>
          nnapi_sl_handle;
      if (tflite_settings.nnapi_settings && !nnapi_sl_path.empty()) {
        // We are not calling dlclose, it will be done once the
        // validator process ends.
        nnapi_sl_handle =
            ::tflite::nnapi::loadNnApiSupportLibrary(nnapi_sl_path);

        if (!nnapi_sl_handle) {
          status = kMiniBenchmarkCannotLoadSupportLibrary;
          break;
        }

        tflite_settings.nnapi_settings->support_library_handle =
            reinterpret_cast<uint64_t>(nnapi_sl_handle->getFL5());
      }

      flatbuffers::FlatBufferBuilder fbb;
      fbb.Finish(
          CreateComputeSettings(fbb, ExecutionPreference_ANY,
                                CreateTFLiteSettings(fbb, &tflite_settings)));
      std::unique_ptr<Validator> validator;
      if (model_path.find("fd:") == 0) {  // NOLINT
        int model_fd, model_offset, model_size;
        if (sscanf(model_path.c_str(), "fd:%d:%d:%d", &model_fd, &model_offset,
                   &model_size) != 3) {
          status = kMinibenchmarkPreconditionNotMet;
        }
        validator = std::make_unique<Validator>(
            model_fd, model_offset, model_size,
            flatbuffers::GetRoot<ComputeSettings>(fbb.GetBufferPointer()));
      } else {
        validator = std::make_unique<Validator>(
            model_path,
            flatbuffers::GetRoot<ComputeSettings>(fbb.GetBufferPointer()));
      }
      Validator::Results results;
      status = validator->RunValidation(&results);
      if (status != kMinibenchmarkSuccess) {
        break;
      }
      fbb.Reset();
      std::vector<int64_t> initialization_times{results.compilation_time_us};
      std::vector<flatbuffers::Offset<tflite::BenchmarkMetric>> metrics;
      for (const auto& name_and_values : results.metrics) {
        metrics.push_back(
            CreateBenchmarkMetric(fbb, fbb.CreateString(name_and_values.first),
                                  fbb.CreateVector(name_and_values.second)));
      }
      return storage.Append(
          &fbb,
          CreateBenchmarkEvent(
              fbb, CreateTFLiteSettings(fbb, &tflite_settings),
              BenchmarkEventType_END,
              CreateBenchmarkResult(fbb, fbb.CreateVector(initialization_times),
                                    fbb.CreateVector(results.execution_time_us),
                                    0, results.ok, fbb.CreateVector(metrics)),
              /* error */ 0, Validator::BootTimeMicros(),
              Validator::WallTimeMicros()));
    }
  }
  flatbuffers::FlatBufferBuilder fbb;
  return storage.Append(
      &fbb, CreateBenchmarkEvent(
                fbb, CreateTFLiteSettings(fbb, &tflite_settings),
                BenchmarkEventType_ERROR, /* result */ 0,
                CreateBenchmarkError(fbb, BenchmarkStage_UNKNOWN, status),
                Validator::BootTimeMicros(), Validator::WallTimeMicros()));
}
}  // extern "C"
}  // namespace acceleration
}  // namespace tflite

#endif  // !_WIN32
