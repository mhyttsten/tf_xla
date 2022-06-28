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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTcc() {
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

// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h"

#include "absl/strings/match.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/util.h"

namespace tpu_driver {

namespace {

typedef absl::flat_hash_map<
    std::string, std::function<xla::StatusOr<std::unique_ptr<TpuDriver>>(
                     const TpuDriverConfig&)>>
    DriverRegistryMap;

DriverRegistryMap* GetDriverRegistryMap() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/xla/python/tpu_driver/tpu_driver.cc", "GetDriverRegistryMap");

  static DriverRegistryMap* driver_registry = new DriverRegistryMap();
  return driver_registry;
}

int64_t ByteSizeOfPrimitiveType(xla::PrimitiveType primitive_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTcc mht_1(mht_1_v, 208, "", "./tensorflow/compiler/xla/python/tpu_driver/tpu_driver.cc", "ByteSizeOfPrimitiveType");

  switch (primitive_type) {
    case xla::PrimitiveType::PRED:
      return sizeof(int8_t);
    case xla::PrimitiveType::S8:
      return sizeof(int8_t);
    case xla::PrimitiveType::S16:
      return sizeof(int16_t);
    case xla::PrimitiveType::S32:
      return sizeof(int32_t);
    case xla::PrimitiveType::S64:
      return sizeof(int64_t);
    case xla::PrimitiveType::U8:
      return sizeof(uint8_t);
    case xla::PrimitiveType::U16:
      return sizeof(uint16_t);
    case xla::PrimitiveType::U32:
      return sizeof(uint32_t);
    case xla::PrimitiveType::U64:
      return sizeof(uint64_t);
    case xla::PrimitiveType::BF16:
      return sizeof(float) / 2;
    case xla::PrimitiveType::F16:
      return sizeof(float) / 2;
    case xla::PrimitiveType::F32:
      return sizeof(float);
    case xla::PrimitiveType::F64:
      return sizeof(double);
    case xla::PrimitiveType::C64:
      return sizeof(std::complex<float>);
    case xla::PrimitiveType::C128:
      return sizeof(std::complex<double>);
    case xla::PrimitiveType::TOKEN:
    case xla::PrimitiveType::TUPLE:
    case xla::PrimitiveType::OPAQUE_TYPE:
      LOG(FATAL) << PrimitiveType_Name(primitive_type)
                 << " primitive type has no definitive size";
    default:
      LOG(FATAL) << "Unhandled primitive type " << primitive_type;
  }
}

}  // namespace

/*static*/ int TpuDriverRegistry::RegisterDriver(
    const std::string& prefix,
    const std::function<xla::StatusOr<std::unique_ptr<TpuDriver>>(
        const TpuDriverConfig&)>& creator) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTcc mht_2(mht_2_v, 259, "", "./tensorflow/compiler/xla/python/tpu_driver/tpu_driver.cc", "TpuDriverRegistry::RegisterDriver");

  (*GetDriverRegistryMap())[prefix] = creator;
  return GetDriverRegistryMap()->size();
}

/*static*/ xla::StatusOr<std::unique_ptr<TpuDriver>> TpuDriverRegistry::Open(
    const TpuDriverConfig& config) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTcc mht_3(mht_3_v, 268, "", "./tensorflow/compiler/xla/python/tpu_driver/tpu_driver.cc", "TpuDriverRegistry::Open");

  for (const auto& driver : *GetDriverRegistryMap()) {
    if (absl::StartsWith(config.worker(), driver.first)) {
      return driver.second(config);
    }
  }
  return xla::NotFound("Unable to find driver in registry given worker: %s",
                       config.worker());
}

int64_t ComputeBytesFromShape(const xla::ShapeProto& shape) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTcc mht_4(mht_4_v, 281, "", "./tensorflow/compiler/xla/python/tpu_driver/tpu_driver.cc", "ComputeBytesFromShape");

  if (shape.tuple_shapes_size() > 0) {
    LOG(FATAL) << "Tuples are not supported at the moment.";
  }

  int64_t num_elems = 1;
  for (auto dim : shape.dimensions()) {
    num_elems *= dim;
  }

  return ByteSizeOfPrimitiveType(shape.element_type()) * num_elems;
}

}  // namespace tpu_driver
