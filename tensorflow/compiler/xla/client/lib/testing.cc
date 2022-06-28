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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPStestingDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPStestingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPStestingDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/testing.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {
namespace {

// Calculates the number of bytes required to store the data within the
// specified shape. In case of a (nested) tuple shape this is the total byte
// size of all sub-shapes within the tuple.
int64_t DataSizeOfShape(const Shape& shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPStestingDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xla/client/lib/testing.cc", "DataSizeOfShape");

  if (shape.IsArray()) {
    return ShapeUtil::ByteSizeOf(shape);
  }

  int64_t total_size = 0;
  for (const Shape& s : shape.tuple_shapes()) {
    total_size += DataSizeOfShape(s);
  }
  return total_size;
}

// Creates a XlaOp for an op what generates fake data with the given shape.
XlaOp BuildFakeDataOpOnDevice(const Shape& shape, XlaBuilder* builder) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSlibPStestingDTcc mht_1(mht_1_v, 220, "", "./tensorflow/compiler/xla/client/lib/testing.cc", "BuildFakeDataOpOnDevice");

  if (shape.IsArray()) {
    return Broadcast(
        ConstantLiteral(builder, LiteralUtil::One(shape.element_type())),
        shape.dimensions());
  }
  std::vector<XlaOp> parts;
  const auto& tuple_shapes = shape.tuple_shapes();
  parts.reserve(tuple_shapes.size());
  for (const Shape& s : tuple_shapes) {
    parts.push_back(BuildFakeDataOpOnDevice(s, builder));
  }
  return Tuple(builder, parts);
}

std::unique_ptr<GlobalData> MakeFakeDataViaDeviceOrDie(
    const Shape& shape, Client* client, DebugOptions* debug_opts) {
  XlaBuilder b(absl::StrCat("make_fake_", ShapeUtil::HumanString(shape)));
  BuildFakeDataOpOnDevice(shape, &b);
  XlaComputation computation = b.Build().ConsumeValueOrDie();

  auto execution_options = CreateDefaultExecutionOptions();
  *execution_options.mutable_shape_with_output_layout() = shape.ToProto();
  if (debug_opts) {
    *execution_options.mutable_debug_options() = *debug_opts;
  }
  return client->Execute(computation, /*arguments=*/{}, &execution_options)
      .ConsumeValueOrDie();
}

}  // namespace

std::unique_ptr<GlobalData> MakeFakeDataOrDie(
    const Shape& shape, Client* client, DebugOptions* debug_opts /*=nullptr*/) {
  if (DataSizeOfShape(shape) < (1LL << 20)) {
    StatusOr<Literal> literal_status = MakeFakeLiteral(shape);
    if (!literal_status.ok()) {
      // If we got an Unimplemented error, fall back to making the fake data via
      // an on-device computation.
      CHECK_EQ(literal_status.status().code(),
               tensorflow::error::UNIMPLEMENTED);
      return MakeFakeDataViaDeviceOrDie(shape, client, debug_opts);
    }
    return client->TransferToServer(literal_status.ValueOrDie()).ValueOrDie();
  }

  // If the data is large, generate it on-device.
  return MakeFakeDataViaDeviceOrDie(shape, client, debug_opts);
}

std::vector<std::unique_ptr<GlobalData>> MakeFakeArgumentsOrDie(
    const XlaComputation& computation, Client* client,
    DebugOptions* debug_opts /*=nullptr*/) {
  CHECK(computation.proto().has_host_program_shape())
      << "Computation should have program shape.";
  auto program_shape = computation.proto().host_program_shape();

  std::vector<std::unique_ptr<GlobalData>> results;
  for (const ShapeProto& shape : program_shape.parameters()) {
    results.push_back(MakeFakeDataOrDie(Shape(shape), client, debug_opts));
  }
  return results;
}

}  // namespace xla
