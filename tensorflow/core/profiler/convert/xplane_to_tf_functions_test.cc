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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functions_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functions_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functions_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/xplane_to_tf_functions.h"

#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/tf_function.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_test_utils.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

const absl::string_view kEager = "eager";
const absl::string_view kConcrete = "concrete";
const absl::string_view kTracedNonXla = "traced-nonXla";
const absl::string_view kTracedXla = "traced-xla";
const absl::string_view kNotTracedNonXla = "notTraced-nonXla";
const absl::string_view kNotTracedXla = "notTraced-xla";

constexpr double kMaxError = 0.001;

TfFunctionDb ConvertXSpaceToTfFunctionDb(const XSpace& space) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPSxplane_to_tf_functions_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/profiler/convert/xplane_to_tf_functions_test.cc", "ConvertXSpaceToTfFunctionDb");

  TfFunctionDb result;
  const XPlane* host_plane = FindPlaneWithName(space, kHostThreadsPlaneName);
  if (host_plane) {
    XPlaneVisitor plane = CreateTfXPlaneVisitor(host_plane);
    plane.ForEachLine([&result](const XLineVisitor& line) {
      TfFunctionDb tf_function_db = ConvertHostThreadsXLineToTfFunctionDb(line);
      CombineTfFunctionDb(tf_function_db, &result);
    });
  }
  return result;
}

TEST(ConvertXPlaneToTfFunctions, CombineTwoThreads) {
  XSpace space;
  XPlaneBuilder host_plane_builder(space.add_planes());
  host_plane_builder.SetName(kHostThreadsPlaneName);
  host_plane_builder.ReserveLines(2);
  std::string kFunctionName = "decrement";

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateTfFunctionCallEvent(&host_plane_builder, &main_thread, kFunctionName,
                            10, 100, kTracedNonXla, 1);
  CreateTfFunctionCallEvent(&host_plane_builder, &main_thread, kFunctionName,
                            150, 20, kNotTracedNonXla, 2);
  CreateTfFunctionCallEvent(&host_plane_builder, &main_thread, kFunctionName,
                            200, 80, kTracedNonXla, 3);

  auto other_thread = host_plane_builder.GetOrCreateLine(1);
  CreateTfFunctionCallEvent(&host_plane_builder, &other_thread, kFunctionName,
                            20, 100, kTracedNonXla, 2);
  CreateTfFunctionCallEvent(&host_plane_builder, &other_thread, kFunctionName,
                            160, 20, kNotTracedNonXla, 2);
  CreateTfFunctionCallEvent(&host_plane_builder, &other_thread, kFunctionName,
                            210, 80, kTracedXla, 4);

  TfFunctionDb tf_function_db = ConvertXSpaceToTfFunctionDb(space);
  EXPECT_EQ(tf_function_db.tf_functions().size(), 1);
  EXPECT_EQ(tf_function_db.tf_functions().count(kFunctionName), 1);
  const TfFunction& tf_function =
      tf_function_db.tf_functions().at(kFunctionName);
  EXPECT_EQ(tf_function.total_tracing_count(), 4);
  EXPECT_EQ(tf_function.compiler(), MIXED_COMPILER);
  EXPECT_NEAR(tf_function.expensive_call_percent(), 90, kMaxError);

  const auto& metrics = tf_function.metrics();
  EXPECT_EQ(metrics.size(), 2);
  EXPECT_EQ(metrics.count(TRACED_MODE), 1);
  EXPECT_EQ(metrics.count(NOT_TRACED_MODE), 1);
  const auto& traced_mode = metrics.at(TRACED_MODE);
  EXPECT_EQ(traced_mode.count(), 4);
  EXPECT_EQ(traced_mode.self_time_ps(), 360);
  const auto& not_traced_mode = metrics.at(NOT_TRACED_MODE);
  EXPECT_EQ(not_traced_mode.count(), 2);
  EXPECT_EQ(not_traced_mode.self_time_ps(), 40);
}

TEST(ConvertXPlaneToTfFunctions, NestedFunctions) {
  XSpace space;
  XPlaneBuilder host_plane_builder(space.add_planes());
  host_plane_builder.SetName(kHostThreadsPlaneName);
  host_plane_builder.ReserveLines(1);
  std::string kOuterFunctionName = "outer";
  std::string kInnerFunctionName = "inner";

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateTfFunctionCallEvent(&host_plane_builder, &main_thread,
                            kOuterFunctionName, 10, 100, kTracedNonXla, 1);
  CreateTfFunctionCallEvent(&host_plane_builder, &main_thread,
                            kInnerFunctionName, 30, 40, kNotTracedXla, 0);
  TfFunctionDb tf_function_db = ConvertXSpaceToTfFunctionDb(space);
  EXPECT_EQ(tf_function_db.tf_functions().size(), 2);
  EXPECT_EQ(tf_function_db.tf_functions().count(kOuterFunctionName), 1);
  EXPECT_EQ(tf_function_db.tf_functions().count(kInnerFunctionName), 1);
  const TfFunction& outer =
      tf_function_db.tf_functions().at(kOuterFunctionName);
  EXPECT_EQ(outer.total_tracing_count(), 1);
  EXPECT_EQ(outer.compiler(), OTHER_COMPILER);
  EXPECT_NEAR(outer.expensive_call_percent(), 100, kMaxError);
  const auto& outer_metrics = outer.metrics();
  EXPECT_EQ(outer_metrics.size(), 1);
  EXPECT_EQ(outer_metrics.count(TRACED_MODE), 1);
  const auto& traced_mode = outer_metrics.at(TRACED_MODE);
  EXPECT_EQ(traced_mode.count(), 1);
  EXPECT_EQ(traced_mode.self_time_ps(), 60);
  const TfFunction& inner =
      tf_function_db.tf_functions().at(kInnerFunctionName);
  EXPECT_EQ(inner.total_tracing_count(), 0);
  EXPECT_EQ(inner.compiler(), XLA_COMPILER);
  EXPECT_NEAR(inner.expensive_call_percent(), 0, kMaxError);
  const auto& inner_metrics = inner.metrics();
  EXPECT_EQ(inner_metrics.size(), 1);
  EXPECT_EQ(inner_metrics.count(NOT_TRACED_MODE), 1);
  const auto& not_traced_mode = inner_metrics.at(NOT_TRACED_MODE);
  EXPECT_EQ(not_traced_mode.count(), 1);
  EXPECT_EQ(not_traced_mode.self_time_ps(), 40);
}

TEST(ConvertXPlaneToTfFunctions, EagerPlusConcrete) {
  XSpace space;
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(&space));
  host_plane_builder.ReserveLines(2);
  std::string kEagerFunctionName = "i_am_eager";
  std::string kConcreteFunctionName = "i_am_concrete";

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateTfFunctionCallEvent(&host_plane_builder, &main_thread,
                            kEagerFunctionName, 10, 200, kEager);
  auto other_thread = host_plane_builder.GetOrCreateLine(1);
  CreateTfFunctionCallEvent(&host_plane_builder, &other_thread,
                            kConcreteFunctionName, 20, 40, kConcrete);
  TfFunctionDb tf_function_db = ConvertXSpaceToTfFunctionDb(space);
  EXPECT_EQ(tf_function_db.tf_functions().size(), 2);
  EXPECT_EQ(tf_function_db.tf_functions().count(kEagerFunctionName), 1);
  EXPECT_EQ(tf_function_db.tf_functions().count(kConcreteFunctionName), 1);
  const TfFunction& eager =
      tf_function_db.tf_functions().at(kEagerFunctionName);
  EXPECT_EQ(eager.total_tracing_count(), 0);
  EXPECT_EQ(eager.compiler(), INVALID_COMPILER);
  EXPECT_NEAR(eager.expensive_call_percent(), 100, kMaxError);
  const auto& eager_metrics = eager.metrics();
  EXPECT_EQ(eager_metrics.size(), 1);
  EXPECT_EQ(eager_metrics.count(EAGER_MODE), 1);
  const auto& eager_mode = eager_metrics.at(EAGER_MODE);
  EXPECT_EQ(eager_mode.count(), 1);
  EXPECT_EQ(eager_mode.self_time_ps(), 200);
  const TfFunction& concrete =
      tf_function_db.tf_functions().at(kConcreteFunctionName);
  EXPECT_EQ(concrete.total_tracing_count(), 0);
  EXPECT_EQ(concrete.compiler(), INVALID_COMPILER);
  EXPECT_NEAR(concrete.expensive_call_percent(), 0, kMaxError);
  const auto& concrete_metrics = concrete.metrics();
  EXPECT_EQ(concrete_metrics.size(), 1);
  EXPECT_EQ(concrete_metrics.count(CONCRETE_MODE), 1);
  const auto& concrete_mode = concrete_metrics.at(CONCRETE_MODE);
  EXPECT_EQ(concrete_mode.count(), 1);
  EXPECT_EQ(concrete_mode.self_time_ps(), 40);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
