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
class MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/tests/xla_compilation_cache_test_helper.h"

#include <string>

#include "absl/strings/match.h"
#include "tensorflow/compiler/jit/xla_compilation_cache.pb.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace {

// Creates a float tensor of linearly increasing values, starting from offset.
Tensor CreateInputTensor(const TensorShape& shape, float offset) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/jit/tests/xla_compilation_cache_test_helper.cc", "CreateInputTensor");

  Tensor tensor(DT_FLOAT, shape);
  for (int64 i = 0; i < tensor.flat<float>().size(); ++i) {
    tensor.flat<float>()(i) = offset + i;
  }
  return tensor;
}

NodeDef MakeNode(
    absl::string_view name, absl::string_view op,
    absl::Span<const std::string> inputs,
    absl::Span<
        const std::pair<std::string, FunctionDefHelper::AttrValueWrapper>>
        attrs) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   mht_1_v.push_back("op: \"" + std::string(op.data(), op.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTcc mht_1(mht_1_v, 217, "", "./tensorflow/compiler/jit/tests/xla_compilation_cache_test_helper.cc", "MakeNode");

  NodeDef node;
  node.set_name(std::string(name));
  node.set_op(std::string(op));
  for (const auto& input : inputs) node.add_input(input);
  for (const auto& attr : attrs)
    node.mutable_attr()->insert({attr.first, attr.second.proto});
  return node;
}

}  // namespace

GraphDef XlaCompilationCacheSerializeTest::GetTestGraph(
    const PartialTensorShape& input_shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTcc mht_2(mht_2_v, 233, "", "./tensorflow/compiler/jit/tests/xla_compilation_cache_test_helper.cc", "XlaCompilationCacheSerializeTest::GetTestGraph");

  FunctionDef make_test_fn = FunctionDefHelper::Define(
      "TestFn", {"a:float", "b:float", "c:float"}, {"m:float"}, {},
      {{{"d"}, "Add", {"a", "b"}, {{"T", DT_FLOAT}}},
       {{"e"}, "Mul", {"d", "c"}, {{"T", DT_FLOAT}}},
       {{"f"}, "Add", {"e", "a"}, {{"T", DT_FLOAT}}},
       {{"g"}, "Mul", {"f", "b"}, {{"T", DT_FLOAT}}},
       // Force two clusters by excluding this node explicitly.
       {{"h"}, "Add", {"g", "f"}, {{"T", DT_FLOAT}, {"_XlaCompile", false}}},
       {{"i"}, "Add", {"h", "e"}, {{"T", DT_FLOAT}}},
       {{"j"}, "Add", {"i", "h"}, {{"T", DT_FLOAT}}},
       {{"k"}, "Add", {"j", "h"}, {{"T", DT_FLOAT}}},
       {{"l"}, "Add", {"k", "h"}, {{"T", DT_FLOAT}}},
       {{"m"}, "Identity", {"l"}, {{"T", DT_FLOAT}}}});

  GraphDef graph;
  *graph.mutable_library()->add_function() = make_test_fn;
  *graph.add_node() = MakeNode("a", "Placeholder", {},
                               {{"dtype", DT_FLOAT}, {"shape", input_shape}});
  *graph.add_node() = MakeNode("b", "Placeholder", {},
                               {{"dtype", DT_FLOAT}, {"shape", input_shape}});
  *graph.add_node() = MakeNode("c", "Placeholder", {},
                               {{"dtype", DT_FLOAT}, {"shape", input_shape}});
  *graph.add_node() = MakeNode("m", "TestFn", {"a", "b", "c"}, {});
  return graph;
}

Status XlaCompilationCacheSerializeTest::ExecuteWithBatch(const GraphDef& graph,
                                                          int batch) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTcc mht_3(mht_3_v, 264, "", "./tensorflow/compiler/jit/tests/xla_compilation_cache_test_helper.cc", "XlaCompilationCacheSerializeTest::ExecuteWithBatch");

  const TensorShape shape({batch, 4});

  // Compute the golden output tensor
  std::vector<Tensor> golden_output_tensors;
  {
    SessionOptions options;
    std::unique_ptr<Session> session(NewSession(options));
    TF_RETURN_IF_ERROR(session->Create(graph));
    RunOptions run_options;

    Tensor input_a = CreateInputTensor(shape, 0);
    Tensor input_b = CreateInputTensor(shape, shape.num_elements());
    Tensor input_c = CreateInputTensor(shape, 2 * shape.num_elements());
    TF_RETURN_IF_ERROR(session->Run(
        run_options,
        {std::make_pair("a", input_a), std::make_pair("b", input_b),
         std::make_pair("c", input_c)},
        {"m"}, {}, &golden_output_tensors, nullptr));
    TF_RETURN_IF_ERROR(session->Close());
  }

  // Compute the XLA compiled output
  std::vector<Tensor> output_tensors;
  {
    SessionOptions options;
    auto& opts =
        *options.config.mutable_graph_options()->mutable_optimizer_options();
    opts.set_global_jit_level(OptimizerOptions::ON_1);
    opts.set_cpu_global_jit(true);

    std::unique_ptr<Session> session(NewSession(options));
    TF_RETURN_IF_ERROR(session->Create(graph));
    RunOptions run_options;
    Tensor input_a = CreateInputTensor(shape, 0);
    Tensor input_b = CreateInputTensor(shape, shape.num_elements());
    Tensor input_c = CreateInputTensor(shape, 2 * shape.num_elements());
    TF_RETURN_IF_ERROR(session->Run(
        run_options,
        {std::make_pair("a", input_a), std::make_pair("b", input_b),
         std::make_pair("c", input_c)},
        {"m"}, {}, &output_tensors, nullptr));
    TF_RETURN_IF_ERROR(session->Close());
  }

  Tensor f32_input(DT_FLOAT, shape);
  for (int64 i = 0; i < f32_input.NumElements(); ++i) {
    EXPECT_NEAR(golden_output_tensors[0].flat<float>()(i),
                output_tensors[0].flat<float>()(i), 1e-3);
  }
  return Status::OK();
}

Status
XlaCompilationCacheSerializeTest::AlterPersistentCacheEntryHloModuleNames(
    absl::string_view persistent_cache_dir_path,
    absl::string_view file_prefix) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("persistent_cache_dir_path: \"" + std::string(persistent_cache_dir_path.data(), persistent_cache_dir_path.size()) + "\"");
   mht_4_v.push_back("file_prefix: \"" + std::string(file_prefix.data(), file_prefix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSxla_compilation_cache_test_helperDTcc mht_4(mht_4_v, 325, "", "./tensorflow/compiler/jit/tests/xla_compilation_cache_test_helper.cc", "XlaCompilationCacheSerializeTest::AlterPersistentCacheEntryHloModuleNames");

  Env* env = Env::Default();
  std::vector<string> file_names;
  TF_RETURN_IF_ERROR(
      env->GetChildren(tensorflow::testing::TmpDir(), &file_names));

  bool altered = false;
  for (const auto& file_name : file_names) {
    if (absl::EndsWith(file_name, ".pb") &&
        absl::StartsWith(file_name, file_prefix)) {
      XlaSerializedCacheEntry entry;
      auto file_path = io::JoinPath(persistent_cache_dir_path, file_name);
      TF_RETURN_IF_ERROR(ReadTextOrBinaryProto(env, file_path, &entry));
      entry.mutable_hlo_module()->set_name(
          absl::StrCat(entry.hlo_module().name(), "_altered"));
      TF_RETURN_IF_ERROR(WriteBinaryProto(env, file_path, entry));
      altered = true;
    }
  }

  if (!altered) {
    return errors::NotFound(
        "Did not find any persistent XLA compilation cache entries to alter.");
  }
  return Status::OK();
}

}  // namespace tensorflow
