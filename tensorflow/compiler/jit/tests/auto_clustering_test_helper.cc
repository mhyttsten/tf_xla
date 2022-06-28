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
class MHTracer_DTPStensorflowPScompilerPSjitPStestsPSauto_clustering_test_helperDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSauto_clustering_test_helperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPStestsPSauto_clustering_test_helperDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/tests/auto_clustering_test_helper.h"

#include "absl/strings/numbers.h"
#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/util/port.h"
#include "tensorflow/tools/optimization/optimization_pass_runner.h"

namespace tensorflow {
namespace {
StatusOr<string> SummarizeClustering(const GraphDef& auto_clustered_graph_def) {
  testing::ResetClusterSequenceNumber();
  Graph graph(OpRegistry::Global());
  GraphConstructorOptions graph_opts;
  graph_opts.expect_device_spec = true;
  graph_opts.allow_internal_ops = true;
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(graph_opts, auto_clustered_graph_def, &graph));

  // cluster_id -> (operation name -> # of operations)
  const int kNoCluster = -1;
  std::map<int, std::map<string, int>> clusters;
  std::map<int, int> cluster_size;
  int clustered_nodes = 0;
  for (Node* n : graph.op_nodes()) {
    int cluster = kNoCluster;
    if (absl::optional<absl::string_view> maybe_cluster =
            GetXlaClusterForNode(*n)) {
      maybe_cluster->remove_prefix(absl::string_view("cluster_").size());
      TF_RET_CHECK(absl::SimpleAtoi(*maybe_cluster, &cluster));
      clustered_nodes++;
    }
    clusters[cluster][n->type_string()]++;
    cluster_size[cluster]++;
  }

  string result =
      absl::StrCat("Clustered nodes: ", clustered_nodes,
                   "\nUnclustered nodes: ", cluster_size[kNoCluster],
                   "\nNumber of clusters: ", clusters.size() - 1, "\n\n");
  for (const auto& pair : clusters) {
    if (pair.first == kNoCluster) {
      absl::StrAppend(&result, "unclustered");
    } else {
      absl::StrAppend(&result, "cluster ", pair.first);
    }

    absl::StrAppend(&result, " size ", cluster_size[pair.first], "\n");

    for (const auto& ops_and_counts : pair.second) {
      absl::StrAppend(&result, " ", ops_and_counts.first, " ",
                      ops_and_counts.second, "\n");
    }
  }

  return result;
}

Status AssertGraphDefIsUnclustered(const GraphDef& graphdef) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSauto_clustering_test_helperDTcc mht_0(mht_0_v, 253, "", "./tensorflow/compiler/jit/tests/auto_clustering_test_helper.cc", "AssertGraphDefIsUnclustered");

  const char* kXlaClusterAttr = "_XlaCluster";
  const char* kXlaAlreadyClusteredAttr = "_XlaAlreadyClustered";

  for (const NodeDef& node : graphdef.node()) {
    if (node.attr().count(kXlaClusterAttr) ||
        node.attr().count(kXlaAlreadyClusteredAttr)) {
      return errors::InvalidArgument(
          "Input files are already clustered, you probably copied in "
          "mark_for_compilation_<n>.pbtxt when you should have copied in "
          "before_mark_for_compilation_<n>.pbtxt");
    }
  }

  return Status::OK();
}

Status ReadTextProtoFromString(Env* env, const string& data,
                               ::tensorflow::protobuf::Message* proto) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("data: \"" + data + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSauto_clustering_test_helperDTcc mht_1(mht_1_v, 275, "", "./tensorflow/compiler/jit/tests/auto_clustering_test_helper.cc", "ReadTextProtoFromString");

  if (!::tensorflow::protobuf::TextFormat::ParseFromString(data, proto)) {
    return errors::DataLoss("Can't parse input data as text proto");
  }
  return Status::OK();
}
}  // namespace

Status AutoClusteringTest::RunAutoClusteringTestImpl(
    GraphDef graphdef, absl::string_view golden_summary_file_path) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("golden_summary_file_path: \"" + std::string(golden_summary_file_path.data(), golden_summary_file_path.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSauto_clustering_test_helperDTcc mht_2(mht_2_v, 288, "", "./tensorflow/compiler/jit/tests/auto_clustering_test_helper.cc", "AutoClusteringTest::RunAutoClusteringTestImpl");

  if (!IsGoogleCudaEnabled()) {
    // There is some slight change in the clustering decisions under
    // --config=cuda.  I have not looked closely at why that is happening, but
    // most likely some of the partial declustering passes behave differently
    // with --config=cuda because of different HostMemory.  So for now only test
    // the non-CUDA config, under the assumption that regressions with
    // --config=cuda would also be detected as regressions without
    // --config=cuda.

    LOG(INFO) << "Not running "
              << ::testing::UnitTest::GetInstance()->current_test_info()->name()
              << " since test was not built with --config=cuda";
    return Status::OK();
  }

  TF_RETURN_IF_ERROR(AssertGraphDefIsUnclustered(graphdef));

  OptimizationPassRunner runner;
  TF_RETURN_IF_ERROR(runner.SetJitLevel(tensorflow::OptimizerOptions::ON_2));
  TF_RETURN_IF_ERROR(runner.AddCpus(32));
  TF_RETURN_IF_ERROR(runner.AddGpus(8));

  for (absl::string_view auto_clustering_pass :
       {"CloneConstantsForBetterClusteringPass", "MarkForCompilationPass",
        "IncreaseDynamismForAutoJitPass", "PartiallyDeclusterPass"}) {
    GraphDef next;
    TF_RETURN_IF_ERROR(
        runner.Run(auto_clustering_pass, std::move(graphdef), &next));
    graphdef = std::move(next);
  }

  TF_ASSIGN_OR_RETURN(string clustering_summary, SummarizeClustering(graphdef));

  // To update golden files flip this to true and run
  //
  // bazel test --test_strategy=local \
  //   tensorflow/compiler/jit/tests:auto_clustering_test
  bool update_golden = false;
  if (update_golden) {
    TF_RETURN_IF_ERROR(WriteStringToFile(
        Env::Default(), string(golden_summary_file_path), clustering_summary));
  }

  string golden_file_contents;
  TF_RETURN_IF_ERROR(ReadFileToString(
      Env::Default(), string(golden_summary_file_path), &golden_file_contents));

  EXPECT_EQ(golden_file_contents, clustering_summary);

  return Status::OK();
}

Status AutoClusteringTest::RunAutoClusteringTestWithPbtxt(
    absl::string_view pbtxt_file_path,
    absl::string_view golden_summary_file_path) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("pbtxt_file_path: \"" + std::string(pbtxt_file_path.data(), pbtxt_file_path.size()) + "\"");
   mht_3_v.push_back("golden_summary_file_path: \"" + std::string(golden_summary_file_path.data(), golden_summary_file_path.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSauto_clustering_test_helperDTcc mht_3(mht_3_v, 348, "", "./tensorflow/compiler/jit/tests/auto_clustering_test_helper.cc", "AutoClusteringTest::RunAutoClusteringTestWithPbtxt");

  GraphDef graphdef;
  TF_RETURN_IF_ERROR(
      ReadTextProto(Env::Default(), string(pbtxt_file_path), &graphdef));
  return RunAutoClusteringTestImpl(std::move(graphdef),
                                   golden_summary_file_path);
}

Status AutoClusteringTest::RunAutoClusteringTestWithGzippedPbtxt(
    absl::string_view gzipped_pbtxt_file_path,
    absl::string_view golden_summary_file_path) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("gzipped_pbtxt_file_path: \"" + std::string(gzipped_pbtxt_file_path.data(), gzipped_pbtxt_file_path.size()) + "\"");
   mht_4_v.push_back("golden_summary_file_path: \"" + std::string(golden_summary_file_path.data(), golden_summary_file_path.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPStestsPSauto_clustering_test_helperDTcc mht_4(mht_4_v, 363, "", "./tensorflow/compiler/jit/tests/auto_clustering_test_helper.cc", "AutoClusteringTest::RunAutoClusteringTestWithGzippedPbtxt");

  Env* env = Env::Default();
  std::unique_ptr<RandomAccessFile> file_reader;
  TF_RETURN_IF_ERROR(
      env->NewRandomAccessFile(string(gzipped_pbtxt_file_path), &file_reader));
  std::unique_ptr<io::RandomAccessInputStream> input_stream(
      new io::RandomAccessInputStream(file_reader.get()));
  constexpr int k_buffer_size = 256 << 10;  // 256kb
  io::ZlibInputStream in(input_stream.get(),
                         /*input_buffer_bytes=*/k_buffer_size,
                         /*output_buffer_bytes=*/k_buffer_size,
                         io::ZlibCompressionOptions::GZIP());
  tstring decompressed_pbtxt_string;
  Status s = in.ReadNBytes(INT_MAX, &decompressed_pbtxt_string);
  if (!s.ok() && !errors::IsOutOfRange(s)) {
    // OutOfRange is fine since we set the number of read bytes to INT_MAX.
    // Only return other kinds of errors.
    return s;
  }

  GraphDef graphdef;
  TF_RETURN_IF_ERROR(ReadTextProtoFromString(
      Env::Default(), decompressed_pbtxt_string, &graphdef));
  return RunAutoClusteringTestImpl(std::move(graphdef),
                                   golden_summary_file_path);
}

#if defined(PLATFORM_GOOGLE)
Status BenchmarkMarkForCompilation(absl::string_view graph_def_path,
                                   benchmark::State& state) {
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(
      ReadTextProto(Env::Default(), string(graph_def_path), &graph_def));

  OptimizationPassRunner runner;
  TF_RETURN_IF_ERROR(runner.SetJitLevel(tensorflow::OptimizerOptions::ON_2));
  TF_RETURN_IF_ERROR(runner.AddCpus(32));
  TF_RETURN_IF_ERROR(runner.AddGpus(8));

  for (auto _ : state) {
    state.PauseTiming();
    GraphDef result;
    GraphDef graph_def_copy = graph_def;
    state.ResumeTiming();
    TF_RETURN_IF_ERROR(runner.Run("MarkForCompilationPass",
                                  std::move(graph_def_copy), &result));
  }

  return Status::OK();
}
#endif  // PLATFORM_GOOGLE

}  // namespace tensorflow
