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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc() {
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

#include "tensorflow/core/grappler/optimizers/data/graph_test_utils.h"

#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace grappler {
namespace graph_tests_utils {

NodeDef MakeBatchV2Node(StringPiece name, StringPiece input_node_name,
                        StringPiece batch_size_node_name,
                        StringPiece drop_remainder_node_name,
                        bool parallel_copy) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/grappler/optimizers/data/graph_test_utils.cc", "MakeBatchV2Node");

  return test::function::NDef(
      name, "BatchDatasetV2",
      {string(input_node_name), string(batch_size_node_name),
       string(drop_remainder_node_name)},
      {{"parallel_copy", parallel_copy},
       {"output_shapes", gtl::ArraySlice<TensorShape>{}},
       {"output_types", gtl::ArraySlice<DataType>{}}});
}

NodeDef MakeParallelBatchNode(StringPiece name, StringPiece input_node_name,
                              StringPiece batch_size_node_name,
                              StringPiece num_parallel_calls_node_name,
                              StringPiece drop_remainder_node_name,
                              StringPiece deterministic) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/grappler/optimizers/data/graph_test_utils.cc", "MakeParallelBatchNode");

  return test::function::NDef(
      name, "ParallelBatchDataset",
      {string(input_node_name), string(batch_size_node_name),
       string(num_parallel_calls_node_name), string(drop_remainder_node_name)},
      {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
       {"output_types", gtl::ArraySlice<DataType>{}},
       {"deterministic", string(deterministic)}});
}

NodeDef MakeCacheV2Node(StringPiece name, StringPiece input_node_name,
                        StringPiece filename_node_name,
                        StringPiece cache_node_name) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc mht_2(mht_2_v, 232, "", "./tensorflow/core/grappler/optimizers/data/graph_test_utils.cc", "MakeCacheV2Node");

  return test::function::NDef(
      name, "CacheDatasetV2",
      {
          string(input_node_name),
          string(filename_node_name),
          string(cache_node_name),
      },
      {
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
      });
}

NodeDef MakeFilterNode(StringPiece name, StringPiece input_node_name,
                       StringPiece function_name) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc mht_3(mht_3_v, 250, "", "./tensorflow/core/grappler/optimizers/data/graph_test_utils.cc", "MakeFilterNode");

  return test::function::NDef(
      name, "FilterDataset", {string(input_node_name)},
      {{"predicate", FunctionDefHelper::FunctionRef(string(function_name))},
       {"Targuments", {}},
       {"output_shapes", gtl::ArraySlice<TensorShape>{}},
       {"output_types", gtl::ArraySlice<DataType>{}}});
}

NodeDef MakeMapAndBatchNode(StringPiece name, StringPiece input_node_name,
                            StringPiece batch_size_node_name,
                            StringPiece num_parallel_calls_node_name,
                            StringPiece drop_remainder_node_name,
                            StringPiece function_name) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc mht_4(mht_4_v, 266, "", "./tensorflow/core/grappler/optimizers/data/graph_test_utils.cc", "MakeMapAndBatchNode");

  return test::function::NDef(
      name, "MapAndBatchDataset",
      {string(input_node_name), string(batch_size_node_name),
       string(num_parallel_calls_node_name), string(drop_remainder_node_name)},
      {{"f", FunctionDefHelper::FunctionRef(string(function_name))},
       {"Targuments", {}},
       {"output_shapes", gtl::ArraySlice<TensorShape>{}},
       {"output_types", gtl::ArraySlice<DataType>{}}});
}

NodeDef MakeMapNode(StringPiece name, StringPiece input_node_name,
                    StringPiece function_name) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc mht_5(mht_5_v, 281, "", "./tensorflow/core/grappler/optimizers/data/graph_test_utils.cc", "MakeMapNode");

  return test::function::NDef(
      name, "MapDataset", {string(input_node_name)},
      {{"f", FunctionDefHelper::FunctionRef(string(function_name))},
       {"Targuments", {}},
       {"output_shapes", gtl::ArraySlice<TensorShape>{}},
       {"output_types", gtl::ArraySlice<DataType>{}}});
}

NodeDef MakeParallelInterleaveV2Node(StringPiece name,
                                     StringPiece input_node_name,
                                     StringPiece cycle_length_node_name,
                                     StringPiece block_length_node_name,
                                     StringPiece num_parallel_calls_node_name,
                                     StringPiece function_name, bool sloppy) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc mht_6(mht_6_v, 298, "", "./tensorflow/core/grappler/optimizers/data/graph_test_utils.cc", "MakeParallelInterleaveV2Node");

  return test::function::NDef(
      name, "ParallelInterleaveDatasetV2",
      {string(input_node_name), string(cycle_length_node_name),
       string(block_length_node_name), string(num_parallel_calls_node_name)},
      {
          {"f", FunctionDefHelper::FunctionRef(string(function_name))},
          {"Targuments", {}},
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
          {"sloppy", sloppy},
      });
}

NodeDef MakeParallelInterleaveV4Node(StringPiece name,
                                     StringPiece input_node_name,
                                     StringPiece cycle_length_node_name,
                                     StringPiece block_length_node_name,
                                     StringPiece num_parallel_calls_node_name,
                                     StringPiece function_name,
                                     StringPiece deterministic) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc mht_7(mht_7_v, 321, "", "./tensorflow/core/grappler/optimizers/data/graph_test_utils.cc", "MakeParallelInterleaveV4Node");

  return test::function::NDef(
      name, "ParallelInterleaveDatasetV4",
      {string(input_node_name), string(cycle_length_node_name),
       string(block_length_node_name), string(num_parallel_calls_node_name)},
      {
          {"f", FunctionDefHelper::FunctionRef(string(function_name))},
          {"Targuments", {}},
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
          {"deterministic", string(deterministic)},
      });
}

NodeDef MakeParallelMapNode(StringPiece name, StringPiece input_node_name,
                            StringPiece num_parallel_calls_node_name,
                            StringPiece function_name, bool sloppy) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc mht_8(mht_8_v, 340, "", "./tensorflow/core/grappler/optimizers/data/graph_test_utils.cc", "MakeParallelMapNode");

  return test::function::NDef(
      name, "ParallelMapDataset",
      {string(input_node_name), string(num_parallel_calls_node_name)},
      {
          {"f", FunctionDefHelper::FunctionRef(string(function_name))},
          {"Targuments", {}},
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
          {"sloppy", sloppy},
      });
}

NodeDef MakeParallelMapV2Node(StringPiece name, StringPiece input_node_name,
                              StringPiece num_parallel_calls_node_name,
                              StringPiece function_name,
                              StringPiece deterministic) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc mht_9(mht_9_v, 359, "", "./tensorflow/core/grappler/optimizers/data/graph_test_utils.cc", "MakeParallelMapV2Node");

  return test::function::NDef(
      name, "ParallelMapDatasetV2",
      {string(input_node_name), string(num_parallel_calls_node_name)},
      {
          {"f", FunctionDefHelper::FunctionRef(string(function_name))},
          {"Targuments", {}},
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
          {"deterministic", string(deterministic)},
      });
}

NodeDef MakeParseExampleNode(StringPiece name, StringPiece input_node_name,
                             StringPiece num_parallel_calls_node_name,
                             bool sloppy) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc mht_10(mht_10_v, 377, "", "./tensorflow/core/grappler/optimizers/data/graph_test_utils.cc", "MakeParseExampleNode");

  return test::function::NDef(
      name, "ParseExampleDataset",
      {string(input_node_name), string(num_parallel_calls_node_name)},
      {
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
          {"sloppy", sloppy},
      });
}

NodeDef MakeShuffleV2Node(StringPiece name, StringPiece input_node_name,
                          StringPiece buffer_size_node_name,
                          StringPiece seed_generator_node_name) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc mht_11(mht_11_v, 393, "", "./tensorflow/core/grappler/optimizers/data/graph_test_utils.cc", "MakeShuffleV2Node");

  return test::function::NDef(
      name, "ShuffleDatasetV2",
      {
          string(input_node_name),
          string(buffer_size_node_name),
          string(seed_generator_node_name),
      },
      {
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
      });
}

NodeDef MakeTakeNode(StringPiece name, StringPiece input_node_name,
                     StringPiece count_node_name) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc mht_12(mht_12_v, 411, "", "./tensorflow/core/grappler/optimizers/data/graph_test_utils.cc", "MakeTakeNode");

  return test::function::NDef(
      name, "TakeDataset",
      {
          string(input_node_name),
          string(count_node_name),
      },
      {
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
      });
}

NodeDef MakeSkipNode(StringPiece name, StringPiece input_node_name,
                     StringPiece count_node_name) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc mht_13(mht_13_v, 428, "", "./tensorflow/core/grappler/optimizers/data/graph_test_utils.cc", "MakeSkipNode");

  return test::function::NDef(
      name, "SkipDataset",
      {
          string(input_node_name),
          string(count_node_name),
      },
      {
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
      });
}

NodeDef MakeShardNode(StringPiece name, StringPiece input_node_name,
                      StringPiece num_shards_node_name,
                      StringPiece index_node_name) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc mht_14(mht_14_v, 446, "", "./tensorflow/core/grappler/optimizers/data/graph_test_utils.cc", "MakeShardNode");

  return test::function::NDef(
      name, "ShardDataset",
      {
          string(input_node_name),
          string(num_shards_node_name),
          string(index_node_name),
      },
      {
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
      });
}

NodeDef MakePrefetchNode(StringPiece name, StringPiece input_node_name,
                         StringPiece buffer_size) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSgraph_test_utilsDTcc mht_15(mht_15_v, 464, "", "./tensorflow/core/grappler/optimizers/data/graph_test_utils.cc", "MakePrefetchNode");

  return test::function::NDef(
      name, "PrefetchDataset", {string(input_node_name), string(buffer_size)},
      {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
       {"output_types", gtl::ArraySlice<DataType>{}},
       {"slack_period", 0},
       {"legacy_autotune", true},
       {"buffer_size_min", 0}});
}

}  // namespace graph_tests_utils
}  // namespace grappler
}  // namespace tensorflow
