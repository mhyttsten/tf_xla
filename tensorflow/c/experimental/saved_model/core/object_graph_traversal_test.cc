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
class MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSobject_graph_traversal_testDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSobject_graph_traversal_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSobject_graph_traversal_testDTcc() {
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

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/c/experimental/saved_model/core/saved_model_utils.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"

namespace tensorflow {
namespace {

SavedObjectGraph ParseSavedObjectGraph(StringPiece text_proto) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPScorePSobject_graph_traversal_testDTcc mht_0(mht_0_v, 196, "", "./tensorflow/c/experimental/saved_model/core/object_graph_traversal_test.cc", "ParseSavedObjectGraph");

  SavedObjectGraph value;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(string(text_proto),
                                                          &value));
  return value;
}

constexpr absl::string_view kSingleChildFoo = R"(
nodes {
  children {
    node_id: 1
    local_name: "foo"
  }
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
)";

constexpr absl::string_view kSingleChildFooWithFuncBar = R"(
nodes {
  children {
    node_id: 1
    local_name: "foo"
  }
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  children {
    node_id: 2
    local_name: "bar"
  }
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  function {
    concrete_functions: "__inference_my_func_5"
    function_spec {
      fullargspec {
        named_tuple_value {
          name: "FullArgSpec"
          values {
            key: "args"
            value {
              list_value {
              }
            }
          }
          values {
            key: "varargs"
            value {
              none_value {
              }
            }
          }
          values {
            key: "varkw"
            value {
              none_value {
              }
            }
          }
          values {
            key: "defaults"
            value {
              none_value {
              }
            }
          }
          values {
            key: "kwonlyargs"
            value {
              list_value {
              }
            }
          }
          values {
            key: "kwonlydefaults"
            value {
              none_value {
              }
            }
          }
          values {
            key: "annotations"
            value {
              dict_value {
              }
            }
          }
        }
      }
      input_signature {
        tuple_value {
        }
      }
    }
  }
}
concrete_functions {
  key: "__inference_my_func_5"
  value {
    canonicalized_input_signature {
      tuple_value {
        values {
          tuple_value {
          }
        }
        values {
          dict_value {
          }
        }
      }
    }
    output_signature {
      tensor_spec_value {
        shape {
        }
        dtype: DT_FLOAT
      }
    }
  }
}
)";

// In this graph, foo.baz and bar.wombat should point to the same object.
constexpr absl::string_view kMultiplePathsToChild = R"(
nodes {
  children {
    node_id: 1
    local_name: "foo"
  }
  children {
    node_id: 2
    local_name: "bar"
  }
  children {
    node_id: 3
    local_name: "signatures"
  }
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  children {
    node_id: 4
    local_name: "baz"
  }
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  children {
    node_id: 4
    local_name: "wombat"
  }
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  user_object {
    identifier: "signature_map"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
)";

// `foo` has edge `bar`, which has edge `parent` pointing back to `foo`.
constexpr absl::string_view kCycleBetweenParentAndChild = R"(
nodes {
  children {
    node_id: 1
    local_name: "foo"
  }
  children {
    node_id: 2
    local_name: "signatures"
  }
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  children {
    node_id: 3
    local_name: "bar"
  }
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  user_object {
    identifier: "signature_map"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
nodes {
  children {
    node_id: 1
    local_name: "parent"
  }
  user_object {
    identifier: "_generic_user_object"
    version {
      producer: 1
      min_consumer: 1
    }
  }
}
)";

TEST(ObjectGraphTraversalTest, Success) {
  SavedObjectGraph object_graph = ParseSavedObjectGraph(kSingleChildFoo);
  absl::optional<int> node = internal::FindNodeAtPath("foo", object_graph);
  ASSERT_TRUE(node.has_value());
  EXPECT_EQ(*node, 1);
}

TEST(ObjectGraphTraversalTest, ObjectNotFound) {
  SavedObjectGraph object_graph = ParseSavedObjectGraph(kSingleChildFoo);
  absl::optional<int> node = internal::FindNodeAtPath("bar", object_graph);
  EXPECT_FALSE(node.has_value());
}

TEST(ObjectGraphTraversalTest, CaseSensitiveMismatch) {
  SavedObjectGraph object_graph = ParseSavedObjectGraph(kSingleChildFoo);
  absl::optional<int> node = internal::FindNodeAtPath("FOO", object_graph);
  EXPECT_FALSE(node.has_value());
}

TEST(ObjectGraphTraversalTest, NestedObjectFound) {
  SavedObjectGraph object_graph =
      ParseSavedObjectGraph(kSingleChildFooWithFuncBar);
  absl::optional<int> node = internal::FindNodeAtPath("foo.bar", object_graph);
  ASSERT_TRUE(node.has_value());
  EXPECT_EQ(*node, 2);
}

TEST(ObjectGraphTraversalTest, MultiplePathsAliasSameObject) {
  SavedObjectGraph object_graph = ParseSavedObjectGraph(kMultiplePathsToChild);
  absl::optional<int> foo_baz_node =
      internal::FindNodeAtPath("foo.baz", object_graph);
  ASSERT_TRUE(foo_baz_node.has_value());
  EXPECT_EQ(*foo_baz_node, 4);

  absl::optional<int> bar_wombat_node =
      internal::FindNodeAtPath("bar.wombat", object_graph);
  ASSERT_TRUE(bar_wombat_node.has_value());
  EXPECT_EQ(*bar_wombat_node, 4);

  EXPECT_EQ(*foo_baz_node, *bar_wombat_node);
}

TEST(ObjectGraphTraversalTest, CyclesAreOK) {
  SavedObjectGraph object_graph =
      ParseSavedObjectGraph(kCycleBetweenParentAndChild);
  absl::optional<int> foo = internal::FindNodeAtPath("foo", object_graph);
  ASSERT_TRUE(foo.has_value());
  EXPECT_EQ(*foo, 1);

  absl::optional<int> foo_bar =
      internal::FindNodeAtPath("foo.bar", object_graph);
  ASSERT_TRUE(foo_bar.has_value());
  EXPECT_EQ(*foo_bar, 3);

  absl::optional<int> foo_bar_parent =
      internal::FindNodeAtPath("foo.bar.parent", object_graph);
  ASSERT_TRUE(foo_bar_parent.has_value());
  EXPECT_EQ(*foo_bar_parent, 1);

  absl::optional<int> foo_bar_parent_bar =
      internal::FindNodeAtPath("foo.bar.parent.bar", object_graph);
  ASSERT_TRUE(foo_bar_parent_bar.has_value());
  EXPECT_EQ(*foo_bar_parent_bar, 3);

  EXPECT_EQ(*foo, *foo_bar_parent);
  EXPECT_EQ(*foo_bar, *foo_bar_parent_bar);
}

}  // namespace
}  // namespace tensorflow
