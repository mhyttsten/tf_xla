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
class MHTracer_DTPStensorflowPScPSpython_apiDTcc {
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
   MHTracer_DTPStensorflowPScPSpython_apiDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSpython_apiDTcc() {
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

#include "tensorflow/c/python_api.h"

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/python/framework/cpp_shape_inference.pb.h"

namespace tensorflow {

void AddControlInput(TF_Graph* graph, TF_Operation* op, TF_Operation* input) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSpython_apiDTcc mht_0(mht_0_v, 193, "", "./tensorflow/c/python_api.cc", "AddControlInput");

  mutex_lock l(graph->mu);
  graph->graph.AddControlEdge(&input->node, &op->node);
  RecordMutation(graph, *op, "adding control input");
}

void SetAttr(TF_Graph* graph, TF_Operation* op, const char* attr_name,
             TF_Buffer* attr_value_proto, TF_Status* status) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSpython_apiDTcc mht_1(mht_1_v, 204, "", "./tensorflow/c/python_api.cc", "SetAttr");

  AttrValue attr_val;
  if (!attr_val.ParseFromArray(attr_value_proto->data,
                               attr_value_proto->length)) {
    status->status =
        tensorflow::errors::InvalidArgument("Invalid AttrValue proto");
    return;
  }

  mutex_lock l(graph->mu);
  op->node.AddAttr(attr_name, attr_val);
  RecordMutation(graph, *op, "setting attribute");
}

void ClearAttr(TF_Graph* graph, TF_Operation* op, const char* attr_name,
               TF_Status* status) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSpython_apiDTcc mht_2(mht_2_v, 223, "", "./tensorflow/c/python_api.cc", "ClearAttr");

  mutex_lock l(graph->mu);
  op->node.ClearAttr(attr_name);
  RecordMutation(graph, *op, "clearing attribute");
}

void SetFullType(TF_Graph* graph, TF_Operation* op,
                 const FullTypeDef& full_type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSpython_apiDTcc mht_3(mht_3_v, 233, "", "./tensorflow/c/python_api.cc", "SetFullType");

  mutex_lock l(graph->mu);
  *op->node.mutable_def()->mutable_experimental_type() = full_type;
  RecordMutation(graph, *op, "setting fulltype");
}

void SetRequestedDevice(TF_Graph* graph, TF_Operation* op, const char* device) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("device: \"" + (device == nullptr ? std::string("nullptr") : std::string((char*)device)) + "\"");
   MHTracer_DTPStensorflowPScPSpython_apiDTcc mht_4(mht_4_v, 243, "", "./tensorflow/c/python_api.cc", "SetRequestedDevice");

  mutex_lock l(graph->mu);
  op->node.set_requested_device(device);
  RecordMutation(graph, *op, "setting device");
}

void UpdateEdge(TF_Graph* graph, TF_Output new_src, TF_Input dst,
                TF_Status* status) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSpython_apiDTcc mht_5(mht_5_v, 253, "", "./tensorflow/c/python_api.cc", "UpdateEdge");

  TF_UpdateEdge(graph, new_src, dst, status);
}

void RemoveAllControlInputs(TF_Graph* graph, TF_Operation* op) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSpython_apiDTcc mht_6(mht_6_v, 260, "", "./tensorflow/c/python_api.cc", "RemoveAllControlInputs");

  mutex_lock l(graph->mu);
  std::vector<const Edge*> control_edges;
  for (const Edge* edge : op->node.in_edges()) {
    if (!edge->IsControlEdge()) continue;
    control_edges.push_back(edge);
  }
  for (const Edge* edge : control_edges) {
    graph->graph.RemoveControlEdge(edge);
  }
}

void SetRequireShapeInferenceFns(TF_Graph* graph, bool require) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSpython_apiDTcc mht_7(mht_7_v, 275, "", "./tensorflow/c/python_api.cc", "SetRequireShapeInferenceFns");

  mutex_lock l(graph->mu);
  graph->refiner.set_require_shape_inference_fns(require);
}

void ExtendSession(TF_Session* session, TF_Status* status) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSpython_apiDTcc mht_8(mht_8_v, 283, "", "./tensorflow/c/python_api.cc", "ExtendSession");

  ExtendSessionGraphHelper(session, status);
  session->extend_before_run = false;
}

std::string GetHandleShapeAndType(TF_Graph* graph, TF_Output output) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSpython_apiDTcc mht_9(mht_9_v, 291, "", "./tensorflow/c/python_api.cc", "GetHandleShapeAndType");

  Node* node = &output.oper->node;
  CppShapeInferenceResult::HandleData handle_data;
  handle_data.set_is_set(true);
  {
    mutex_lock l(graph->mu);
    tensorflow::shape_inference::InferenceContext* ic =
        graph->refiner.GetContext(node);
    CHECK(ic != nullptr);
    CHECK_LT(output.index, ic->num_outputs());
    const auto* shapes_and_types =
        ic->output_handle_shapes_and_types(output.index);
    if (shapes_and_types == nullptr) return "";

    for (const auto& p : *shapes_and_types) {
      auto* out_shape_and_type = handle_data.add_shape_and_type();
      ic->ShapeHandleToProto(p.shape, out_shape_and_type->mutable_shape());
      out_shape_and_type->set_dtype(p.dtype);
      *out_shape_and_type->mutable_type() = p.type;
    }
  }
  string result;
  handle_data.SerializeToString(&result);
  return result;
}

void SetHandleShapeAndType(TF_Graph* graph, TF_Output output, const void* proto,
                           size_t proto_len, TF_Status* status) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSpython_apiDTcc mht_10(mht_10_v, 321, "", "./tensorflow/c/python_api.cc", "SetHandleShapeAndType");

  tensorflow::CppShapeInferenceResult::HandleData handle_data;
  if (!handle_data.ParseFromArray(proto, proto_len)) {
    status->status = tensorflow::errors::InvalidArgument(
        "Couldn't deserialize HandleData proto");
    return;
  }
  DCHECK(handle_data.is_set());

  tensorflow::mutex_lock l(graph->mu);
  tensorflow::shape_inference::InferenceContext* ic =
      graph->refiner.GetContext(&output.oper->node);

  std::vector<tensorflow::shape_inference::ShapeAndType> shapes_and_types;
  for (const auto& shape_and_type_proto : handle_data.shape_and_type()) {
    tensorflow::shape_inference::ShapeHandle shape;
    status->status =
        ic->MakeShapeFromShapeProto(shape_and_type_proto.shape(), &shape);
    if (TF_GetCode(status) != TF_OK) return;
    shapes_and_types.emplace_back(shape, shape_and_type_proto.dtype(),
                                  shape_and_type_proto.type());
  }
  ic->set_output_handle_shapes_and_types(output.index, shapes_and_types);
}

void AddWhileInputHack(TF_Graph* graph, TF_Output new_src, TF_Operation* dst,
                       TF_Status* status) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSpython_apiDTcc mht_11(mht_11_v, 350, "", "./tensorflow/c/python_api.cc", "AddWhileInputHack");

  mutex_lock l(graph->mu);
  status->status = graph->graph.AddWhileInputHack(&new_src.oper->node,
                                                  new_src.index, &dst->node);
  if (TF_GetCode(status) == TF_OK) {
    // This modification only updates the destination node for
    // the purposes of running this graph in a session. Thus, we don't
    // record the source node as being modified.
    RecordMutation(graph, *dst, "adding input tensor");
  }
}

}  // namespace tensorflow
