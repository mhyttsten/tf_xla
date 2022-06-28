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
class MHTracer_DTPStensorflowPSexamplesPSlabel_imagePSmainDTcc {
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
   MHTracer_DTPStensorflowPSexamplesPSlabel_imagePSmainDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSexamplesPSlabel_imagePSmainDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// A minimal but useful C++ example showing how to load an Imagenet-style object
// recognition TensorFlow model, prepare input images for it, run them through
// the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. In particular, using
// auto for the types of a lot of the returned values from TensorFlow calls can
// remove a lot of boilerplate, but I find the explicit types useful in sample
// code to make it simple to look up the classes involved.
//
// To use it, compile and then run in a working directory with the
// learning/brain/tutorials/label_image/data/ folder below it, and you should
// see the top five labels for the example Lena image output. You can then
// customize it to use your own models or images by changing the file names at
// the top of the main() function.
//
// The googlenet_graph.pb file included by default is created from Inception.
//
// Note that, for GIF inputs, to reuse existing code, only single-frame ones
// are supported.

#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::tstring;

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status ReadLabelsFile(const string& file_name, std::vector<string>* result,
                      size_t* found_label_count) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPSexamplesPSlabel_imagePSmainDTcc mht_0(mht_0_v, 243, "", "./tensorflow/examples/label_image/main.cc", "ReadLabelsFile");

  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return Status::OK();
}

static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSexamplesPSlabel_imagePSmainDTcc mht_1(mht_1_v, 267, "", "./tensorflow/examples/label_image/main.cc", "ReadEntireFile");

  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<tstring>()() = tstring(data);
  return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPSexamplesPSlabel_imagePSmainDTcc mht_2(mht_2_v, 297, "", "./tensorflow/examples/label_image/main.cc", "ReadTensorFromImageFile");

  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";

  // read file_name into a tensor named input
  Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  TF_RETURN_IF_ERROR(
      ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

  // use a placeholder to read input data
  auto file_reader =
      Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {"input", input},
  };

  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  if (tensorflow::str_util::EndsWith(file_name, ".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
    // gif decoder returns 4-D tensor, remove the first dim
    image_reader =
        Squeeze(root.WithOpName("squeeze_first_dim"),
                DecodeGif(root.WithOpName("gif_reader"), file_reader));
  } else if (tensorflow::str_util::EndsWith(file_name, ".bmp")) {
    image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }
  // Now cast the image data to float so we can do normal math on it.
  auto float_caster =
      Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);
  // Bilinearly resize the image to fit the required dimensions.
  auto resized = ResizeBilinear(
      root, dims_expander,
      Const(root.WithOpName("size"), {input_height, input_width}));
  // Subtract the mean and divide by the scale.
  Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
      {input_std});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));
  return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("graph_file_name: \"" + graph_file_name + "\"");
   MHTracer_DTPStensorflowPSexamplesPSlabel_imagePSmainDTcc mht_3(mht_3_v, 370, "", "./tensorflow/examples/label_image/main.cc", "LoadGraph");

  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                    Tensor* indices, Tensor* scores) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSexamplesPSlabel_imagePSmainDTcc mht_4(mht_4_v, 392, "", "./tensorflow/examples/label_image/main.cc", "GetTopLabels");

  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string output_name = "top_k";
  TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  // The TopK node returns two outputs, the scores and their original indices,
  // so we have to append :0 and :1 to specify them both.
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
}

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status PrintTopLabels(const std::vector<Tensor>& outputs,
                      const string& labels_file_name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("labels_file_name: \"" + labels_file_name + "\"");
   MHTracer_DTPStensorflowPSexamplesPSlabel_imagePSmainDTcc mht_5(mht_5_v, 423, "", "./tensorflow/examples/label_image/main.cc", "PrintTopLabels");

  std::vector<string> labels;
  size_t label_count;
  Status read_labels_status =
      ReadLabelsFile(labels_file_name, &labels, &label_count);
  if (!read_labels_status.ok()) {
    LOG(ERROR) << read_labels_status;
    return read_labels_status;
  }
  const int how_many_labels = std::min(5, static_cast<int>(label_count));
  Tensor indices;
  Tensor scores;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  for (int pos = 0; pos < how_many_labels; ++pos) {
    const int label_index = indices_flat(pos);
    const float score = scores_flat(pos);
    LOG(INFO) << labels[label_index] << " (" << label_index << "): " << score;
  }
  return Status::OK();
}

// This is a testing function that returns whether the top label index is the
// one that's expected.
Status CheckTopLabel(const std::vector<Tensor>& outputs, int expected,
                     bool* is_expected) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSexamplesPSlabel_imagePSmainDTcc mht_6(mht_6_v, 452, "", "./tensorflow/examples/label_image/main.cc", "CheckTopLabel");

  *is_expected = false;
  Tensor indices;
  Tensor scores;
  const int how_many_labels = 1;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  if (indices_flat(0) != expected) {
    LOG(ERROR) << "Expected label #" << expected << " but got #"
               << indices_flat(0);
    *is_expected = false;
  } else {
    *is_expected = true;
  }
  return Status::OK();
}

int main(int argc, char* argv[]) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSexamplesPSlabel_imagePSmainDTcc mht_7(mht_7_v, 472, "", "./tensorflow/examples/label_image/main.cc", "main");

  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than inception_v3, then you'll need to update these.
  string image = "tensorflow/examples/label_image/data/grace_hopper.jpg";
  string graph =
      "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb";
  string labels =
      "tensorflow/examples/label_image/data/imagenet_slim_labels.txt";
  int32_t input_width = 299;
  int32_t input_height = 299;
  float input_mean = 0;
  float input_std = 255;
  string input_layer = "input";
  string output_layer = "InceptionV3/Predictions/Reshape_1";
  bool self_test = false;
  string root_dir = "";
  std::vector<Flag> flag_list = {
      Flag("image", &image, "image to be processed"),
      Flag("graph", &graph, "graph to be executed"),
      Flag("labels", &labels, "name of file containing labels"),
      Flag("input_width", &input_width, "resize image to this width in pixels"),
      Flag("input_height", &input_height,
           "resize image to this height in pixels"),
      Flag("input_mean", &input_mean, "scale pixel values to this mean"),
      Flag("input_std", &input_std, "scale pixel values to this std deviation"),
      Flag("input_layer", &input_layer, "name of input layer"),
      Flag("output_layer", &output_layer, "name of output layer"),
      Flag("self_test", &self_test, "run a self test"),
      Flag("root_dir", &root_dir,
           "interpret image and graph file names relative to this directory"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  std::vector<Tensor> resized_tensors;
  string image_path = tensorflow::io::JoinPath(root_dir, image);
  Status read_tensor_status =
      ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
                              input_std, &resized_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }
  const Tensor& resized_tensor = resized_tensors[0];

  // Actually run the image through the model.
  std::vector<Tensor> outputs;
  Status run_status = session->Run({{input_layer, resized_tensor}},
                                   {output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  // This is for automated testing to make sure we get the expected result with
  // the default settings. We know that label 653 (military uniform) should be
  // the top label for the Admiral Hopper image.
  if (self_test) {
    bool expected_matches;
    Status check_status = CheckTopLabel(outputs, 653, &expected_matches);
    if (!check_status.ok()) {
      LOG(ERROR) << "Running check failed: " << check_status;
      return -1;
    }
    if (!expected_matches) {
      LOG(ERROR) << "Self-test failed!";
      return -1;
    }
  }

  // Do something interesting with the results we've generated.
  Status print_status = PrintTopLabels(outputs, labels);
  if (!print_status.ok()) {
    LOG(ERROR) << "Running print failed: " << print_status;
    return -1;
  }

  return 0;
}
