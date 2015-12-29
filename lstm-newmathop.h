/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#ifndef EXAMPLES_LSTM_LSTM_H_
#define EXAMPLES_LSTM_LSTM_H_

#include <string>
#include <vector>
#include "singa/singa.h"
#include "./lstm.pb.h"

//namespace lstm{
  namespace singa{
using std::vector;
using singa::LayerProto;
using singa::Layer;
using singa::Param;
using singa::Blob;
using singa::Metric;

class LSTMLayer : virtual public singa::Layer
{
 public:

  inline int max_lenght() { return max_length_; }

 protected:

  int max_length_;
};

/************************************************************
Get max length (timesteps) of all instances in one mini-batch
Add PAD index, -2,  at the end for shorter instances.
add start index, -1 , end index </s> remain the 0
***************************************************************/
class DataLayer : public LSTMLayer, public singa::InputLayer
{
 public:
  ~DataLayer();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  int max_window() const {
    return max_window_;
  }
  int batch_size() const{return batch_size_;}

 private:
  int max_window_;
  int batch_size_;
  singa::io::Store* store_ = nullptr;
};



/********************************************************************************
Return an empty Blob if the input is b * 1 PAD index, b for batchsize
Get input data from src_layer[k], k is unroll id if there are > 1; k =0 otherwise.
Return -1 in ComputeFeature if all indexes are END indexes.
***********************************************************************************/
class EmbeddingLayer : public LSTMLayer
{
 public:
  ~EmbeddingLayer();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{embed_};
    return params;
  }


 private:
  int word_dim_;
  int vocab_size_;
  int batch_size_;
  //!< word embedding matrix of size vocab_size_ x word_dim_
  Param* embed_;
};

class HiddenLayer : public LSTMLayer
{
 public:
  ~HiddenLayer();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
  void MergeBlob(Blob<float>* src1, Blob<float>* src2, Blob<float>* blob);
  void GetBlob(Blob<float>* blob,Blob<float>* dst1, Blob<float>* dst2);
  const std::vector<Param*> GetParams() const override
  {
    std::vector<Param*> params{
                        weight_h_i_,weight_x_i_,
                        weight_h_f_,weight_x_f_,
                        weight_h_o_,weight_x_o_,
                        weight_h_g_,weight_x_g_};
    return params;
  }

protected:
  int word_dim,batch_size;
  Blob<float> *i_t_,*f_t_,*o_t_,*g_t_,*c_t_,*h_t_,*c_t_1_,*h_t_1_;//,x_t_;


 private:
  Param* weight_h_i_,*weight_x_i_;
  Param* weight_h_f_,*weight_x_f_;
  Param* weight_h_o_,*weight_x_o_;
  Param* weight_h_g_,*weight_x_g_;
};

/**
 * p(word at t+1 is from class c) = softmax(src[t]*Wc)[c]
 * p(w|c) = softmax(src[t]*Ww[Start(c):End(c)])
 * p(word at t+1 is w)=p(word at t+1 is from class c)*p(w|c)
 */
/*class LossLayer1 : public LSTMLayer
{
 public:
  ~LossLayer1();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;

  const std::string ToString(bool debug, int flag) override;
  const std::vector<Param*> GetParams() const override
  {
    std::vector<Param*> params{word_weight_, class_weight_};
    return params;
  }

 private:
  std::vector<Blob<float>> pword_;
  Blob<float> pclass_;
  Param* word_weight_, *class_weight_;
  float loss_, ppl_;
  int num_;
};*/

}  // namespace lstm
//}
#endif  // EXAMPLES_LSTM_LSTM_H_
