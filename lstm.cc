#include "./lstm.h"

#include <string>
#include <algorithm>
#include "mshadow/tensor.h"
#include "mshadow/tensor_expr.h"
#include "mshadow/tensor_expr_ext.h"
#include "mshadow/tensor_expr_engine-inl.hpp"
#include "mshadow/cxxnet_op.h"
#include "./lstm.pb.h"
#include <iostream>
using namespace std;
namespace lstm {
using std::vector;
using std::string;

using namespace mshadow;
using mshadow::cpu;
using mshadow::Shape;
using mshadow::Shape1;
using mshadow::Shape2;
using mshadow::Shape3;
using mshadow::Tensor;

inline Tensor<cpu, 3> RTensor3(Blob<float>* blob) {
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 3> tensor(blob->mutable_cpu_data(),
      Shape3(shape[0], shape[1], blob->count() / shape[0] / shape[1]));
  return tensor;
}

inline Tensor<cpu, 2> RTensor2(Blob<float>* blob) {
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 2> tensor(blob->mutable_cpu_data(),
      Shape2(shape[0], blob->count() / shape[0]));
  return tensor;
}

inline Tensor<cpu, 1> RTensor1(Blob<float>* blob) {
  Tensor<cpu, 1> tensor(blob->mutable_cpu_data(), Shape1(blob->count()));
  return tensor;
}


DataLayer::~DataLayer()
{
////cout<<endl<<"data_delete_begin"<<endl;
	if(store_!=nullptr){
		delete store_;
 cout<<endl<<"data_delete_end"<<endl;}
}

void DataLayer::Setup(const LayerProto& conf, const vector<Layer*>& srclayers)
{
  //////cout<<endl<<"data::setup"<<endl;
	LSTMLayer::Setup(conf, srclayers);
  	//string key;
  	max_window_=conf.GetExtension(data_conf).max_window();
	//get the batch size
	batch_size_=conf.GetExtension(data_conf).batch_size();
  	//max_window+1 for the begin of the sentence
    //////cout<<endl<<"before reshape :: mw: "<<max_window_<<"    bs: "<<batch_size_<<endl;
	data_.Reshape(vector<int>{batch_size_,(max_window_)});//max_window_+1,batch_size_});

}

void DataLayer::SetIndex(int type, const WordRecord& word, Blob<float>* blob, int param)
{
	float* ptr=blob->mutable_cpu_data()+param*4;
  //  ////cout<<endl<<"type: "<<type;//<<"    word: "<<word.word();
	if(type==0)//WORD AND end
	{
		ptr[0]=static_cast<float>(word.word_index());
		ptr[1]=static_cast<float>(word.class_index());
		ptr[2]=static_cast<float>(word.class_start());
		ptr[3]=static_cast<float>(word.class_end());

	}
	else//BEGIN and pad
  {
    // begin fill -1 and pad fill -2
		//is that right?
    ptr[0]=static_cast<float>(type);
		ptr[1]=static_cast<float>(type);
		ptr[2]=static_cast<float>(type);
		ptr[3]=static_cast<float>(type);
	}


}

void DataLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers)
{

	if (store_ == nullptr)
	{
    //store_= new singa::io::Store();
    store_ = singa::io::OpenStore(
			layer_conf_.GetExtension(data_conf).backend(),
			layer_conf_.GetExtension(data_conf).path(),
			singa::io::kRead);
      cout<<endl<<"open ok"<<endl;
	}
  ////cout<<endl<<"count2"<<endl;
	max_length_=0;
	int temp_length=0;
	for(int i=0;i<batch_size_;i++)
	{
		//max_length_=0;
		if(max_length_<=temp_length)
			max_length_=temp_length;
		temp_length=0;

		SetIndex(-1,word,&data_,i*max_window_);  //set start of string
		int temp_flag_for_eos=1;
		for(int j=1;j<=max_window_;j++)
		{


			if(temp_flag_for_eos!=0)
			{

        if(!store_->Read(&key,&value))
  			{
        //  cout<<endl<<"seek to first count"<<endl;
  				store_->SeekToFirst();
  				CHECK(store_->Read(&key,&value));

  			}
////cout<<endl<<value<<endl;
				word.ParseFromString(value);
      //cout<<endl<<"word: "<<word.word()<<"    i:"<<i<<"    j:"<<j;
				SetIndex(0,word,&data_,i*max_window_+j);
				//float* ptr=data_->mutable_cpu_data()+i*max_window_+j;
				//ptr[0]=static_cast<float>(word.word_index());
				if(word.word_index()!=0)
					temp_length++;
				else
					temp_flag_for_eos=0;
			}
			else
			{

				SetIndex(-2,word,&data_,i*max_window_+j);//set padding

			}
		}
	}
  //cout<<endl<<"data::cf::end"<<endl;

}




UnrollLayer::~UnrollLayer()
{

  for (size_t i = 1; i < datavec_.size(); ++i) {
//    cout<<endl<<"i "<<i;
    if (datavec_[i] != nullptr) delete datavec_[i];
    if (gradvec_[i] != nullptr) delete gradvec_[i];
  }
}

void UnrollLayer::Setup(const LayerProto& conf,
                       const vector<Layer*>& srclayers)
{
  //////cout<<endl<<"unroll setup";
  LSTMLayer::Setup(conf,srclayers);
  batch_size=conf.GetExtension(unroll_conf).batch_size();
  max_window=conf.GetExtension(unroll_conf).max_window();
  //////cout<<endl<<"bs "<<layer_conf_.GetExtension(unroll_conf).batch_size()<<"  mw  "<<max_window<<endl;
  CHECK_EQ(srclayers.size(), 1);
  Layer::Setup(conf, srclayers);
  vector<int> shape = srclayers[0]->data(this).shape();
  ////cout<<endl<<" shape   "<<shape[1]<<endl;

  CHECK_GE(batch_size, 0);
  //CHECK_LT(batch_size, shape.size());
  CHECK_GT(max_window, 0);
  // add max_window-1 more blobs
  for (int i = 1; i < max_window; ++i) {
    datavec_.push_back(new Blob<float>());
    gradvec_.push_back(new Blob<float>());
  }
  // TODO(wangsh): remove equal-size restrict later
//  CHECK_EQ(shape[batch_size] % max_window, 0);
//  shape[batch_size] /= max_window;
  for (int i = 0; i < max_window; ++i) {
    // if (i == Unroll_num - 1) shape[batch_size] += remain;
    datavec_[i]->Reshape(vector<int>{batch_size,1});
    gradvec_[i]->Reshape(vector<int>{batch_size,1});
  }
  ////cout<<endl<<"end setup<"<<endl;
}
//max_window is the unroll size
void UnrollLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  ////cout<<endl<<"ur cf"<<endl;
  //cout<<" Unroll "<<&batch_size<<" "<<max_window;
  CHECK_EQ(srclayers.size(), 1);
  const Blob<float>& blob = srclayers[0]->data(this);
  const float *src=blob.cpu_data();

  for(int i=0;i<max_window;i++)
  {
    float *dst=datavec_[i]->mutable_cpu_data();
    for(int j=0;j<batch_size;j++)
    {
      const float * src=blob.cpu_data();
//      memcpy(dst+j,src[j]+i,sizeof(float));
      *(dst+j)=src[j]+i;
      //cout<<"   index: "<<*(dst+j);

    }
  }
  //cout<<endl<<"end of unroll"<<endl;
}

void UnrollLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  //CHECK_EQ(srclayers.size(), 1);
  //Blob<float>* blob = srclayers[0]->mutable_grad(this);
  // calculate step for each memcpy
  //int step = gradvec_[0]->shape()[batch_size];
  //for (size_t i = batch_size + 1; i < gradvec_[0]->shape().size(); ++i)
  //  step *= gradvec_[0]->shape()[i];
  //int srclayer_offset = 0;
  ////int Unroll_offset = 0;
  ///while (srclayer_offset < blob->count()) {
  //  for (int i = 0; i < max_window; ++i) {
  //    const float* src = gradvec_[i]->cpu_data() + Unroll_offset;
  //    float* dst = blob->mutable_cpu_data() + srclayer_offset;
  //    memcpy(dst, src, step * sizeof(float));
  //    srclayer_offset += step;
//    }
  //  Unroll_offset += step;
//  }
}
////////////////////////
///////////////////////////////
//  for test . I dont know how to use the datavec yet
/////////////////////////////////
/////////////////////////////
const Blob<float>& UnrollLayer::data(const Layer* from) const {
//  CHECK(from);
//  CHECK_LT(from->partition_id(), max_window);
//////cout<<from->partition_id();
  return *datavec_[0];//from->partition_id()];
}

const Blob<float>& UnrollLayer::grad(const Layer* from) const {
//CHECK(from);
//  CHECK_LT(from->partition_id(), max_window);
////cout<<"here2";
  return *gradvec_[0];//from->partition_id()];
}

Blob<float>* UnrollLayer::mutable_data(const Layer* from) {
//  CHECK(from);
///  CHECK_LT(from->partition_id(), max_window);
////cout<<"here3";
  return datavec_[0];//from->partition_id()];
}

Blob<float>* UnrollLayer::mutable_grad(const Layer* from) {
//  CHECK(from);
//  CHECK_LT(from->partition_id(), max_window);
////cout<<"here4";
  return gradvec_[0];//from->partition_id()];
}


EmbeddingLayer::~EmbeddingLayer()
{

////cout<<endl<<"emb_delete_begin"<<endl;
	delete embed_;
////cout<<endl<<"emb_delete_end"<<endl;
}

void EmbeddingLayer::Setup(const LayerProto& conf, const vector<Layer*>& srclayers)
{
////cout<<endl<<"embed::setup"<<endl;
  LSTMLayer::Setup(conf,srclayers);
	CHECK_EQ(srclayers.size(),1);
	///*int max_window*/int batch_size = srclayers[0]->data(this).shape()[0];
	word_dim_ = conf.GetExtension(embedding_conf).word_dim();
	//get the batch size
        batch_size_=conf.GetExtension(embedding_conf).batch_size();

  	data_.Reshape(vector<int>{batch_size_,word_dim_});
  	grad_.ReshapeLike(data_);
  	vocab_size_ = conf.GetExtension(embedding_conf).vocab_size();

  	embed_ = Param::Create(conf.param(0));
  	embed_->Setup(vector<int>{vocab_size_, word_dim_});
    ////cout<<endl<<"embed::setup::ok"<<endl;
}

void EmbeddingLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers)
{
////cout<<endl<<"embed::cf"<<endl;
//cout<<endl<<" Embed "<<&word_dim_<<" "<<&vocab_size_<<" "<<&batch_size_<<endl;

//cout<<endl<<"srclayer size: "<<srclayers.size();

  auto datalayer=dynamic_cast<UnrollLayer*>(srclayers[0]);

	auto words=RTensor2(&data_);

	auto embed=RTensor2(embed_->mutable_data());
//cout<<endl<<"data shape  "<<embed.shape[0]<<" "<<embed.shape[1];
///////////	const float* idxptr=datalayer->data(this).cpu_data();

const float* idxptr=datalayer->data(this).cpu_data();


//  ////cout<< endl<<"idxptr"<<*idxptr;
	int temp_pad_count=0;
	int temp_end_count=0;
	int temp_begin_count=0;
	for (int t=0;t<batch_size_;t++)
	{
		int idx=static_cast<int>(idxptr[t*4]);
    //////cout<<endl<<"idx: "<<idx;
		if(idx==-1)
		{	//for the start index
			//*******
			temp_begin_count++;
			//skip the Copy operation , remain words[t] null?
			continue;
		}
		if(idx==-2)
		{
		//for the padding index
			// like the operation of start index
			temp_pad_count++;
			continue;
		}
		if(idx==0)
		{
			//for the end index
			temp_end_count++;
			//sign the end index to return -1
			//continue;
			//end index will be filled into the words[t]
		}
		CHECK_GE(idx, 0);
    CHECK_LT(idx, vocab_size_);
    Copy(words[t], embed[idx]);
    //cout<<"data   ::  "<<idx<<endl;
	}
	if(temp_pad_count==batch_size_||temp_begin_count==batch_size_)
	{
		//if there is another fill option fo the all pad or start index
	}
	if(temp_end_count==batch_size_)
	{
		//all end indexes return -1
	}
  //cout<<endl<<"embed::cf::end"<<endl;
//////cout<<endl<<"embed to string "<<this->ToString(1,singa::kTrain)<<endl;
}


void EmbeddingLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers)
{
////cout<<endl<<"embed::cg"<<endl;

        auto datalayer=dynamic_cast<DataLayer*>(srclayers[0]);

        auto grad=RTensor2(&grad_);
        auto gembed=RTensor2(embed_->mutable_grad());

        const float* idxptr=datalayer->data(this).cpu_data();

        for (int t=0;t<batch_size_;t++)
        {
                int idx=static_cast<int>(idxptr[t*4]);

                Copy(gembed[idx], grad[t]);
        }
        ////cout<<endl<<"embed::cg::end"<<endl;
}

HiddenLayer::~HiddenLayer()
{
  ////cout<<endl<<"hid_delete_begin"<<endl;

  delete weight_h_i_;
  delete weight_x_i_;
  delete weight_h_f_;
  delete weight_x_f_;
          delete weight_h_o_;
          delete weight_x_o_;
          delete weight_h_g_;
          delete weight_x_g_;
            ////cout<<endl<<"hid_delete_end"<<endl;
}

void HiddenLayer::Setup(const LayerProto& conf,const vector<Layer*>& srclayers)
{
////cout<<endl<<"hid::setup"<<endl;
  	LSTMLayer::Setup(conf, srclayers);
  	//CHECK_EQ(srclayers.size(), 2);
    //there are two srclayers, one is the embed layer
    //another is the previous lstm hidden layer
    word_dim = conf.GetExtension(hid_conf).word_dim();
  	//get the batch size
          batch_size=conf.GetExtension(hid_conf).batch_size();
  	//data_.ReshapeLike(srclayers[0]->data(this));
  	grad_.ReshapeLike(srclayers[0]->grad(this));
  //	int word_dim =// data_.shape()[1];
  //  int batch_size=//data_.shape()[0];
    //////cout<<endl<<"wd:  "<<word_dim<<"  bs:   "<<batch_size<<endl;
  	weight_h_i_ = Param::Create(conf.param(0));
  	weight_h_i_->Setup(std::vector<int>{word_dim, word_dim});

    weight_x_i_ = Param::Create(conf.param(1));
    weight_x_i_->Setup(std::vector<int>{word_dim, word_dim});
    weight_h_f_ = Param::Create(conf.param(2));
    weight_h_f_->Setup(std::vector<int>{word_dim, word_dim});
    weight_x_f_ = Param::Create(conf.param(3));
    weight_x_f_->Setup(std::vector<int>{word_dim, word_dim});
    weight_h_o_ = Param::Create(conf.param(4));
    //////cout<<endl<<"reshape"<<endl;
    weight_h_o_->Setup(std::vector<int>{word_dim, word_dim});
    weight_x_o_ = Param::Create(conf.param(5));
    weight_x_o_->Setup(std::vector<int>{word_dim, word_dim});
    weight_h_g_ = Param::Create(conf.param(6));
    weight_h_g_->Setup(std::vector<int>{word_dim, word_dim});
    weight_x_g_ = Param::Create(conf.param(7));
    weight_x_g_->Setup(std::vector<int>{word_dim, word_dim});
    //data_ should be reshaped to contain 2*batch_size_*word_dim data
    //data_[0] will be h_t
    //data_[1] will be c_t
    //
    //////cout<<"&&&&&&&&&&&&&&&&";
    data_.Reshape(vector<int>{2,batch_size,word_dim});//{word_dim,batch_size,2});//2,batch_size,word_dim});
    //////cout<<"**********************";

    i_t_.Reshape(vector<int>{batch_size,word_dim});
    f_t_.Reshape(vector<int>{batch_size,word_dim});
    o_t_.Reshape(vector<int>{batch_size,word_dim});
    g_t_.Reshape(vector<int>{batch_size,word_dim});
    c_t_.Reshape(vector<int>{batch_size,word_dim});
    h_t_.Reshape(vector<int>{batch_size,word_dim});
    c_t_1_.Reshape(vector<int>{batch_size,word_dim});
    h_t_1_.Reshape(vector<int>{batch_size,word_dim});
    //x_t_.Reshape(vector<int>{batch_size,word_dim});
    //Reshape all the matrices to batch_size*word_dim in case to convert Tensor

    ////cout<<endl<<"hid::setup::ok"<<endl;
}

//I've another code that seperate the cell layer
void HiddenLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers)
{

////cout<<endl<<"hid::cf"<<endl;
  ////these weight could be merge into one matrix
  ////i will think this later
  ////so do the gates
  auto weight_h_i = RTensor2(weight_h_i_->mutable_data());
  auto weight_x_i = RTensor2(weight_x_i_->mutable_data());
  auto weight_h_f = RTensor2(weight_h_f_->mutable_data());
  auto weight_x_f = RTensor2(weight_x_f_->mutable_data());
  auto weight_h_o = RTensor2(weight_h_o_->mutable_data());
  auto weight_x_o = RTensor2(weight_x_o_->mutable_data());
  auto weight_h_g = RTensor2(weight_h_g_->mutable_data());
  auto weight_x_g = RTensor2(weight_x_g_->mutable_data());

  //initialize the variables Tensor
  auto i_t=RTensor2(&i_t_);//.mutable_cpu_data());
  auto f_t=RTensor2(&f_t_);//.mutable_cpu_data());
  auto o_t=RTensor2(&o_t_);//.mutable_cpu_data());
  auto g_t=RTensor2(&g_t_);//.mutable_cpu_data());
  auto c_t=RTensor2(&c_t_);//.mutable_cpu_data());
  auto h_t=RTensor2(&h_t_);//.mutable_cpu_data());
  auto c_t_1=RTensor2(&c_t_1_);//.mutable_cpu_data());
  auto h_t_1=RTensor2(&h_t_1_);//.mutable_cpu_data());
  //auto x_t=RTensor2(x_t_->mutable_data());
/*////cout<<endl<<"print the dptr "<<weight_x_i_<<" "
<<weight_x_f_<<" "<<weight_x_o_<<" "<<weight_x_g_<<" "<<weight_h_i_<<" "
<<weight_h_f_<<" "<<weight_h_o_<<" "<<weight_h_g_;/*<<" "<<i_t.dptr<<" "
<<f_t.dptr<<" "<<o_t.dptr<<" "<<g_t.dptr<<" "
<<c_t.dptr<<" "<<h_t.dptr<<" "
<<c_t_1.dptr<<" "<<h_t_1.dptr<<" ";*/
  auto data = RTensor3(&data_);
  //x_t stands for the input word
  auto x_t = RTensor2(srclayers[0]->mutable_data(this));
  //Copy(x_t,srclayers[0]->mutable_data(this));

  /////I dont know if the first word has no previous layer

  if(srclayers.size()==1)// this means the first lstm node
  {

    i_t=expr::F<op::sigmoid>(dot(weight_x_i,x_t));
            //expr::F<op::mul>(weight_x_i,x_t));
//////cout<<endl<<"shape w  "<<weight_x_i.shape[0]<<" "<<weight_x_i.shape[1]<<"  shape x  "<<x_t.shape[0]<<" "<<x_t.shape[1]<<endl;
    f_t=expr::F<op::sigmoid>(dot(weight_x_f,x_t));
            //expr::F<op::mul>(weight_x_f,x_t));

    o_t=expr::F<op::sigmoid>(dot(weight_x_o,x_t));
            //expr::F<op::mul>(weight_x_o,x_t));

    g_t=expr::F<op::tanh>(dot(weight_x_g,x_t));
            //expr::F<op::mul>(weight_x_g,x_t));

//////cout<<endl<<"shape i  "<<i_t.shape[0]<<" "<<i_t.shape[1]<<"  shape g  "<<g_t.shape[0]<<" "<<g_t.shape[1]<<endl;
    //c_t=dot(i_t,g_t);
    c_t=expr::F<op::mul>(i_t,g_t);
//////cout<<endl<<"end calc"<<endl;
    auto temp_tanh_c_t=expr::F<op::tanh>(c_t);
    //h_t=dot(o_t,temp_tanh_c_t_);
    h_t=expr::F<op::mul>(o_t,temp_tanh_c_t);
    //h_t=dot(o_t,expr::F<op::tanh>(c_t));

    //data[0]=h_t;
    Copy(data[0],h_t);
    //data[1]=c_t;
    Copy(data[1],c_t);
    //////cout<<endl<<" to string "<<this->ToString(1,singa::kTrain)<<endl;
//    ////cout<<endl<<"data shape "<<data.shape[0]<<"  "<<data.shape[1]<<"  "<<data.shape[2]<<endl;
//////cout<<endl<<"*********dptr:  "<<(data[0].dptr)<<"  "<<data[1].dptr<<endl;
////cout<<endl<<"end  hid one srclayer"<<endl;
    return;

    //end the first node
  }
  // so need not to operate the h_t-1 and c_t-1

  //////0+1 times further steps had the previous layer connected

  //h_t_1 c_t_1 stands for the h(t-1) c(t-1) from previous lstm hidden layer
  //extract from the src_t_1 variable
  auto src_t_1=RTensor3(srclayers[1]->mutable_data(this));
  //auto h_t_1=src_t_1[0];
  //auto c_t_1=src_t_1[1];
  Copy(h_t_1,src_t_1[0]);
  Copy(c_t_1,src_t_1[1]);
  //cout<<endl<<"ht1 addr "<<h_t_1.dptr<<" srct1[0] addr "<<src_t_1[0].dptr;


	/*forwardpass for hidden layer
  i_t := sigmoid[ W_{hi} * h_{t-1} + W_{xi} * x_t + b_i ]
	f_t := sigmoid[ W_{hf} * h_{t-1} + W_{xf} * x_t + b_f ]
	o_t := sigmoid[ W_{ho} * h_{t-1} + W_{xo} * x_t + b_o ]
	g_t := tanh[ W_{hg} * h_{t-1} + W_{xg} * x_t + b_g ]
	c_t := (f_t .* c_{t-1}) + (i_t .* g_t)
	h_t := o_t .* tanh[c_t]
	*/
  //  ignore the bias first, add then

  /*i_t=expr::F<op::sigmoid>(
        expr::F<op::plus>(
          expr::F<op::mul>(weight_h_i,h_t_1),
          expr::F<op::mul>(weight_x_i,x_t)));*/
  ///
  ////cout<<endl<<" start 2 layers calc"<<endl;
  i_t=expr::F<op::sigmoid>(
    dot(weight_h_i,h_t_1)+dot(weight_x_i,x_t)
  );
////cout<<endl<<"x_t shape  "<<i_t.shape[0]<<"  "<<i_t.shape[1]<<endl;
//cout<<endl<<"matrix";
for(int i=0;i<(static_cast<int>(i_t.shape[0]));i++){
  //cout<<endl;
  for(int j=0;j<(static_cast<int>(i_t.shape[1]));j++){
  //cout<<" "<<i_t[i][j];
}
}



  /*f_t=expr::F<op::sigmoid>(
        expr::F<op::plus>(
            expr::F<op::mul>(weight_h_f,h_t_1),
            expr::F<op::mul>(weight_x_f,x_t)));*/
  ///
  f_t=expr::F<op::sigmoid>(
    dot(weight_h_f,h_t_1)+dot(weight_x_f,x_t)
  );


  /*o_t=expr::F<op::sigmoid>(
        expr::F<op::plus>(
            expr::F<op::mul>(weight_h_o,h_t_1),
            expr::F<op::mul>(weight_x_o,x_t)));*/
  ///
  o_t=expr::F<op::sigmoid>(
    dot(weight_h_o,h_t_1)+dot(weight_x_o,x_t)
  );


  /*g_t=expr::F<op::tanh>(
        expr::F<op::plus>(
            expr::F<op::mul>(weight_h_g,h_t_1),
            expr::F<op::mul>(weight_x_g,x_t)));*/
  ///
  g_t=expr::F<op::tanh>(
    dot(weight_h_g,h_t_1)+dot(weight_x_g,x_t)
  );


  /*c_t=expr::F<op::plus>(
        dot(f_t,c_t_1),
        dot(i_t,g_t));*/
  ///
  c_t=dot(f_t,c_t_1)+dot(i_t,g_t);

  auto temp_tanh_c_t=expr::F<op::tanh>(c_t);
  //h_t=dot(o_t,temp_tanh_c_t_);
  h_t=expr::F<op::mul>(o_t,temp_tanh_c_t);// F<op::mul>
  //h_t=dot(o_t,expr::F<op::tanh>(c_t));

  //data[0]=h_t;
  Copy(data[0],h_t);
  //data[1]=c_t;
  Copy(data[1],c_t);


//temp_tanh_c_t.FreeSpace();

//cout<<endl<<"  "<<i_t.dptr<<" "<<h_t_1.dptr<<" "<<weight_h_i.dptr<<" "<<weight_x_g.dptr;
//cout<<endl<<"*********dptr:  "<<&i_t_<<"  "<<&h_t_1<<" "<<weight_h_i_<<" "<<weight_x_g_;

  //the c(t) h(t) should be pass to next lstm hidden layer
  //////
  //the t-1 time data is from srclayers.
  //one lstm unit only do the unrolled operation
  //////

////cout<<endl<<"hid::cf::end"<<endl;

}


void HiddenLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers)
{
/*///cout<<endl<<"hid::cg"<<endl;
  //auto data = RTensor3(&data_);
  //auto grad = RTensor2(&grad_);
  //auto weight = RTensor2(weight_->mutable_data());
  //auto gweight = RTensor2(weight_->mutable_grad());
  //auto gsrc = RTensor2(srclayers[0]->mutable_grad(this));

  auto data = RTensor3(&data_);
  auto grad = RTensor3(&grad_);

  auto x_t = RTensor2(srclayers[0]->mutable_data(this));

  auto weight_h_i = RTensor2(weight_h_i_->mutable_data());
  auto weight_x_i = RTensor2(weight_x_i_->mutable_data());
  auto weight_h_f = RTensor2(weight_h_f_->mutable_data());
  auto weight_x_f = RTensor2(weight_x_f_->mutable_data());
  auto weight_h_o = RTensor2(weight_h_o_->mutable_data());
  auto weight_x_o = RTensor2(weight_x_o_->mutable_data());
  auto weight_h_g = RTensor2(weight_h_g_->mutable_data());
  auto weight_x_g = RTensor2(weight_x_g_->mutable_data());
  auto i_t=RTensor2(&i_t_);//.mutable_cpu_data());
  auto f_t=RTensor2(&f_t_);//.mutable_cpu_data());
  auto o_t=RTensor2(&o_t_);//.mutable_cpu_data());
  auto g_t=RTensor2(&g_t_);//.mutable_cpu_data());
  auto c_t=RTensor2(&c_t_);//.mutable_cpu_data());
  auto h_t=RTensor2(&h_t_);//.mutable_cpu_data());
  auto c_t_1=RTensor2(&c_t_1_);//.mutable_cpu_data());
  auto h_t_1=RTensor2(&h_t_1_);//.mutable_cpu_data());
  //auto x_t=RTensor2(x_t_->mutable_data());
  //these above have values done in the forwardpass
  //the values will be used in the backwardpass gradient compute


  if(srclayers.size()==1)// this means the first lstm node
  {

    return;
    //end the first node
  }


  //Backwardpass gradient euqations below
  /////////////////////////////////
  //eq1. Forward:   h_t := o_t .* tanh[c_t]
  //grad_h_t is known
  //<1> grad_o_t = (grad_h_t) dot (tanh(c_t))
  //<2> grad_c_t = (grad_h_t) dot (o_t) dot (tan_grad(c_t))
  /////////////////////////////////

  auto grad_c_t=dot(dot(grad_h_t,o_t),expr::F<OP::tanh_grad>(c_t));
  auto grad_o_t=dot(grad_h_t,expr::F<op::tanh>(grad_c_t));


  /////////////////////////////////////
  //eq2. Forward:   c_t := (f_t .* c_{t-1}) + (i_t .* g_t)
  //grad_c_t is calculated
  //<3> grad_i_t = (grad_c_t) dot (g_t)
  //<4> grad_f_t = (grad_c_t) dot (c_t_1)
  //<5> grad_g_t = (grad_c_t) dot (i_t)
  //<6> grad_c_t_1 = (grad_c_t) dot (f_t)
  /////////////////////////////////////

  auto grad_i_t=dot(grad_c_t,g_t);
  auto grad_f_t=dot(grad_c_t,c_t_1);
  auto grad_g_t=dot(grad_c_t,i_t);
  auto grad_c_t_1=dot(grad_c_t,f_t);

  ////////////////////////////////////////
  //eq3. Weight update step1
  //calculate the [i,f,o,g]=Wh*h(t-1)+Wx*x
  //<7> grad_i = (grad_i_t) dot (sigmoid_grad(i_t))
  //<8> grad_f = (grad_f_t) dot (sigmoid_grad(f_t))
  //<9> grad_o = (grad_o_t) dot (tanh_grad(o_t))
  //<10> grad_g = (grad_g_t) dot (sigmoid_grad(g_t))
  //////////////////////////////////////

  auto grad_i=dot(grad_i_t,expr::F<op::sigmoid_grad(i_t));
  auto grad_f=dot(grad_f_t,expr::F<op::sigmoid_grad(f_t));
  auto grad_o=dot(grad_o_t,expr::F<op::tanh_grad(o_t));
  auto grad_g=dot(grad_g_t,expr::F<op::sigmoid_grad(g_t));

  //////////////////////////////////////////
  //eq4. Weight update step2
  //calculate the grad_w(ifog) and grad_h_t_1
  //<11> grad_w(ifog) = grad_(ifog) multiply x_t.transform()
  //<12> grad_h_t_1 =
  /////////////////////////////////

*/
////cout<<endl<<"hid::cg::end"<<endl;

}

/*********** Implementation for LossLayer **********/
LossLayer::~LossLayer()
{

}

void LossLayer::Setup(const LayerProto& conf, const vector<Layer*>& srclayers)
{

}

void LossLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers)
{

}

void LossLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers)
{

}

const std::string LossLayer::ToString(bool debug, int flag)
{

}


////end of namespace lstm
}
