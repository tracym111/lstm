#include "./lstm-newmathop.h"

#include <string>
#include <algorithm>

#include "singa/utils/math_blob.h"
#include "singa/utils/singa_op.h"
#include <glog/logging.h>
#include "singa/utils/singleton.h"

#include "./lstm.pb.h"
#include <iostream>
using namespace std;
using namespace singa;

//namespace lstm{

namespace singa{

using std::vector;
using std::string;



DataLayer::~DataLayer()
{
	if(store_!=nullptr)
		delete store_;
}

void DataLayer::Setup(const LayerProto& conf, const vector<Layer*>& srclayers)
{
	LSTMLayer::Setup(conf, srclayers);
  	string key;
  	max_window_=conf.GetExtension(data_conf).max_window();
	//get the batch size
	batch_size_=conf.GetExtension(data_conf).batch_size();
  	//max_window+1 for the begin of the sentence
	data_.Reshape(vector<int>{batch_size_,max_window_+1});

}

void SetIndex(int type, const WordRecord& word, Blob<float>* blob, int param)
{
	float* ptr=blob->mutable_cpu_data()+param*4;
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
	string key,value;
	WordRecord word;
	if (store_ == nullptr)
	{
		store_ = singa::io::OpenStore(
			layer_conf_.GetExtension(data_conf).backend(),
			layer_conf_.GetExtension(data_conf).path(),
			singa::io::kRead);
	}
	max_length_=0;
	int temp_length=0;
	for(int i=0;i<batch_size_;i++)
	{
		//max_length_=0;
		if(max_length_<=temp_length)
			max_length_=temp_length;
		temp_length=0;
		SetIndex(-1,word,&data_,i*max_window_);  //set start of string
		int temp_flag_for_eos=0;
		for(int j=1;j<=max_window_;j++)
		{
			if(!store_->Read(&key,&value))
			{
				store_->SeekToFirst();
				CHECK(store_->Read(&key,&value));
			}
			if(temp_flag_for_eos!=0)
			{
				word.ParseFromString(value);
				SetIndex(0,word,&data_,i*max_window_+j);
				//float* ptr=data_->mutable_cpu_data()+i*max_window_+j;
				//ptr[0]=static_cast<float>(word.word_index());
				if(word.word_index()!=0)
					temp_length++;
				else
					temp_flag_for_eos==0;
			}
			else
			{

				SetIndex(-2,word,&data_,i*max_window_+j);//set padding

			}
		}
	}
}



EmbeddingLayer::~EmbeddingLayer()
{
	delete embed_;
}


void InsertValue(Blob<float>* src, Blob<float>* blob, int param1,int param2)
{
	float* ptr=blob->mutable_cpu_data()+param1;
  float* ptr_src=src->mutable_cpu_data()+param2;
	ptr[0]=static_cast<float>(*ptr_src);

}

void EmbeddingLayer::Setup(const LayerProto& conf, const vector<Layer*>& srclayers)
{
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
}


void EmbeddingLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers)
{
	auto datalayer=dynamic_cast<DataLayer*>(srclayers[0]);

	//auto words=RTensor2(&data_);
	//auto embed=RTensor2(embed_->mutable_data());

  Blob<float> *embed=embed_->mutable_data();

	const float* idxptr=datalayer->data(this).cpu_data();
	int temp_pad_count=0;
	int temp_end_count=0;
	int temp_begin_count=0;
	for (int t=0;t<batch_size_;t++)
	{
		int idx=static_cast<int>(idxptr[t*4]);
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
    		//Copy(data_[t], embed[idx]);
        InsertValue(&data_,embed,t,idx);
	}
	if(temp_pad_count==batch_size_||temp_begin_count==batch_size_)
	{
		//if there is another fill option fo the all pad or start index
	}
	if(temp_end_count==batch_size_)
	{
		//all end indexes return -1
	}
}


void EmbeddingLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers)
{
        auto datalayer=dynamic_cast<DataLayer*>(srclayers[0]);

        //auto grad=RTensor2(&grad_);
        //auto gembed=RTensor2(embed->mutable_grad());
        Blob<float> *gembed=embed_->mutable_grad();

        const float* idxptr=datalayer->data(this).cpu_data();

        for (int t=0;t<batch_size_;t++)
        {
                int idx=static_cast<int>(idxptr[t*4]);

                //Copy(gembed[idx], grad_[t]);
                InsertValue(gembed,&grad_,idx,t);
        }
}

HiddenLayer::~HiddenLayer()
{
  delete  weight_h_i_,weight_x_i_,
          weight_h_f_,weight_x_f_,
          weight_h_o_,weight_x_o_,
          weight_h_g_,weight_x_g_;
}


void HiddenLayer::Setup(const LayerProto& conf,const vector<Layer*>& srclayers)
{
  	LSTMLayer::Setup(conf, srclayers);
  	//CHECK_EQ(srclayers.size(), 2);
    //there are two srclayers, one is the embed layer
    //another is the previous lstm hidden layer

  	data_.ReshapeLike(srclayers[0]->data(this));
  	grad_.ReshapeLike(srclayers[0]->grad(this));
  	word_dim = data_.shape()[1];
    batch_size=data_.shape()[0];
  	weight_h_i_ = Param::Create(conf.param(0));
  	weight_h_i_->Setup(std::vector<int>{word_dim, word_dim});
    weight_x_i_ = Param::Create(conf.param(1));
    weight_x_i_->Setup(std::vector<int>{word_dim, word_dim});
    weight_h_f_ = Param::Create(conf.param(2));
    weight_h_f_->Setup(std::vector<int>{word_dim, word_dim});
    weight_x_f_ = Param::Create(conf.param(3));
    weight_x_f_->Setup(std::vector<int>{word_dim, word_dim});
    weight_h_o_ = Param::Create(conf.param(4));
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
    data_.Reshape(vector<int>{2,batch_size,word_dim});


    i_t_=new Blob<float>(batch_size,word_dim);
    f_t_=new Blob<float>(batch_size,word_dim);
    o_t_=new Blob<float>(batch_size,word_dim);
    g_t_=new Blob<float>(batch_size,word_dim);
    c_t_=new Blob<float>(batch_size,word_dim);
    h_t_=new Blob<float>(batch_size,word_dim);
    c_t_1_=new Blob<float>(batch_size,word_dim);
    h_t_1_=new Blob<float>(batch_size,word_dim);



  /*  Blob<float> i_t_(batch_size,word_dim);
    Blob<float> f_t_(batch_size,word_dim);
    Blob<float> o_t_(batch_size,word_dim);
    Blob<float> g_t_(batch_size,word_dim);
    Blob<float> c_t_(batch_size,word_dim);
    Blob<float> h_t_(batch_size,word_dim);
    Blob<float> h_t_1_(batch_size,word_dim);
    Blob<float> c_t_1_(batch_size,word_dim);*/

    //x_t_.Reshape(vector<int>{batch_size,word_dim});
    //Reshape all the matrices to batch_size*word_dim in case to convert Tensor
}

void HiddenLayer::MergeBlob(Blob<float>* src1, Blob<float>* src2, Blob<float>* blob)
{
	float* ptr0=src1->mutable_cpu_data();
	float* ptr1=src2->mutable_cpu_data();
	float* ptr2=blob->mutable_cpu_data();
	float* ptr3=blob->mutable_cpu_data()+batch_size*word_dim;

	*ptr2=*ptr0;
	*ptr3=*ptr1;


}

void HiddenLayer::GetBlob(Blob<float>* blob,Blob<float>* dst1, Blob<float>* dst2)
{
	float* ptr0=dst1->mutable_cpu_data();
	float* ptr1=dst2->mutable_cpu_data();
	float* ptr2=blob->mutable_cpu_data();
	float* ptr3=blob->mutable_cpu_data()+batch_size*word_dim;

	*ptr0=*ptr2;
	*ptr1=*ptr3;


}


//I've another code that seperate the cell layer
void HiddenLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers)
{


  /*Blob<float> temp1(batch_size,word_dim);
  Blob<float> temp2(batch_size,word_dim);
  Blob<float> temp3(batch_size,word_dim);*/
	Blob<float>* temp1=new Blob<float>(batch_size,word_dim);
	Blob<float>* temp2=new Blob<float>(batch_size,word_dim);
	Blob<float>* temp3=new Blob<float>(batch_size,word_dim);

  Blob<float> *weight_h_i = Transpose(weight_h_i_->data());
  Blob<float> *weight_x_i = Transpose(weight_x_i_->data());
  Blob<float> *weight_h_f = Transpose(weight_h_f_->data());
  Blob<float> *weight_x_f = Transpose(weight_x_f_->data());
  Blob<float> *weight_h_o = Transpose(weight_h_o_->data());
  Blob<float> *weight_x_o = Transpose(weight_x_o_->data());
  Blob<float> *weight_h_g = Transpose(weight_h_g_->data());
  Blob<float> *weight_x_g = Transpose(weight_x_g_->data());

  //auto data = RTensor3(&data_);
  //x_t stands for the input word
  //auto x_t = RTensor2(srclayers[0]->mutable_data(this));
  Blob<float> *x_t=srclayers[0]->mutable_data(this);
  //Copy(x_t,srclayers[0]->mutable_data(this));


  /////I dont know if the first word has no previous layer
  /////this srclayers.size() will be 1 ? or still be 2 ?
  if(srclayers.size()==1)// this means the first lstm node
  {
    GEMM(ccpu,1.0f,0.0f,*weight_x_i,*x_t,i_t_);
    Map<op::Sigmoid<float>,float>(ccpu,*i_t_,i_t_);

    GEMM(ccpu,1.0f,0.0f,*weight_x_f,*x_t,f_t_);
    Map<op::Sigmoid<float>,float>(ccpu,*f_t_,f_t_);

    GEMM(ccpu,1.0f,0.0f,*weight_x_o,*x_t,o_t_);
    Map<op::Sigmoid<float>,float>(ccpu,*o_t_,o_t_);

    GEMM(ccpu,1.0f,0.0f,*weight_x_g,*x_t,g_t_);
    Map<op::Tanh<float>,float>(ccpu,*g_t_,g_t_);

    //*c_t_=VVDot(ccpu,)
    MMDot(ccpu,*i_t_,*g_t_,c_t_);
    //c_t=dot(i_t,g_t));

    Blob<float> *temp_tanh_c_t_=new Blob<float>(batch_size,word_dim);
    Map<op::Tanh<float>,float>(ccpu,*c_t_,temp_tanh_c_t_);
    MMDot(ccpu,*o_t_,*temp_tanh_c_t_,h_t_);
    //h_t=dot(o_t,expr::F<op::tanh>(c_t));

    //data[0]=h_t_;
    //Copy(data_[0],h_t_);
    //data[1]=c_t_;
    //Copy(data_[1],c_t_);
		MergeBlob(h_t_,c_t_,&data_);

    return;
    //end the first node
  }
  // so need not to operate the h_t-1 and c_t-1

  //////0+1 times further steps had the previous layer connected

  //h_t_1 c_t_1 stands for the h(t-1) c(t-1) from previous lstm hidden layer
  //extract from the src_t_1 variable
  //auto src_t_1=RTensor3(srclayers[1]->mutable_data(this));
  //Blob<float> *src_t_1=
  auto src_t_1=srclayers[1]->mutable_data(this);
  //auto h_t_1=src_t_1[0];
  //auto c_t_1=src_t_1[1];
  //Copy(h_t_1_,src_t_1[0]);
  //Copy(c_t_1_,src_t_1[1]);
	GetBlob(src_t_1,h_t_1_,c_t_1_);


	/*forwardpass for hidden layer
  i_t := sigmoid[ W_{hi} * h_{t-1} + W_{xi} * x_t + b_i ]
	f_t := sigmoid[ W_{hf} * h_{t-1} + W_{xf} * x_t + b_f ]
	o_t := sigmoid[ W_{ho} * h_{t-1} + W_{xo} * x_t + b_o ]
	g_t := tanh[ W_{hg} * h_{t-1} + W_{xg} * x_t + b_g ]
	c_t := (f_t .* c_{t-1}) + (i_t .* g_t)
	h_t := o_t .* tanh[c_t]
	*/
  //  ignore the bias first, add then


  GEMM(ccpu,1.0f,0.0f,*weight_h_i,*h_t_1_,temp1);
  GEMM(ccpu,1.0f,0.0f,*weight_x_i,*x_t,temp2);
  Add<float>(ccpu,*temp1,*temp2,temp3);
  Map<op::Sigmoid<float>,float>(ccpu,*temp3,i_t_);


  GEMM(ccpu,1.0f,0.0f,*weight_h_f,*h_t_1_,temp1);
  GEMM(ccpu,1.0f,0.0f,*weight_x_f,*x_t,temp2);
  Add<float>(ccpu,*temp1,*temp2,temp3);
  Map<op::Sigmoid<float>,float>(ccpu,*temp3,f_t_);


  GEMM(ccpu,1.0f,0.0f,*weight_h_o,*h_t_1_,temp1);
  GEMM(ccpu,1.0f,0.0f,*weight_x_o,*x_t,temp2);
  Add<float>(ccpu,*temp1,*temp2,temp3);
  Map<op::Sigmoid<float>,float>(ccpu,*temp3,o_t_);


  GEMM(ccpu,1.0f,0.0f,*weight_h_g,*h_t_1_,temp1);
  GEMM(ccpu,1.0f,0.0f,*weight_x_g,*x_t,temp2);
  Add<float>(ccpu,*temp1,*temp2,temp3);
  Map<op::Tanh<float>,float>(ccpu,*temp3,g_t_);


  MMDot(ccpu,*i_t_,*g_t_,temp1);
  MMDot(ccpu,*f_t_,*c_t_1_,temp2);
  Add<float>(ccpu,*temp1,*temp2,c_t_);

  //h_t=dot(o_t,expr::F<op::tanh>(c_t));
  Blob<float> *temp_tanh_c_t_=new Blob<float>(batch_size,word_dim);
  Map<op::Tanh<float>,float>(ccpu,*c_t_,temp_tanh_c_t_);
  MMDot(ccpu,*o_t_,*temp_tanh_c_t_,h_t_);

  //data[0]=h_t_;
  //Copy(data_[0],h_t_);
  //data[1]=c_t_;
//  Copy(data_[1],c_t_);
	MergeBlob(h_t_,c_t_,&data_);


  //the c(t) h(t) should be pass to next lstm hidden layer
  //////
  //the t-1 time data is from srclayers.
  //one lstm unit only do the unrolled operation
  //////



  //for develop notes of the math operations
	//sigmoid operation is like this:
	//data[t] = expr::F<op::sigmoid>(src[t]);
	//grad[t] = expr::F<op::sigmoid_grad>(data[t])* grad[t];
	//tanh operation is like this:
	//data = expr::F<op::stanh>(src);
	//gsrc = expr::F<op::stanh_grad>(data) * grad;

}


void HiddenLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers)
{
/*
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
  auto i_t=RTensor2(i_t_->mutable_data());
  auto f_t=RTensor2(f_t_->mutable_data());
  auto o_t=RTensor2(o_t_->mutable_data());
  auto g_t=RTensor2(g_t_->mutable_data());
  auto c_t=RTensor2(c_t_->mutable_data());
  auto h_t=RTensor2(h_t_->mutable_data());
  auto c_t_1=RTensor2(c_t_1_->mutable_data());
  auto h_t_1=RTensor2(h_t_1_->mutable_data());
  //these above have values done in the forwardpass
  //the values will be used in the backwardpass gradient compute


  if(srclayers.size()==1)// this means the first lstm node
  {

    return;
    //end the first node
  }

*/
  //Backwardpass gradient euqations below
  /*////////////////////////////////
  eq1. Forward:   h_t := o_t .* tanh[c_t]
  grad_h_t is known
  <1> grad_o_t = (grad_h_t) dot (tanh(c_t))
  <2> grad_c_t = (grad_h_t) dot (o_t) dot (tan_grad(c_t))

  eq2. Forward:   c_t := (f_t .* c_{t-1}) + (i_t .* g_t)
  grad_c_t is calculated
  <3> grad_i_t = (grad_c_t) dot (g_t)
  <4> grad_f_t = (grad_c_t) dot (c_t_1)
  <5> grad_g_t = (grad_c_t) dot (i_t)
  <6> grad_c_t_1 = (grad_c_t) dot (f_t)

  eq3. Weight update step1
  calculate the [i,f,o,g]=Wh*h(t-1)+Wx*x
  <7> grad_i = (grad_i_t) dot (sigmoid_grad(i_t))
  <8> grad_f = (grad_f_t) dot (sigmoid_grad(f_t))
  <9> grad_o = (grad_o_t) dot (tanh_grad(o_t))
  <10> grad_g = (grad_g_t) dot (sigmoid_grad(g_t))

  eq4. Weight update step2
  calculate the grad_w(ifog) and grad_h_t_1
  <11> grad_w(ifog) = grad_(ifog) iply x_t.transform()
  <12> grad_h_t_1 =
  /*////////////////////////////////




}


////end of namespace lstm
}

//}
