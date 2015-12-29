#/************************************************************
#*
#* Licensed to the Apache Software Foundation (ASF) under one
#* or more contributor license agreements.  See the NOTICE file
#* distributed with this work for additional information
#* regarding copyright ownership.  The ASF licenses this file
#* to you under the Apache License, Version 2.0 (the
#* "License"); you may not use this file except in compliance
#* with the License.  You may obtain a copy of the License at
#*
#*   http://www.apache.org/licenses/LICENSE-2.0
#*
#* Unless required by applicable law or agreed to in writing,
#* software distributed under the License is distributed on an
#* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#* KIND, either express or implied.  See the License for the
#* specific language governing permissions and limitations
#* under the License.
#*
#*************************************************************/

MSHADOW_FLAGS :=-DMSHADOW_USE_CUDA=0 -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0

libs :=singa glog protobuf
filename = rnnlm-0.4b.tgz
# note: filelink for rnnlm-0.4b may change
filelink = https://f25ea9ccb7d3346ce6891573d543960492b92c30.googledrive.com/host/0ByxdPXuxLPS5RFM5dVNvWVhTd0U
dirname = $(patsubst %.tgz,%, $(filename))
numclass = 100
dirshards = train_shard valid_shard test_shard



lstm:
	protoc --proto_path=../../src/proto --proto_path=. --cpp_out=. lstm.proto
	$(CXX) -g main.cc lstm.cc lstm.pb.cc $(MSHADOW_FLAGS) -msse3 -std=c++11 -lsinga -lglog -lprotobuf -lopenblas -I../../include -I../../include/singa/proto \
		-L../../.libs/ -L/usr/local  -Wl,-unresolved-symbols=ignore-in-shared-libs -Wl,-rpath=../../.libs/\
		-o lstm.bin


download:
	wget $(filelink)/$(filename)
	tar zxf $(filename)
	rm $(filename)

create:
	protoc --proto_path=../../src/proto --proto_path=. --cpp_out=. rnnlm.proto
	$(CXX) create_data.cc rnnlm.pb.cc -std=c++11 -lsinga -lprotobuf -lzookeeper_mt -lglog -I../../include -I../../include/singa/proto \
		-L../../.libs/ -L/usr/local/lib -Wl,-unresolved-symbols=ignore-in-shared-libs -Wl,-rpath=../../.libs/ \
		-o create_data.bin
	for d in $(dirshards); do mkdir -p $${d}; done
	./create_data.bin -train $(dirname)/train -test $(dirname)/test -valid $(dirname)/valid -class_size $(numclass)

gru:
			protoc --proto_path=../../src/proto --proto_path=. --cpp_out=. lstm.proto
			$(CXX) main.cc gru.cc lstm.pb.cc $(MSHADOW_FLAGS) -msse3 -std=c++11 -lsinga -lglog -lprotobuf -lopenblas -I../../include -I../../include/singa/proto \
				-L../../.libs/ -L/usr/local  -Wl,-unresolved-symbols=ignore-in-shared-libs -Wl,-rpath=../../.libs/\
				-o lstm.bin

lstmnew:
		protoc --proto_path=../../src/proto --proto_path=. --cpp_out=. lstm.proto
		$(CXX) main.cc lstm-newmathop.cc lstm.pb.cc $(MSHADOW_FLAGS) -msse3 -std=c++11 -lsinga -lglog -lprotobuf -lopenblas -I../../include -I../../include/singa/proto \
			-L../../.libs/ -L/usr/local  -Wl,-unresolved-symbols=ignore-in-shared-libs -Wl,-rpath=../../.libs/\
			-o lstm.bin
