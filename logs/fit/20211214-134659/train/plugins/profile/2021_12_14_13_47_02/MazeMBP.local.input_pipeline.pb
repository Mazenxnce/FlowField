	G>#@G>#@!G>#@	6?}0F??6?}0F??!6?}0F??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:G>#@:[@h=|??A {?T"@Y????&M??rEagerKernelExecute 0*	      l@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?.PR`??!B?AfO@)7?A`?о?1J?$I??J@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?~j?t???!۶m۶m5@)@?R??1?A?A5.@:Preprocessing2U
Iterator::Model::ParallelMapV2?A`??"??!?$I?$?@)?A`??"??1?$I?$?@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??u????!?W|?WL@)??? ??1??????@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???x軋?!????.@)j4????1????@:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[10]::ConcatenateL7?A`???!n۶m?v@)Ę??Rx??1??????@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch?,{?|?!fffff&	@)?,{?|?1fffff&	@:Preprocessing2F
Iterator::ModelGW#???!PuPU @)
?????t?1?A?@:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[11]::Concatenate{?G?zt?!?m۶m?@)?,D??q?1PuP???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?(??/??!_?_??Q@)??2R??l?1?+??+???:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??R`?!?W|?W???)??R`?1?W|?W???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?I+?V?!I?$I????)?I+?V?1I?$I????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[11]::Concatenate[1]::FromTensor?'eRC;?!X|?W|???)?'eRC;?1X|?W|???:Preprocessing2?
PIterator::Model::ParallelMapV2::Zip[0]::FlatMap[11]::Concatenate[0]::TensorSlice????Mb0?!$I?$I???)????Mb0?1$I?$I???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[10]::Concatenate[1]::FromTensor?'eRC+?!X|?W|ŷ?)?'eRC+?1X|?W|ŷ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no95?}0F??I/	>???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	:[@h=|??:[@h=|??!:[@h=|??      ?!       "      ?!       *      ?!       2	 {?T"@ {?T"@! {?T"@:      ?!       B      ?!       J	????&M??????&M??!????&M??R      ?!       Z	????&M??????&M??!????&M??b      ?!       JCPU_ONLYY5?}0F??b q/	>???X@