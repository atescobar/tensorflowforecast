Ú
´
:
Add
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
ś
AsString

input"T

output"
Ttype:
	2	
"
	precisionint˙˙˙˙˙˙˙˙˙"

scientificbool( "
shortestbool( "
widthint˙˙˙˙˙˙˙˙˙"
fillstring 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
B
Equal
x"T
y"T
z
"
Ttype:
2	

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
ď
ParseExample

serialized	
names
sparse_keys*Nsparse

dense_keys*Ndense
dense_defaults2Tdense
sparse_indices	*Nsparse
sparse_values2sparse_types
sparse_shapes	*Nsparse
dense_values2Tdense"
Nsparseint("
Ndenseint("%
sparse_types
list(type)(:
2	"
Tdense
list(type)(:
2	"
dense_shapeslist(shape)(
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
D
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.8.02v1.8.0-0-g93bc2e2072¤

global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 

global_step
VariableV2*
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step*
	container *
shape: 
˛
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step
j
global_step/readIdentityglobal_step*
_output_shapes
: *
T0	*
_class
loc:@global_step
o
input_example_tensorPlaceholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
U
ParseExample/ConstConst*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_1Const*
dtype0*
_output_shapes
: *
valueB 
W
ParseExample/Const_2Const*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_3Const*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_4Const*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_5Const*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_6Const*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_7Const*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_8Const*
valueB *
dtype0*
_output_shapes
: 
W
ParseExample/Const_9Const*
valueB *
dtype0*
_output_shapes
: 
X
ParseExample/Const_10Const*
valueB *
dtype0*
_output_shapes
: 
X
ParseExample/Const_11Const*
valueB *
dtype0*
_output_shapes
: 
X
ParseExample/Const_12Const*
valueB *
dtype0*
_output_shapes
: 
X
ParseExample/Const_13Const*
dtype0*
_output_shapes
: *
valueB 
X
ParseExample/Const_14Const*
valueB *
dtype0*
_output_shapes
: 
X
ParseExample/Const_15Const*
valueB *
dtype0*
_output_shapes
: 
X
ParseExample/Const_16Const*
valueB *
dtype0*
_output_shapes
: 
X
ParseExample/Const_17Const*
valueB *
dtype0*
_output_shapes
: 
X
ParseExample/Const_18Const*
valueB *
dtype0*
_output_shapes
: 
X
ParseExample/Const_19Const*
valueB *
dtype0*
_output_shapes
: 
X
ParseExample/Const_20Const*
valueB *
dtype0*
_output_shapes
: 
X
ParseExample/Const_21Const*
dtype0*
_output_shapes
: *
valueB 
X
ParseExample/Const_22Const*
valueB *
dtype0*
_output_shapes
: 
X
ParseExample/Const_23Const*
valueB *
dtype0*
_output_shapes
: 
X
ParseExample/Const_24Const*
valueB *
dtype0*
_output_shapes
: 
X
ParseExample/Const_25Const*
dtype0*
_output_shapes
: *
valueB 
X
ParseExample/Const_26Const*
valueB *
dtype0*
_output_shapes
: 
b
ParseExample/ParseExample/namesConst*
valueB *
dtype0*
_output_shapes
: 
y
&ParseExample/ParseExample/dense_keys_0Const*#
valueB Betapa_Calificacion*
dtype0*
_output_shapes
: 
w
&ParseExample/ParseExample/dense_keys_1Const*!
valueB Betapa_Desarrollo*
dtype0*
_output_shapes
: 
x
&ParseExample/ParseExample/dense_keys_2Const*"
valueB Betapa_Negociacion*
dtype0*
_output_shapes
: 
x
&ParseExample/ParseExample/dense_keys_3Const*"
valueB Betapa_Prospeccion*
dtype0*
_output_shapes
: 
s
&ParseExample/ParseExample/dense_keys_4Const*
valueB Betapa_Prueba*
dtype0*
_output_shapes
: 
u
&ParseExample/ParseExample/dense_keys_5Const*
valueB Betapa_Solucion*
dtype0*
_output_shapes
: 

&ParseExample/ParseExample/dense_keys_6Const*?
value6B4 B.modalidad_4445B64B-338B-E611-80F0-2C59E53A5504*
dtype0*
_output_shapes
: 

&ParseExample/ParseExample/dense_keys_7Const*?
value6B4 B.modalidad_4645B64B-338B-E611-80F0-2C59E53A5504*
dtype0*
_output_shapes
: 

&ParseExample/ParseExample/dense_keys_8Const*?
value6B4 B.modalidad_4845B64B-338B-E611-80F0-2C59E53A5504*
dtype0*
_output_shapes
: 

&ParseExample/ParseExample/dense_keys_9Const*
dtype0*
_output_shapes
: *?
value6B4 B.modalidad_4A45B64B-338B-E611-80F0-2C59E53A5504

'ParseExample/ParseExample/dense_keys_10Const*?
value6B4 B.modalidad_59C2334F-CDF2-E611-80FF-C4346BB56EE0*
dtype0*
_output_shapes
: 

'ParseExample/ParseExample/dense_keys_11Const*0
value'B% Bnaturaleza_AmpliacionoUpselling*
dtype0*
_output_shapes
: 
x
'ParseExample/ParseExample/dense_keys_12Const*!
valueB Bnaturaleza_Nuevo*
dtype0*
_output_shapes
: 

'ParseExample/ParseExample/dense_keys_13Const*-
value$B" Bnaturaleza_NuevoCrossSelling*
dtype0*
_output_shapes
: 
z
'ParseExample/ParseExample/dense_keys_14Const*
dtype0*
_output_shapes
: *#
valueB Bnaturaleza_Paragua
{
'ParseExample/ParseExample/dense_keys_15Const*$
valueB Bnaturaleza_Reajuste*
dtype0*
_output_shapes
: 

'ParseExample/ParseExample/dense_keys_16Const*3
value*B( B"naturaleza_Reconocimientodeingreso*
dtype0*
_output_shapes
: 

'ParseExample/ParseExample/dense_keys_17Const*)
value B Bnaturaleza_Renegociacion*
dtype0*
_output_shapes
: 
}
'ParseExample/ParseExample/dense_keys_18Const*
dtype0*
_output_shapes
: *&
valueB Bnaturaleza_Renovacion

'ParseExample/ParseExample/dense_keys_19Const*1
value(B& B naturaleza_Renovacioncompetitiva*
dtype0*
_output_shapes
: 
w
'ParseExample/ParseExample/dense_keys_20Const*
dtype0*
_output_shapes
: * 
valueB Bperiod_1quarter
w
'ParseExample/ParseExample/dense_keys_21Const* 
valueB Bperiod_2quarter*
dtype0*
_output_shapes
: 
w
'ParseExample/ParseExample/dense_keys_22Const* 
valueB Bperiod_3quarter*
dtype0*
_output_shapes
: 
w
'ParseExample/ParseExample/dense_keys_23Const*
dtype0*
_output_shapes
: * 
valueB Bperiod_4quarter
x
'ParseExample/ParseExample/dense_keys_24Const*!
valueB Bperiod_overayear*
dtype0*
_output_shapes
: 
q
'ParseExample/ParseExample/dense_keys_25Const*
dtype0*
_output_shapes
: *
valueB B	relevante
x
'ParseExample/ParseExample/dense_keys_26Const*!
valueB Btotalamount_base*
dtype0*
_output_shapes
: 
š
ParseExample/ParseExampleParseExampleinput_example_tensorParseExample/ParseExample/names&ParseExample/ParseExample/dense_keys_0&ParseExample/ParseExample/dense_keys_1&ParseExample/ParseExample/dense_keys_2&ParseExample/ParseExample/dense_keys_3&ParseExample/ParseExample/dense_keys_4&ParseExample/ParseExample/dense_keys_5&ParseExample/ParseExample/dense_keys_6&ParseExample/ParseExample/dense_keys_7&ParseExample/ParseExample/dense_keys_8&ParseExample/ParseExample/dense_keys_9'ParseExample/ParseExample/dense_keys_10'ParseExample/ParseExample/dense_keys_11'ParseExample/ParseExample/dense_keys_12'ParseExample/ParseExample/dense_keys_13'ParseExample/ParseExample/dense_keys_14'ParseExample/ParseExample/dense_keys_15'ParseExample/ParseExample/dense_keys_16'ParseExample/ParseExample/dense_keys_17'ParseExample/ParseExample/dense_keys_18'ParseExample/ParseExample/dense_keys_19'ParseExample/ParseExample/dense_keys_20'ParseExample/ParseExample/dense_keys_21'ParseExample/ParseExample/dense_keys_22'ParseExample/ParseExample/dense_keys_23'ParseExample/ParseExample/dense_keys_24'ParseExample/ParseExample/dense_keys_25'ParseExample/ParseExample/dense_keys_26ParseExample/ConstParseExample/Const_1ParseExample/Const_2ParseExample/Const_3ParseExample/Const_4ParseExample/Const_5ParseExample/Const_6ParseExample/Const_7ParseExample/Const_8ParseExample/Const_9ParseExample/Const_10ParseExample/Const_11ParseExample/Const_12ParseExample/Const_13ParseExample/Const_14ParseExample/Const_15ParseExample/Const_16ParseExample/Const_17ParseExample/Const_18ParseExample/Const_19ParseExample/Const_20ParseExample/Const_21ParseExample/Const_22ParseExample/Const_23ParseExample/Const_24ParseExample/Const_25ParseExample/Const_26*
Nsparse *ś
dense_shapesĽ
˘:::::::::::::::::::::::::::*
sparse_types
 *)
Tdense
2*
Ndense*
_output_shapes
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Cdnn/input_from_feature_columns/input_layer/etapa_Calificacion/ShapeShapeParseExample/ParseExample*
T0*
out_type0*
_output_shapes
:

Qdnn/input_from_feature_columns/input_layer/etapa_Calificacion/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Sdnn/input_from_feature_columns/input_layer/etapa_Calificacion/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Sdnn/input_from_feature_columns/input_layer/etapa_Calificacion/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ż
Kdnn/input_from_feature_columns/input_layer/etapa_Calificacion/strided_sliceStridedSliceCdnn/input_from_feature_columns/input_layer/etapa_Calificacion/ShapeQdnn/input_from_feature_columns/input_layer/etapa_Calificacion/strided_slice/stackSdnn/input_from_feature_columns/input_layer/etapa_Calificacion/strided_slice/stack_1Sdnn/input_from_feature_columns/input_layer/etapa_Calificacion/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

Mdnn/input_from_feature_columns/input_layer/etapa_Calificacion/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Š
Kdnn/input_from_feature_columns/input_layer/etapa_Calificacion/Reshape/shapePackKdnn/input_from_feature_columns/input_layer/etapa_Calificacion/strided_sliceMdnn/input_from_feature_columns/input_layer/etapa_Calificacion/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
ř
Ednn/input_from_feature_columns/input_layer/etapa_Calificacion/ReshapeReshapeParseExample/ParseExampleKdnn/input_from_feature_columns/input_layer/etapa_Calificacion/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Adnn/input_from_feature_columns/input_layer/etapa_Desarrollo/ShapeShapeParseExample/ParseExample:1*
T0*
out_type0*
_output_shapes
:

Odnn/input_from_feature_columns/input_layer/etapa_Desarrollo/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Qdnn/input_from_feature_columns/input_layer/etapa_Desarrollo/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

Qdnn/input_from_feature_columns/input_layer/etapa_Desarrollo/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ľ
Idnn/input_from_feature_columns/input_layer/etapa_Desarrollo/strided_sliceStridedSliceAdnn/input_from_feature_columns/input_layer/etapa_Desarrollo/ShapeOdnn/input_from_feature_columns/input_layer/etapa_Desarrollo/strided_slice/stackQdnn/input_from_feature_columns/input_layer/etapa_Desarrollo/strided_slice/stack_1Qdnn/input_from_feature_columns/input_layer/etapa_Desarrollo/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

Kdnn/input_from_feature_columns/input_layer/etapa_Desarrollo/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ł
Idnn/input_from_feature_columns/input_layer/etapa_Desarrollo/Reshape/shapePackIdnn/input_from_feature_columns/input_layer/etapa_Desarrollo/strided_sliceKdnn/input_from_feature_columns/input_layer/etapa_Desarrollo/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
ö
Cdnn/input_from_feature_columns/input_layer/etapa_Desarrollo/ReshapeReshapeParseExample/ParseExample:1Idnn/input_from_feature_columns/input_layer/etapa_Desarrollo/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Bdnn/input_from_feature_columns/input_layer/etapa_Negociacion/ShapeShapeParseExample/ParseExample:2*
_output_shapes
:*
T0*
out_type0

Pdnn/input_from_feature_columns/input_layer/etapa_Negociacion/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

Rdnn/input_from_feature_columns/input_layer/etapa_Negociacion/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Rdnn/input_from_feature_columns/input_layer/etapa_Negociacion/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ş
Jdnn/input_from_feature_columns/input_layer/etapa_Negociacion/strided_sliceStridedSliceBdnn/input_from_feature_columns/input_layer/etapa_Negociacion/ShapePdnn/input_from_feature_columns/input_layer/etapa_Negociacion/strided_slice/stackRdnn/input_from_feature_columns/input_layer/etapa_Negociacion/strided_slice/stack_1Rdnn/input_from_feature_columns/input_layer/etapa_Negociacion/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0

Ldnn/input_from_feature_columns/input_layer/etapa_Negociacion/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ś
Jdnn/input_from_feature_columns/input_layer/etapa_Negociacion/Reshape/shapePackJdnn/input_from_feature_columns/input_layer/etapa_Negociacion/strided_sliceLdnn/input_from_feature_columns/input_layer/etapa_Negociacion/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
ř
Ddnn/input_from_feature_columns/input_layer/etapa_Negociacion/ReshapeReshapeParseExample/ParseExample:2Jdnn/input_from_feature_columns/input_layer/etapa_Negociacion/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

Bdnn/input_from_feature_columns/input_layer/etapa_Prospeccion/ShapeShapeParseExample/ParseExample:3*
T0*
out_type0*
_output_shapes
:

Pdnn/input_from_feature_columns/input_layer/etapa_Prospeccion/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Rdnn/input_from_feature_columns/input_layer/etapa_Prospeccion/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Rdnn/input_from_feature_columns/input_layer/etapa_Prospeccion/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ş
Jdnn/input_from_feature_columns/input_layer/etapa_Prospeccion/strided_sliceStridedSliceBdnn/input_from_feature_columns/input_layer/etapa_Prospeccion/ShapePdnn/input_from_feature_columns/input_layer/etapa_Prospeccion/strided_slice/stackRdnn/input_from_feature_columns/input_layer/etapa_Prospeccion/strided_slice/stack_1Rdnn/input_from_feature_columns/input_layer/etapa_Prospeccion/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

Ldnn/input_from_feature_columns/input_layer/etapa_Prospeccion/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ś
Jdnn/input_from_feature_columns/input_layer/etapa_Prospeccion/Reshape/shapePackJdnn/input_from_feature_columns/input_layer/etapa_Prospeccion/strided_sliceLdnn/input_from_feature_columns/input_layer/etapa_Prospeccion/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
ř
Ddnn/input_from_feature_columns/input_layer/etapa_Prospeccion/ReshapeReshapeParseExample/ParseExample:3Jdnn/input_from_feature_columns/input_layer/etapa_Prospeccion/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

=dnn/input_from_feature_columns/input_layer/etapa_Prueba/ShapeShapeParseExample/ParseExample:4*
T0*
out_type0*
_output_shapes
:

Kdnn/input_from_feature_columns/input_layer/etapa_Prueba/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Mdnn/input_from_feature_columns/input_layer/etapa_Prueba/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Mdnn/input_from_feature_columns/input_layer/etapa_Prueba/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

Ednn/input_from_feature_columns/input_layer/etapa_Prueba/strided_sliceStridedSlice=dnn/input_from_feature_columns/input_layer/etapa_Prueba/ShapeKdnn/input_from_feature_columns/input_layer/etapa_Prueba/strided_slice/stackMdnn/input_from_feature_columns/input_layer/etapa_Prueba/strided_slice/stack_1Mdnn/input_from_feature_columns/input_layer/etapa_Prueba/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

Gdnn/input_from_feature_columns/input_layer/etapa_Prueba/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Ednn/input_from_feature_columns/input_layer/etapa_Prueba/Reshape/shapePackEdnn/input_from_feature_columns/input_layer/etapa_Prueba/strided_sliceGdnn/input_from_feature_columns/input_layer/etapa_Prueba/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
î
?dnn/input_from_feature_columns/input_layer/etapa_Prueba/ReshapeReshapeParseExample/ParseExample:4Ednn/input_from_feature_columns/input_layer/etapa_Prueba/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

?dnn/input_from_feature_columns/input_layer/etapa_Solucion/ShapeShapeParseExample/ParseExample:5*
T0*
out_type0*
_output_shapes
:

Mdnn/input_from_feature_columns/input_layer/etapa_Solucion/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Odnn/input_from_feature_columns/input_layer/etapa_Solucion/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Odnn/input_from_feature_columns/input_layer/etapa_Solucion/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

Gdnn/input_from_feature_columns/input_layer/etapa_Solucion/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/etapa_Solucion/ShapeMdnn/input_from_feature_columns/input_layer/etapa_Solucion/strided_slice/stackOdnn/input_from_feature_columns/input_layer/etapa_Solucion/strided_slice/stack_1Odnn/input_from_feature_columns/input_layer/etapa_Solucion/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask

Idnn/input_from_feature_columns/input_layer/etapa_Solucion/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :

Gdnn/input_from_feature_columns/input_layer/etapa_Solucion/Reshape/shapePackGdnn/input_from_feature_columns/input_layer/etapa_Solucion/strided_sliceIdnn/input_from_feature_columns/input_layer/etapa_Solucion/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
ň
Adnn/input_from_feature_columns/input_layer/etapa_Solucion/ReshapeReshapeParseExample/ParseExample:5Gdnn/input_from_feature_columns/input_layer/etapa_Solucion/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
_dnn/input_from_feature_columns/input_layer/modalidad_4445B64B-338B-E611-80F0-2C59E53A5504/ShapeShapeParseExample/ParseExample:6*
_output_shapes
:*
T0*
out_type0
ˇ
mdnn/input_from_feature_columns/input_layer/modalidad_4445B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
š
odnn/input_from_feature_columns/input_layer/modalidad_4445B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
š
odnn/input_from_feature_columns/input_layer/modalidad_4445B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ť
gdnn/input_from_feature_columns/input_layer/modalidad_4445B64B-338B-E611-80F0-2C59E53A5504/strided_sliceStridedSlice_dnn/input_from_feature_columns/input_layer/modalidad_4445B64B-338B-E611-80F0-2C59E53A5504/Shapemdnn/input_from_feature_columns/input_layer/modalidad_4445B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stackodnn/input_from_feature_columns/input_layer/modalidad_4445B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stack_1odnn/input_from_feature_columns/input_layer/modalidad_4445B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
Ť
idnn/input_from_feature_columns/input_layer/modalidad_4445B64B-338B-E611-80F0-2C59E53A5504/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
ý
gdnn/input_from_feature_columns/input_layer/modalidad_4445B64B-338B-E611-80F0-2C59E53A5504/Reshape/shapePackgdnn/input_from_feature_columns/input_layer/modalidad_4445B64B-338B-E611-80F0-2C59E53A5504/strided_sliceidnn/input_from_feature_columns/input_layer/modalidad_4445B64B-338B-E611-80F0-2C59E53A5504/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
˛
adnn/input_from_feature_columns/input_layer/modalidad_4445B64B-338B-E611-80F0-2C59E53A5504/ReshapeReshapeParseExample/ParseExample:6gdnn/input_from_feature_columns/input_layer/modalidad_4445B64B-338B-E611-80F0-2C59E53A5504/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ş
_dnn/input_from_feature_columns/input_layer/modalidad_4645B64B-338B-E611-80F0-2C59E53A5504/ShapeShapeParseExample/ParseExample:7*
T0*
out_type0*
_output_shapes
:
ˇ
mdnn/input_from_feature_columns/input_layer/modalidad_4645B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
š
odnn/input_from_feature_columns/input_layer/modalidad_4645B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
š
odnn/input_from_feature_columns/input_layer/modalidad_4645B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ť
gdnn/input_from_feature_columns/input_layer/modalidad_4645B64B-338B-E611-80F0-2C59E53A5504/strided_sliceStridedSlice_dnn/input_from_feature_columns/input_layer/modalidad_4645B64B-338B-E611-80F0-2C59E53A5504/Shapemdnn/input_from_feature_columns/input_layer/modalidad_4645B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stackodnn/input_from_feature_columns/input_layer/modalidad_4645B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stack_1odnn/input_from_feature_columns/input_layer/modalidad_4645B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
Ť
idnn/input_from_feature_columns/input_layer/modalidad_4645B64B-338B-E611-80F0-2C59E53A5504/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
ý
gdnn/input_from_feature_columns/input_layer/modalidad_4645B64B-338B-E611-80F0-2C59E53A5504/Reshape/shapePackgdnn/input_from_feature_columns/input_layer/modalidad_4645B64B-338B-E611-80F0-2C59E53A5504/strided_sliceidnn/input_from_feature_columns/input_layer/modalidad_4645B64B-338B-E611-80F0-2C59E53A5504/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 
˛
adnn/input_from_feature_columns/input_layer/modalidad_4645B64B-338B-E611-80F0-2C59E53A5504/ReshapeReshapeParseExample/ParseExample:7gdnn/input_from_feature_columns/input_layer/modalidad_4645B64B-338B-E611-80F0-2C59E53A5504/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
_dnn/input_from_feature_columns/input_layer/modalidad_4845B64B-338B-E611-80F0-2C59E53A5504/ShapeShapeParseExample/ParseExample:8*
T0*
out_type0*
_output_shapes
:
ˇ
mdnn/input_from_feature_columns/input_layer/modalidad_4845B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
š
odnn/input_from_feature_columns/input_layer/modalidad_4845B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
š
odnn/input_from_feature_columns/input_layer/modalidad_4845B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ť
gdnn/input_from_feature_columns/input_layer/modalidad_4845B64B-338B-E611-80F0-2C59E53A5504/strided_sliceStridedSlice_dnn/input_from_feature_columns/input_layer/modalidad_4845B64B-338B-E611-80F0-2C59E53A5504/Shapemdnn/input_from_feature_columns/input_layer/modalidad_4845B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stackodnn/input_from_feature_columns/input_layer/modalidad_4845B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stack_1odnn/input_from_feature_columns/input_layer/modalidad_4845B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Ť
idnn/input_from_feature_columns/input_layer/modalidad_4845B64B-338B-E611-80F0-2C59E53A5504/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
ý
gdnn/input_from_feature_columns/input_layer/modalidad_4845B64B-338B-E611-80F0-2C59E53A5504/Reshape/shapePackgdnn/input_from_feature_columns/input_layer/modalidad_4845B64B-338B-E611-80F0-2C59E53A5504/strided_sliceidnn/input_from_feature_columns/input_layer/modalidad_4845B64B-338B-E611-80F0-2C59E53A5504/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
˛
adnn/input_from_feature_columns/input_layer/modalidad_4845B64B-338B-E611-80F0-2C59E53A5504/ReshapeReshapeParseExample/ParseExample:8gdnn/input_from_feature_columns/input_layer/modalidad_4845B64B-338B-E611-80F0-2C59E53A5504/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
_dnn/input_from_feature_columns/input_layer/modalidad_4A45B64B-338B-E611-80F0-2C59E53A5504/ShapeShapeParseExample/ParseExample:9*
T0*
out_type0*
_output_shapes
:
ˇ
mdnn/input_from_feature_columns/input_layer/modalidad_4A45B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
š
odnn/input_from_feature_columns/input_layer/modalidad_4A45B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
š
odnn/input_from_feature_columns/input_layer/modalidad_4A45B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ť
gdnn/input_from_feature_columns/input_layer/modalidad_4A45B64B-338B-E611-80F0-2C59E53A5504/strided_sliceStridedSlice_dnn/input_from_feature_columns/input_layer/modalidad_4A45B64B-338B-E611-80F0-2C59E53A5504/Shapemdnn/input_from_feature_columns/input_layer/modalidad_4A45B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stackodnn/input_from_feature_columns/input_layer/modalidad_4A45B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stack_1odnn/input_from_feature_columns/input_layer/modalidad_4A45B64B-338B-E611-80F0-2C59E53A5504/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Ť
idnn/input_from_feature_columns/input_layer/modalidad_4A45B64B-338B-E611-80F0-2C59E53A5504/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
ý
gdnn/input_from_feature_columns/input_layer/modalidad_4A45B64B-338B-E611-80F0-2C59E53A5504/Reshape/shapePackgdnn/input_from_feature_columns/input_layer/modalidad_4A45B64B-338B-E611-80F0-2C59E53A5504/strided_sliceidnn/input_from_feature_columns/input_layer/modalidad_4A45B64B-338B-E611-80F0-2C59E53A5504/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 
˛
adnn/input_from_feature_columns/input_layer/modalidad_4A45B64B-338B-E611-80F0-2C59E53A5504/ReshapeReshapeParseExample/ParseExample:9gdnn/input_from_feature_columns/input_layer/modalidad_4A45B64B-338B-E611-80F0-2C59E53A5504/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
_dnn/input_from_feature_columns/input_layer/modalidad_59C2334F-CDF2-E611-80FF-C4346BB56EE0/ShapeShapeParseExample/ParseExample:10*
T0*
out_type0*
_output_shapes
:
ˇ
mdnn/input_from_feature_columns/input_layer/modalidad_59C2334F-CDF2-E611-80FF-C4346BB56EE0/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
š
odnn/input_from_feature_columns/input_layer/modalidad_59C2334F-CDF2-E611-80FF-C4346BB56EE0/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
š
odnn/input_from_feature_columns/input_layer/modalidad_59C2334F-CDF2-E611-80FF-C4346BB56EE0/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ť
gdnn/input_from_feature_columns/input_layer/modalidad_59C2334F-CDF2-E611-80FF-C4346BB56EE0/strided_sliceStridedSlice_dnn/input_from_feature_columns/input_layer/modalidad_59C2334F-CDF2-E611-80FF-C4346BB56EE0/Shapemdnn/input_from_feature_columns/input_layer/modalidad_59C2334F-CDF2-E611-80FF-C4346BB56EE0/strided_slice/stackodnn/input_from_feature_columns/input_layer/modalidad_59C2334F-CDF2-E611-80FF-C4346BB56EE0/strided_slice/stack_1odnn/input_from_feature_columns/input_layer/modalidad_59C2334F-CDF2-E611-80FF-C4346BB56EE0/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
Ť
idnn/input_from_feature_columns/input_layer/modalidad_59C2334F-CDF2-E611-80FF-C4346BB56EE0/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
ý
gdnn/input_from_feature_columns/input_layer/modalidad_59C2334F-CDF2-E611-80FF-C4346BB56EE0/Reshape/shapePackgdnn/input_from_feature_columns/input_layer/modalidad_59C2334F-CDF2-E611-80FF-C4346BB56EE0/strided_sliceidnn/input_from_feature_columns/input_layer/modalidad_59C2334F-CDF2-E611-80FF-C4346BB56EE0/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 
ł
adnn/input_from_feature_columns/input_layer/modalidad_59C2334F-CDF2-E611-80FF-C4346BB56EE0/ReshapeReshapeParseExample/ParseExample:10gdnn/input_from_feature_columns/input_layer/modalidad_59C2334F-CDF2-E611-80FF-C4346BB56EE0/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
Pdnn/input_from_feature_columns/input_layer/naturaleza_AmpliacionoUpselling/ShapeShapeParseExample/ParseExample:11*
T0*
out_type0*
_output_shapes
:
¨
^dnn/input_from_feature_columns/input_layer/naturaleza_AmpliacionoUpselling/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ş
`dnn/input_from_feature_columns/input_layer/naturaleza_AmpliacionoUpselling/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ş
`dnn/input_from_feature_columns/input_layer/naturaleza_AmpliacionoUpselling/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
đ
Xdnn/input_from_feature_columns/input_layer/naturaleza_AmpliacionoUpselling/strided_sliceStridedSlicePdnn/input_from_feature_columns/input_layer/naturaleza_AmpliacionoUpselling/Shape^dnn/input_from_feature_columns/input_layer/naturaleza_AmpliacionoUpselling/strided_slice/stack`dnn/input_from_feature_columns/input_layer/naturaleza_AmpliacionoUpselling/strided_slice/stack_1`dnn/input_from_feature_columns/input_layer/naturaleza_AmpliacionoUpselling/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0

Zdnn/input_from_feature_columns/input_layer/naturaleza_AmpliacionoUpselling/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Đ
Xdnn/input_from_feature_columns/input_layer/naturaleza_AmpliacionoUpselling/Reshape/shapePackXdnn/input_from_feature_columns/input_layer/naturaleza_AmpliacionoUpselling/strided_sliceZdnn/input_from_feature_columns/input_layer/naturaleza_AmpliacionoUpselling/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 

Rdnn/input_from_feature_columns/input_layer/naturaleza_AmpliacionoUpselling/ReshapeReshapeParseExample/ParseExample:11Xdnn/input_from_feature_columns/input_layer/naturaleza_AmpliacionoUpselling/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Adnn/input_from_feature_columns/input_layer/naturaleza_Nuevo/ShapeShapeParseExample/ParseExample:12*
_output_shapes
:*
T0*
out_type0

Odnn/input_from_feature_columns/input_layer/naturaleza_Nuevo/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

Qdnn/input_from_feature_columns/input_layer/naturaleza_Nuevo/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Qdnn/input_from_feature_columns/input_layer/naturaleza_Nuevo/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ľ
Idnn/input_from_feature_columns/input_layer/naturaleza_Nuevo/strided_sliceStridedSliceAdnn/input_from_feature_columns/input_layer/naturaleza_Nuevo/ShapeOdnn/input_from_feature_columns/input_layer/naturaleza_Nuevo/strided_slice/stackQdnn/input_from_feature_columns/input_layer/naturaleza_Nuevo/strided_slice/stack_1Qdnn/input_from_feature_columns/input_layer/naturaleza_Nuevo/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

Kdnn/input_from_feature_columns/input_layer/naturaleza_Nuevo/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ł
Idnn/input_from_feature_columns/input_layer/naturaleza_Nuevo/Reshape/shapePackIdnn/input_from_feature_columns/input_layer/naturaleza_Nuevo/strided_sliceKdnn/input_from_feature_columns/input_layer/naturaleza_Nuevo/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
÷
Cdnn/input_from_feature_columns/input_layer/naturaleza_Nuevo/ReshapeReshapeParseExample/ParseExample:12Idnn/input_from_feature_columns/input_layer/naturaleza_Nuevo/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
Mdnn/input_from_feature_columns/input_layer/naturaleza_NuevoCrossSelling/ShapeShapeParseExample/ParseExample:13*
T0*
out_type0*
_output_shapes
:
Ľ
[dnn/input_from_feature_columns/input_layer/naturaleza_NuevoCrossSelling/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
§
]dnn/input_from_feature_columns/input_layer/naturaleza_NuevoCrossSelling/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
§
]dnn/input_from_feature_columns/input_layer/naturaleza_NuevoCrossSelling/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
á
Udnn/input_from_feature_columns/input_layer/naturaleza_NuevoCrossSelling/strided_sliceStridedSliceMdnn/input_from_feature_columns/input_layer/naturaleza_NuevoCrossSelling/Shape[dnn/input_from_feature_columns/input_layer/naturaleza_NuevoCrossSelling/strided_slice/stack]dnn/input_from_feature_columns/input_layer/naturaleza_NuevoCrossSelling/strided_slice/stack_1]dnn/input_from_feature_columns/input_layer/naturaleza_NuevoCrossSelling/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0

Wdnn/input_from_feature_columns/input_layer/naturaleza_NuevoCrossSelling/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ç
Udnn/input_from_feature_columns/input_layer/naturaleza_NuevoCrossSelling/Reshape/shapePackUdnn/input_from_feature_columns/input_layer/naturaleza_NuevoCrossSelling/strided_sliceWdnn/input_from_feature_columns/input_layer/naturaleza_NuevoCrossSelling/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:

Odnn/input_from_feature_columns/input_layer/naturaleza_NuevoCrossSelling/ReshapeReshapeParseExample/ParseExample:13Udnn/input_from_feature_columns/input_layer/naturaleza_NuevoCrossSelling/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

Cdnn/input_from_feature_columns/input_layer/naturaleza_Paragua/ShapeShapeParseExample/ParseExample:14*
T0*
out_type0*
_output_shapes
:

Qdnn/input_from_feature_columns/input_layer/naturaleza_Paragua/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Sdnn/input_from_feature_columns/input_layer/naturaleza_Paragua/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Sdnn/input_from_feature_columns/input_layer/naturaleza_Paragua/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ż
Kdnn/input_from_feature_columns/input_layer/naturaleza_Paragua/strided_sliceStridedSliceCdnn/input_from_feature_columns/input_layer/naturaleza_Paragua/ShapeQdnn/input_from_feature_columns/input_layer/naturaleza_Paragua/strided_slice/stackSdnn/input_from_feature_columns/input_layer/naturaleza_Paragua/strided_slice/stack_1Sdnn/input_from_feature_columns/input_layer/naturaleza_Paragua/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0

Mdnn/input_from_feature_columns/input_layer/naturaleza_Paragua/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Š
Kdnn/input_from_feature_columns/input_layer/naturaleza_Paragua/Reshape/shapePackKdnn/input_from_feature_columns/input_layer/naturaleza_Paragua/strided_sliceMdnn/input_from_feature_columns/input_layer/naturaleza_Paragua/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
ű
Ednn/input_from_feature_columns/input_layer/naturaleza_Paragua/ReshapeReshapeParseExample/ParseExample:14Kdnn/input_from_feature_columns/input_layer/naturaleza_Paragua/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
Ddnn/input_from_feature_columns/input_layer/naturaleza_Reajuste/ShapeShapeParseExample/ParseExample:15*
T0*
out_type0*
_output_shapes
:

Rdnn/input_from_feature_columns/input_layer/naturaleza_Reajuste/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Tdnn/input_from_feature_columns/input_layer/naturaleza_Reajuste/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Tdnn/input_from_feature_columns/input_layer/naturaleza_Reajuste/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
´
Ldnn/input_from_feature_columns/input_layer/naturaleza_Reajuste/strided_sliceStridedSliceDdnn/input_from_feature_columns/input_layer/naturaleza_Reajuste/ShapeRdnn/input_from_feature_columns/input_layer/naturaleza_Reajuste/strided_slice/stackTdnn/input_from_feature_columns/input_layer/naturaleza_Reajuste/strided_slice/stack_1Tdnn/input_from_feature_columns/input_layer/naturaleza_Reajuste/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

Ndnn/input_from_feature_columns/input_layer/naturaleza_Reajuste/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ź
Ldnn/input_from_feature_columns/input_layer/naturaleza_Reajuste/Reshape/shapePackLdnn/input_from_feature_columns/input_layer/naturaleza_Reajuste/strided_sliceNdnn/input_from_feature_columns/input_layer/naturaleza_Reajuste/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
ý
Fdnn/input_from_feature_columns/input_layer/naturaleza_Reajuste/ReshapeReshapeParseExample/ParseExample:15Ldnn/input_from_feature_columns/input_layer/naturaleza_Reajuste/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
Sdnn/input_from_feature_columns/input_layer/naturaleza_Reconocimientodeingreso/ShapeShapeParseExample/ParseExample:16*
_output_shapes
:*
T0*
out_type0
Ť
adnn/input_from_feature_columns/input_layer/naturaleza_Reconocimientodeingreso/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
­
cdnn/input_from_feature_columns/input_layer/naturaleza_Reconocimientodeingreso/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
­
cdnn/input_from_feature_columns/input_layer/naturaleza_Reconocimientodeingreso/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
˙
[dnn/input_from_feature_columns/input_layer/naturaleza_Reconocimientodeingreso/strided_sliceStridedSliceSdnn/input_from_feature_columns/input_layer/naturaleza_Reconocimientodeingreso/Shapeadnn/input_from_feature_columns/input_layer/naturaleza_Reconocimientodeingreso/strided_slice/stackcdnn/input_from_feature_columns/input_layer/naturaleza_Reconocimientodeingreso/strided_slice/stack_1cdnn/input_from_feature_columns/input_layer/naturaleza_Reconocimientodeingreso/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0

]dnn/input_from_feature_columns/input_layer/naturaleza_Reconocimientodeingreso/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ů
[dnn/input_from_feature_columns/input_layer/naturaleza_Reconocimientodeingreso/Reshape/shapePack[dnn/input_from_feature_columns/input_layer/naturaleza_Reconocimientodeingreso/strided_slice]dnn/input_from_feature_columns/input_layer/naturaleza_Reconocimientodeingreso/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 

Udnn/input_from_feature_columns/input_layer/naturaleza_Reconocimientodeingreso/ReshapeReshapeParseExample/ParseExample:16[dnn/input_from_feature_columns/input_layer/naturaleza_Reconocimientodeingreso/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
Idnn/input_from_feature_columns/input_layer/naturaleza_Renegociacion/ShapeShapeParseExample/ParseExample:17*
T0*
out_type0*
_output_shapes
:
Ą
Wdnn/input_from_feature_columns/input_layer/naturaleza_Renegociacion/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ł
Ydnn/input_from_feature_columns/input_layer/naturaleza_Renegociacion/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ł
Ydnn/input_from_feature_columns/input_layer/naturaleza_Renegociacion/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Í
Qdnn/input_from_feature_columns/input_layer/naturaleza_Renegociacion/strided_sliceStridedSliceIdnn/input_from_feature_columns/input_layer/naturaleza_Renegociacion/ShapeWdnn/input_from_feature_columns/input_layer/naturaleza_Renegociacion/strided_slice/stackYdnn/input_from_feature_columns/input_layer/naturaleza_Renegociacion/strided_slice/stack_1Ydnn/input_from_feature_columns/input_layer/naturaleza_Renegociacion/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

Sdnn/input_from_feature_columns/input_layer/naturaleza_Renegociacion/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
ť
Qdnn/input_from_feature_columns/input_layer/naturaleza_Renegociacion/Reshape/shapePackQdnn/input_from_feature_columns/input_layer/naturaleza_Renegociacion/strided_sliceSdnn/input_from_feature_columns/input_layer/naturaleza_Renegociacion/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:

Kdnn/input_from_feature_columns/input_layer/naturaleza_Renegociacion/ReshapeReshapeParseExample/ParseExample:17Qdnn/input_from_feature_columns/input_layer/naturaleza_Renegociacion/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
Fdnn/input_from_feature_columns/input_layer/naturaleza_Renovacion/ShapeShapeParseExample/ParseExample:18*
T0*
out_type0*
_output_shapes
:

Tdnn/input_from_feature_columns/input_layer/naturaleza_Renovacion/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
 
Vdnn/input_from_feature_columns/input_layer/naturaleza_Renovacion/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
 
Vdnn/input_from_feature_columns/input_layer/naturaleza_Renovacion/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ž
Ndnn/input_from_feature_columns/input_layer/naturaleza_Renovacion/strided_sliceStridedSliceFdnn/input_from_feature_columns/input_layer/naturaleza_Renovacion/ShapeTdnn/input_from_feature_columns/input_layer/naturaleza_Renovacion/strided_slice/stackVdnn/input_from_feature_columns/input_layer/naturaleza_Renovacion/strided_slice/stack_1Vdnn/input_from_feature_columns/input_layer/naturaleza_Renovacion/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

Pdnn/input_from_feature_columns/input_layer/naturaleza_Renovacion/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
˛
Ndnn/input_from_feature_columns/input_layer/naturaleza_Renovacion/Reshape/shapePackNdnn/input_from_feature_columns/input_layer/naturaleza_Renovacion/strided_slicePdnn/input_from_feature_columns/input_layer/naturaleza_Renovacion/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:

Hdnn/input_from_feature_columns/input_layer/naturaleza_Renovacion/ReshapeReshapeParseExample/ParseExample:18Ndnn/input_from_feature_columns/input_layer/naturaleza_Renovacion/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
Qdnn/input_from_feature_columns/input_layer/naturaleza_Renovacioncompetitiva/ShapeShapeParseExample/ParseExample:19*
T0*
out_type0*
_output_shapes
:
Š
_dnn/input_from_feature_columns/input_layer/naturaleza_Renovacioncompetitiva/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ť
adnn/input_from_feature_columns/input_layer/naturaleza_Renovacioncompetitiva/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ť
adnn/input_from_feature_columns/input_layer/naturaleza_Renovacioncompetitiva/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ő
Ydnn/input_from_feature_columns/input_layer/naturaleza_Renovacioncompetitiva/strided_sliceStridedSliceQdnn/input_from_feature_columns/input_layer/naturaleza_Renovacioncompetitiva/Shape_dnn/input_from_feature_columns/input_layer/naturaleza_Renovacioncompetitiva/strided_slice/stackadnn/input_from_feature_columns/input_layer/naturaleza_Renovacioncompetitiva/strided_slice/stack_1adnn/input_from_feature_columns/input_layer/naturaleza_Renovacioncompetitiva/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0

[dnn/input_from_feature_columns/input_layer/naturaleza_Renovacioncompetitiva/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
Ó
Ydnn/input_from_feature_columns/input_layer/naturaleza_Renovacioncompetitiva/Reshape/shapePackYdnn/input_from_feature_columns/input_layer/naturaleza_Renovacioncompetitiva/strided_slice[dnn/input_from_feature_columns/input_layer/naturaleza_Renovacioncompetitiva/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 

Sdnn/input_from_feature_columns/input_layer/naturaleza_Renovacioncompetitiva/ReshapeReshapeParseExample/ParseExample:19Ydnn/input_from_feature_columns/input_layer/naturaleza_Renovacioncompetitiva/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

@dnn/input_from_feature_columns/input_layer/period_1quarter/ShapeShapeParseExample/ParseExample:20*
T0*
out_type0*
_output_shapes
:

Ndnn/input_from_feature_columns/input_layer/period_1quarter/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Pdnn/input_from_feature_columns/input_layer/period_1quarter/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Pdnn/input_from_feature_columns/input_layer/period_1quarter/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
 
Hdnn/input_from_feature_columns/input_layer/period_1quarter/strided_sliceStridedSlice@dnn/input_from_feature_columns/input_layer/period_1quarter/ShapeNdnn/input_from_feature_columns/input_layer/period_1quarter/strided_slice/stackPdnn/input_from_feature_columns/input_layer/period_1quarter/strided_slice/stack_1Pdnn/input_from_feature_columns/input_layer/period_1quarter/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

Jdnn/input_from_feature_columns/input_layer/period_1quarter/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
 
Hdnn/input_from_feature_columns/input_layer/period_1quarter/Reshape/shapePackHdnn/input_from_feature_columns/input_layer/period_1quarter/strided_sliceJdnn/input_from_feature_columns/input_layer/period_1quarter/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
ő
Bdnn/input_from_feature_columns/input_layer/period_1quarter/ReshapeReshapeParseExample/ParseExample:20Hdnn/input_from_feature_columns/input_layer/period_1quarter/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

@dnn/input_from_feature_columns/input_layer/period_2quarter/ShapeShapeParseExample/ParseExample:21*
T0*
out_type0*
_output_shapes
:

Ndnn/input_from_feature_columns/input_layer/period_2quarter/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Pdnn/input_from_feature_columns/input_layer/period_2quarter/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Pdnn/input_from_feature_columns/input_layer/period_2quarter/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
 
Hdnn/input_from_feature_columns/input_layer/period_2quarter/strided_sliceStridedSlice@dnn/input_from_feature_columns/input_layer/period_2quarter/ShapeNdnn/input_from_feature_columns/input_layer/period_2quarter/strided_slice/stackPdnn/input_from_feature_columns/input_layer/period_2quarter/strided_slice/stack_1Pdnn/input_from_feature_columns/input_layer/period_2quarter/strided_slice/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 

Jdnn/input_from_feature_columns/input_layer/period_2quarter/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
 
Hdnn/input_from_feature_columns/input_layer/period_2quarter/Reshape/shapePackHdnn/input_from_feature_columns/input_layer/period_2quarter/strided_sliceJdnn/input_from_feature_columns/input_layer/period_2quarter/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
ő
Bdnn/input_from_feature_columns/input_layer/period_2quarter/ReshapeReshapeParseExample/ParseExample:21Hdnn/input_from_feature_columns/input_layer/period_2quarter/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

@dnn/input_from_feature_columns/input_layer/period_3quarter/ShapeShapeParseExample/ParseExample:22*
T0*
out_type0*
_output_shapes
:

Ndnn/input_from_feature_columns/input_layer/period_3quarter/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Pdnn/input_from_feature_columns/input_layer/period_3quarter/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Pdnn/input_from_feature_columns/input_layer/period_3quarter/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
 
Hdnn/input_from_feature_columns/input_layer/period_3quarter/strided_sliceStridedSlice@dnn/input_from_feature_columns/input_layer/period_3quarter/ShapeNdnn/input_from_feature_columns/input_layer/period_3quarter/strided_slice/stackPdnn/input_from_feature_columns/input_layer/period_3quarter/strided_slice/stack_1Pdnn/input_from_feature_columns/input_layer/period_3quarter/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0

Jdnn/input_from_feature_columns/input_layer/period_3quarter/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
 
Hdnn/input_from_feature_columns/input_layer/period_3quarter/Reshape/shapePackHdnn/input_from_feature_columns/input_layer/period_3quarter/strided_sliceJdnn/input_from_feature_columns/input_layer/period_3quarter/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
ő
Bdnn/input_from_feature_columns/input_layer/period_3quarter/ReshapeReshapeParseExample/ParseExample:22Hdnn/input_from_feature_columns/input_layer/period_3quarter/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

@dnn/input_from_feature_columns/input_layer/period_4quarter/ShapeShapeParseExample/ParseExample:23*
_output_shapes
:*
T0*
out_type0

Ndnn/input_from_feature_columns/input_layer/period_4quarter/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Pdnn/input_from_feature_columns/input_layer/period_4quarter/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

Pdnn/input_from_feature_columns/input_layer/period_4quarter/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
 
Hdnn/input_from_feature_columns/input_layer/period_4quarter/strided_sliceStridedSlice@dnn/input_from_feature_columns/input_layer/period_4quarter/ShapeNdnn/input_from_feature_columns/input_layer/period_4quarter/strided_slice/stackPdnn/input_from_feature_columns/input_layer/period_4quarter/strided_slice/stack_1Pdnn/input_from_feature_columns/input_layer/period_4quarter/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

Jdnn/input_from_feature_columns/input_layer/period_4quarter/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
 
Hdnn/input_from_feature_columns/input_layer/period_4quarter/Reshape/shapePackHdnn/input_from_feature_columns/input_layer/period_4quarter/strided_sliceJdnn/input_from_feature_columns/input_layer/period_4quarter/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
ő
Bdnn/input_from_feature_columns/input_layer/period_4quarter/ReshapeReshapeParseExample/ParseExample:23Hdnn/input_from_feature_columns/input_layer/period_4quarter/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

Adnn/input_from_feature_columns/input_layer/period_overayear/ShapeShapeParseExample/ParseExample:24*
_output_shapes
:*
T0*
out_type0

Odnn/input_from_feature_columns/input_layer/period_overayear/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Qdnn/input_from_feature_columns/input_layer/period_overayear/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Qdnn/input_from_feature_columns/input_layer/period_overayear/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ľ
Idnn/input_from_feature_columns/input_layer/period_overayear/strided_sliceStridedSliceAdnn/input_from_feature_columns/input_layer/period_overayear/ShapeOdnn/input_from_feature_columns/input_layer/period_overayear/strided_slice/stackQdnn/input_from_feature_columns/input_layer/period_overayear/strided_slice/stack_1Qdnn/input_from_feature_columns/input_layer/period_overayear/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0

Kdnn/input_from_feature_columns/input_layer/period_overayear/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ł
Idnn/input_from_feature_columns/input_layer/period_overayear/Reshape/shapePackIdnn/input_from_feature_columns/input_layer/period_overayear/strided_sliceKdnn/input_from_feature_columns/input_layer/period_overayear/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
÷
Cdnn/input_from_feature_columns/input_layer/period_overayear/ReshapeReshapeParseExample/ParseExample:24Idnn/input_from_feature_columns/input_layer/period_overayear/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

:dnn/input_from_feature_columns/input_layer/relevante/ShapeShapeParseExample/ParseExample:25*
_output_shapes
:*
T0*
out_type0

Hdnn/input_from_feature_columns/input_layer/relevante/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Jdnn/input_from_feature_columns/input_layer/relevante/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Jdnn/input_from_feature_columns/input_layer/relevante/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

Bdnn/input_from_feature_columns/input_layer/relevante/strided_sliceStridedSlice:dnn/input_from_feature_columns/input_layer/relevante/ShapeHdnn/input_from_feature_columns/input_layer/relevante/strided_slice/stackJdnn/input_from_feature_columns/input_layer/relevante/strided_slice/stack_1Jdnn/input_from_feature_columns/input_layer/relevante/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask

Ddnn/input_from_feature_columns/input_layer/relevante/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Bdnn/input_from_feature_columns/input_layer/relevante/Reshape/shapePackBdnn/input_from_feature_columns/input_layer/relevante/strided_sliceDdnn/input_from_feature_columns/input_layer/relevante/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
é
<dnn/input_from_feature_columns/input_layer/relevante/ReshapeReshapeParseExample/ParseExample:25Bdnn/input_from_feature_columns/input_layer/relevante/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Adnn/input_from_feature_columns/input_layer/totalamount_base/ShapeShapeParseExample/ParseExample:26*
T0*
out_type0*
_output_shapes
:

Odnn/input_from_feature_columns/input_layer/totalamount_base/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

Qdnn/input_from_feature_columns/input_layer/totalamount_base/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Qdnn/input_from_feature_columns/input_layer/totalamount_base/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ľ
Idnn/input_from_feature_columns/input_layer/totalamount_base/strided_sliceStridedSliceAdnn/input_from_feature_columns/input_layer/totalamount_base/ShapeOdnn/input_from_feature_columns/input_layer/totalamount_base/strided_slice/stackQdnn/input_from_feature_columns/input_layer/totalamount_base/strided_slice/stack_1Qdnn/input_from_feature_columns/input_layer/totalamount_base/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0

Kdnn/input_from_feature_columns/input_layer/totalamount_base/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ł
Idnn/input_from_feature_columns/input_layer/totalamount_base/Reshape/shapePackIdnn/input_from_feature_columns/input_layer/totalamount_base/strided_sliceKdnn/input_from_feature_columns/input_layer/totalamount_base/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
÷
Cdnn/input_from_feature_columns/input_layer/totalamount_base/ReshapeReshapeParseExample/ParseExample:26Idnn/input_from_feature_columns/input_layer/totalamount_base/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Ű
1dnn/input_from_feature_columns/input_layer/concatConcatV2Ednn/input_from_feature_columns/input_layer/etapa_Calificacion/ReshapeCdnn/input_from_feature_columns/input_layer/etapa_Desarrollo/ReshapeDdnn/input_from_feature_columns/input_layer/etapa_Negociacion/ReshapeDdnn/input_from_feature_columns/input_layer/etapa_Prospeccion/Reshape?dnn/input_from_feature_columns/input_layer/etapa_Prueba/ReshapeAdnn/input_from_feature_columns/input_layer/etapa_Solucion/Reshapeadnn/input_from_feature_columns/input_layer/modalidad_4445B64B-338B-E611-80F0-2C59E53A5504/Reshapeadnn/input_from_feature_columns/input_layer/modalidad_4645B64B-338B-E611-80F0-2C59E53A5504/Reshapeadnn/input_from_feature_columns/input_layer/modalidad_4845B64B-338B-E611-80F0-2C59E53A5504/Reshapeadnn/input_from_feature_columns/input_layer/modalidad_4A45B64B-338B-E611-80F0-2C59E53A5504/Reshapeadnn/input_from_feature_columns/input_layer/modalidad_59C2334F-CDF2-E611-80FF-C4346BB56EE0/ReshapeRdnn/input_from_feature_columns/input_layer/naturaleza_AmpliacionoUpselling/ReshapeCdnn/input_from_feature_columns/input_layer/naturaleza_Nuevo/ReshapeOdnn/input_from_feature_columns/input_layer/naturaleza_NuevoCrossSelling/ReshapeEdnn/input_from_feature_columns/input_layer/naturaleza_Paragua/ReshapeFdnn/input_from_feature_columns/input_layer/naturaleza_Reajuste/ReshapeUdnn/input_from_feature_columns/input_layer/naturaleza_Reconocimientodeingreso/ReshapeKdnn/input_from_feature_columns/input_layer/naturaleza_Renegociacion/ReshapeHdnn/input_from_feature_columns/input_layer/naturaleza_Renovacion/ReshapeSdnn/input_from_feature_columns/input_layer/naturaleza_Renovacioncompetitiva/ReshapeBdnn/input_from_feature_columns/input_layer/period_1quarter/ReshapeBdnn/input_from_feature_columns/input_layer/period_2quarter/ReshapeBdnn/input_from_feature_columns/input_layer/period_3quarter/ReshapeBdnn/input_from_feature_columns/input_layer/period_4quarter/ReshapeCdnn/input_from_feature_columns/input_layer/period_overayear/Reshape<dnn/input_from_feature_columns/input_layer/relevante/ReshapeCdnn/input_from_feature_columns/input_layer/totalamount_base/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
T0
Ĺ
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
valueB"   ?   
ˇ
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
valueB
 *Ľ2ž*
dtype0*
_output_shapes
: 
ˇ
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
valueB
 *Ľ2>*
dtype0*
_output_shapes
: 

Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes

:?*

seed *
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0

>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
Ź
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
_output_shapes

:?*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0

:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:?
Ç
dnn/hiddenlayer_0/kernel/part_0
VariableV2*
dtype0*
_output_shapes

:?*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
	container *
shape
:?

&dnn/hiddenlayer_0/kernel/part_0/AssignAssigndnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(*
_output_shapes

:?*
use_locking(
Ž
$dnn/hiddenlayer_0/kernel/part_0/readIdentitydnn/hiddenlayer_0/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:?
Ž
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
valueB?*    *
dtype0*
_output_shapes
:?
ť
dnn/hiddenlayer_0/bias/part_0
VariableV2*
dtype0*
_output_shapes
:?*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
	container *
shape:?
ţ
$dnn/hiddenlayer_0/bias/part_0/AssignAssigndnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*
validate_shape(*
_output_shapes
:?*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0
¤
"dnn/hiddenlayer_0/bias/part_0/readIdentitydnn/hiddenlayer_0/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
:?
s
dnn/hiddenlayer_0/kernelIdentity$dnn/hiddenlayer_0/kernel/part_0/read*
T0*
_output_shapes

:?
Ç
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙?*
transpose_a( *
transpose_b( 
k
dnn/hiddenlayer_0/biasIdentity"dnn/hiddenlayer_0/bias/part_0/read*
T0*
_output_shapes
:?

dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙?
k
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙?
[
dnn/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 

dnn/zero_fraction/EqualEqualdnn/hiddenlayer_0/Reludnn/zero_fraction/zero*'
_output_shapes
:˙˙˙˙˙˙˙˙˙?*
T0
x
dnn/zero_fraction/CastCastdnn/zero_fraction/Equal*'
_output_shapes
:˙˙˙˙˙˙˙˙˙?*

DstT0*

SrcT0

h
dnn/zero_fraction/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

dnn/zero_fraction/MeanMeandnn/zero_fraction/Castdnn/zero_fraction/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
 
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values*
dtype0*
_output_shapes
: 
Ť
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/Mean*
T0*
_output_shapes
: 

$dnn/dnn/hiddenlayer_0/activation/tagConst*
dtype0*
_output_shapes
: *1
value(B& B dnn/dnn/hiddenlayer_0/activation

 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
ˇ
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*+
_class!
loc:@dnn/logits/kernel/part_0*
valueB"?      *
dtype0*
_output_shapes
:
Š
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*+
_class!
loc:@dnn/logits/kernel/part_0*
valueB
 *qÄž*
dtype0*
_output_shapes
: 
Š
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/logits/kernel/part_0*
valueB
 *qÄ>*
dtype0*
_output_shapes
: 

Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*

seed *
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
seed2 *
dtype0*
_output_shapes

:?
ţ
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
: *
T0*+
_class!
loc:@dnn/logits/kernel/part_0

7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:?

3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:?
š
dnn/logits/kernel/part_0
VariableV2*
dtype0*
_output_shapes

:?*
shared_name *+
_class!
loc:@dnn/logits/kernel/part_0*
	container *
shape
:?
÷
dnn/logits/kernel/part_0/AssignAssigndnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*
validate_shape(*
_output_shapes

:?*
use_locking(*
T0*+
_class!
loc:@dnn/logits/kernel/part_0

dnn/logits/kernel/part_0/readIdentitydnn/logits/kernel/part_0*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:?
 
(dnn/logits/bias/part_0/Initializer/zerosConst*
dtype0*
_output_shapes
:*)
_class
loc:@dnn/logits/bias/part_0*
valueB*    
­
dnn/logits/bias/part_0
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@dnn/logits/bias/part_0
â
dnn/logits/bias/part_0/AssignAssigndnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(*
_output_shapes
:

dnn/logits/bias/part_0/readIdentitydnn/logits/bias/part_0*
T0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:
e
dnn/logits/kernelIdentitydnn/logits/kernel/part_0/read*
T0*
_output_shapes

:?

dnn/logits/MatMulMatMuldnn/hiddenlayer_0/Reludnn/logits/kernel*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
]
dnn/logits/biasIdentitydnn/logits/bias/part_0/read*
T0*
_output_shapes
:

dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
]
dnn/zero_fraction_1/zeroConst*
dtype0*
_output_shapes
: *
valueB
 *    

dnn/zero_fraction_1/EqualEqualdnn/logits/BiasAdddnn/zero_fraction_1/zero*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
dnn/zero_fraction_1/CastCastdnn/zero_fraction_1/Equal*

SrcT0
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
j
dnn/zero_fraction_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

dnn/zero_fraction_1/MeanMeandnn/zero_fraction_1/Castdnn/zero_fraction_1/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

+dnn/dnn/logits/fraction_of_zero_values/tagsConst*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
dtype0*
_output_shapes
: 

&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_1/Mean*
T0*
_output_shapes
: 
w
dnn/dnn/logits/activation/tagConst**
value!B Bdnn/dnn/logits/activation*
dtype0*
_output_shapes
: 

dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
_output_shapes
: *
T0
s
!dnn/head/predictions/logits/ShapeShapednn/logits/BiasAdd*
T0*
out_type0*
_output_shapes
:
w
5dnn/head/predictions/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
g
_dnn/head/predictions/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
X
Pdnn/head/predictions/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
n
dnn/head/predictions/logisticSigmoiddnn/logits/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
dnn/head/predictions/zeros_like	ZerosLikednn/logits/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
*dnn/head/predictions/two_class_logits/axisConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ů
%dnn/head/predictions/two_class_logitsConcatV2dnn/head/predictions/zeros_likednn/logits/BiasAdd*dnn/head/predictions/two_class_logits/axis*

Tidx0*
T0*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

"dnn/head/predictions/probabilitiesSoftmax%dnn/head/predictions/two_class_logits*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
(dnn/head/predictions/class_ids/dimensionConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ć
dnn/head/predictions/class_idsArgMax%dnn/head/predictions/two_class_logits(dnn/head/predictions/class_ids/dimension*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
T0
n
#dnn/head/predictions/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
°
dnn/head/predictions/ExpandDims
ExpandDimsdnn/head/predictions/class_ids#dnn/head/predictions/ExpandDims/dim*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tdim0
Ý
 dnn/head/predictions/str_classesAsStringdnn/head/predictions/ExpandDims*
width˙˙˙˙˙˙˙˙˙*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	precision˙˙˙˙˙˙˙˙˙*
shortest( *
T0	*

fill *

scientific( 
p
dnn/head/ShapeShape"dnn/head/predictions/probabilities*
T0*
out_type0*
_output_shapes
:
f
dnn/head/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
h
dnn/head/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
h
dnn/head/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ś
dnn/head/strided_sliceStridedSlicednn/head/Shapednn/head/strided_slice/stackdnn/head/strided_slice/stack_1dnn/head/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
V
dnn/head/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
V
dnn/head/range/limitConst*
dtype0*
_output_shapes
: *
value	B :
V
dnn/head/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

dnn/head/rangeRangednn/head/range/startdnn/head/range/limitdnn/head/range/delta*
_output_shapes
:*

Tidx0
°
dnn/head/AsStringAsStringdnn/head/range*
width˙˙˙˙˙˙˙˙˙*
_output_shapes
:*
	precision˙˙˙˙˙˙˙˙˙*
shortest( *
T0*

fill *

scientific( 
Y
dnn/head/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 

dnn/head/ExpandDims
ExpandDimsdnn/head/AsStringdnn/head/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
[
dnn/head/Tile/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 

dnn/head/Tile/multiplesPackdnn/head/strided_slicednn/head/Tile/multiples/1*
T0*

axis *
N*
_output_shapes
:

dnn/head/TileTilednn/head/ExpandDimsdnn/head/Tile/multiples*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0*
T0
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_16774954d0644b04a765f5ff52dade47/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
Ń
save/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*v
valuemBkBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step
¤
save/SaveV2/shape_and_slicesConst"/device:CPU:0*E
value<B:B63 0,63B27 63 0,27:0,63B1 0,1B63 1 0,63:0,1B *
dtype0*
_output_shapes
:

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices"dnn/hiddenlayer_0/bias/part_0/read$dnn/hiddenlayer_0/kernel/part_0/readdnn/logits/bias/part_0/readdnn/logits/kernel/part_0/readglobal_step"/device:CPU:0*
dtypes	
2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
Ź
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
Ô
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*v
valuemBkBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step
§
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*E
value<B:B63 0,63B27 63 0,27:0,63B1 0,1B63 1 0,63:0,1B *
dtype0*
_output_shapes
:
Ă
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes	
2	*8
_output_shapes&
$:?:?::?:
Ä
save/AssignAssigndnn/hiddenlayer_0/bias/part_0save/RestoreV2*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes
:?*
use_locking(
Đ
save/Assign_1Assigndnn/hiddenlayer_0/kernel/part_0save/RestoreV2:1*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(*
_output_shapes

:?
ş
save/Assign_2Assigndnn/logits/bias/part_0save/RestoreV2:2*
T0*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(*
_output_shapes
:*
use_locking(
Â
save/Assign_3Assigndnn/logits/kernel/part_0save/RestoreV2:3*
use_locking(*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
validate_shape(*
_output_shapes

:?
 
save/Assign_4Assignglobal_stepsave/RestoreV2:4*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
h
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4
-
save/restore_allNoOp^save/restore_shard

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_1/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_310d494975c74e2f846d32528be249d1/part
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
Ó
save_1/SaveV2/tensor_namesConst"/device:CPU:0*v
valuemBkBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:
Ś
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*E
value<B:B63 0,63B27 63 0,27:0,63B1 0,1B63 1 0,63:0,1B *
dtype0*
_output_shapes
:
˘
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slices"dnn/hiddenlayer_0/bias/part_0/read$dnn/hiddenlayer_0/kernel/part_0/readdnn/logits/bias/part_0/readdnn/logits/kernel/part_0/readglobal_step"/device:CPU:0*
dtypes	
2	
¨
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
˛
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
Ö
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*v
valuemBkBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:
Š
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*E
value<B:B63 0,63B27 63 0,27:0,63B1 0,1B63 1 0,63:0,1B 
Ë
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes	
2	*8
_output_shapes&
$:?:?::?:
Č
save_1/AssignAssigndnn/hiddenlayer_0/bias/part_0save_1/RestoreV2*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes
:?
Ô
save_1/Assign_1Assigndnn/hiddenlayer_0/kernel/part_0save_1/RestoreV2:1*
validate_shape(*
_output_shapes

:?*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
ž
save_1/Assign_2Assigndnn/logits/bias/part_0save_1/RestoreV2:2*
use_locking(*
T0*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(*
_output_shapes
:
Ć
save_1/Assign_3Assigndnn/logits/kernel/part_0save_1/RestoreV2:3*
use_locking(*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
validate_shape(*
_output_shapes

:?
¤
save_1/Assign_4Assignglobal_stepsave_1/RestoreV2:4*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step
t
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"é
	variablesŰŘ
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
Ů
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign&dnn/hiddenlayer_0/kernel/part_0/read:0"&
dnn/hiddenlayer_0/kernel?  "?2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:0
Ă
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign$dnn/hiddenlayer_0/bias/part_0/read:0"!
dnn/hiddenlayer_0/bias? "?21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:0
ś
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assigndnn/logits/kernel/part_0/read:0"
dnn/logits/kernel?  "?25dnn/logits/kernel/part_0/Initializer/random_uniform:0
 
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assigndnn/logits/bias/part_0/read:0"
dnn/logits/bias "2*dnn/logits/bias/part_0/Initializer/zeros:0" 
legacy_init_op


group_deps"­
	summaries

/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0"
trainable_variablesţ
Ů
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign&dnn/hiddenlayer_0/kernel/part_0/read:0"&
dnn/hiddenlayer_0/kernel?  "?2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:0
Ă
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign$dnn/hiddenlayer_0/bias/part_0/read:0"!
dnn/hiddenlayer_0/bias? "?21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:0
ś
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assigndnn/logits/kernel/part_0/read:0"
dnn/logits/kernel?  "?25dnn/logits/kernel/part_0/Initializer/random_uniform:0
 
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assigndnn/logits/bias/part_0/read:0"
dnn/logits/bias "2*dnn/logits/bias/part_0/Initializer/zeros:0*ľ
predictŠ
5
examples)
input_example_tensor:0˙˙˙˙˙˙˙˙˙B
logistic6
dnn/head/predictions/logistic:0˙˙˙˙˙˙˙˙˙E
	class_ids8
!dnn/head/predictions/ExpandDims:0	˙˙˙˙˙˙˙˙˙L
probabilities;
$dnn/head/predictions/probabilities:0˙˙˙˙˙˙˙˙˙D
classes9
"dnn/head/predictions/str_classes:0˙˙˙˙˙˙˙˙˙5
logits+
dnn/logits/BiasAdd:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict*ß
classificationĚ
3
inputs)
input_example_tensor:0˙˙˙˙˙˙˙˙˙1
classes&
dnn/head/Tile:0˙˙˙˙˙˙˙˙˙E
scores;
$dnn/head/predictions/probabilities:0˙˙˙˙˙˙˙˙˙tensorflow/serving/classify*Ł

regression
3
inputs)
input_example_tensor:0˙˙˙˙˙˙˙˙˙A
outputs6
dnn/head/predictions/logistic:0˙˙˙˙˙˙˙˙˙tensorflow/serving/regress*ŕ
serving_defaultĚ
3
inputs)
input_example_tensor:0˙˙˙˙˙˙˙˙˙1
classes&
dnn/head/Tile:0˙˙˙˙˙˙˙˙˙E
scores;
$dnn/head/predictions/probabilities:0˙˙˙˙˙˙˙˙˙tensorflow/serving/classify