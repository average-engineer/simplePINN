¼
·
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
?
Select
	condition

t"T
e"T
output"T"	
Ttype
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
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.02v2.6.0-rc2-32-g919f693420e8ñ¶
w
Layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*
shared_nameLayer1/kernel
p
!Layer1/kernel/Read/ReadVariableOpReadVariableOpLayer1/kernel*
_output_shapes
:	¬*
dtype0
o
Layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*
shared_nameLayer1/bias
h
Layer1/bias/Read/ReadVariableOpReadVariableOpLayer1/bias*
_output_shapes	
:¬*
dtype0
x
Layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬¬*
shared_nameLayer2/kernel
q
!Layer2/kernel/Read/ReadVariableOpReadVariableOpLayer2/kernel* 
_output_shapes
:
¬¬*
dtype0
o
Layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*
shared_nameLayer2/bias
h
Layer2/bias/Read/ReadVariableOpReadVariableOpLayer2/bias*
_output_shapes	
:¬*
dtype0
x
Layer3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬¬*
shared_nameLayer3/kernel
q
!Layer3/kernel/Read/ReadVariableOpReadVariableOpLayer3/kernel* 
_output_shapes
:
¬¬*
dtype0
o
Layer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*
shared_nameLayer3/bias
h
Layer3/bias/Read/ReadVariableOpReadVariableOpLayer3/bias*
_output_shapes	
:¬*
dtype0

lambda_layer_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬¬*'
shared_namelambda_layer_33/kernel

*lambda_layer_33/kernel/Read/ReadVariableOpReadVariableOplambda_layer_33/kernel* 
_output_shapes
:
¬¬*
dtype0

lambda_layer_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*%
shared_namelambda_layer_33/bias
z
(lambda_layer_33/bias/Read/ReadVariableOpReadVariableOplambda_layer_33/bias*
_output_shapes	
:¬*
dtype0

lambda_layer_33/LambdaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namelambda_layer_33/Lambda
}
*lambda_layer_33/Lambda/Read/ReadVariableOpReadVariableOplambda_layer_33/Lambda*
_output_shapes
:*
dtype0

OutputLayer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*#
shared_nameOutputLayer/kernel
z
&OutputLayer/kernel/Read/ReadVariableOpReadVariableOpOutputLayer/kernel*
_output_shapes
:	¬*
dtype0
x
OutputLayer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameOutputLayer/bias
q
$OutputLayer/bias/Read/ReadVariableOpReadVariableOpOutputLayer/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/Layer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*%
shared_nameAdam/Layer1/kernel/m
~
(Adam/Layer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Layer1/kernel/m*
_output_shapes
:	¬*
dtype0
}
Adam/Layer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*#
shared_nameAdam/Layer1/bias/m
v
&Adam/Layer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Layer1/bias/m*
_output_shapes	
:¬*
dtype0

Adam/Layer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬¬*%
shared_nameAdam/Layer2/kernel/m

(Adam/Layer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Layer2/kernel/m* 
_output_shapes
:
¬¬*
dtype0
}
Adam/Layer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*#
shared_nameAdam/Layer2/bias/m
v
&Adam/Layer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Layer2/bias/m*
_output_shapes	
:¬*
dtype0

Adam/Layer3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬¬*%
shared_nameAdam/Layer3/kernel/m

(Adam/Layer3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Layer3/kernel/m* 
_output_shapes
:
¬¬*
dtype0
}
Adam/Layer3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*#
shared_nameAdam/Layer3/bias/m
v
&Adam/Layer3/bias/m/Read/ReadVariableOpReadVariableOpAdam/Layer3/bias/m*
_output_shapes	
:¬*
dtype0

Adam/lambda_layer_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬¬*.
shared_nameAdam/lambda_layer_33/kernel/m

1Adam/lambda_layer_33/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lambda_layer_33/kernel/m* 
_output_shapes
:
¬¬*
dtype0

Adam/lambda_layer_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*,
shared_nameAdam/lambda_layer_33/bias/m

/Adam/lambda_layer_33/bias/m/Read/ReadVariableOpReadVariableOpAdam/lambda_layer_33/bias/m*
_output_shapes	
:¬*
dtype0

Adam/lambda_layer_33/Lambda/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/lambda_layer_33/Lambda/m

1Adam/lambda_layer_33/Lambda/m/Read/ReadVariableOpReadVariableOpAdam/lambda_layer_33/Lambda/m*
_output_shapes
:*
dtype0

Adam/OutputLayer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬**
shared_nameAdam/OutputLayer/kernel/m

-Adam/OutputLayer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/OutputLayer/kernel/m*
_output_shapes
:	¬*
dtype0

Adam/OutputLayer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/OutputLayer/bias/m

+Adam/OutputLayer/bias/m/Read/ReadVariableOpReadVariableOpAdam/OutputLayer/bias/m*
_output_shapes
:*
dtype0

Adam/Layer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬*%
shared_nameAdam/Layer1/kernel/v
~
(Adam/Layer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Layer1/kernel/v*
_output_shapes
:	¬*
dtype0
}
Adam/Layer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*#
shared_nameAdam/Layer1/bias/v
v
&Adam/Layer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Layer1/bias/v*
_output_shapes	
:¬*
dtype0

Adam/Layer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬¬*%
shared_nameAdam/Layer2/kernel/v

(Adam/Layer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Layer2/kernel/v* 
_output_shapes
:
¬¬*
dtype0
}
Adam/Layer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*#
shared_nameAdam/Layer2/bias/v
v
&Adam/Layer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Layer2/bias/v*
_output_shapes	
:¬*
dtype0

Adam/Layer3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬¬*%
shared_nameAdam/Layer3/kernel/v

(Adam/Layer3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Layer3/kernel/v* 
_output_shapes
:
¬¬*
dtype0
}
Adam/Layer3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*#
shared_nameAdam/Layer3/bias/v
v
&Adam/Layer3/bias/v/Read/ReadVariableOpReadVariableOpAdam/Layer3/bias/v*
_output_shapes	
:¬*
dtype0

Adam/lambda_layer_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬¬*.
shared_nameAdam/lambda_layer_33/kernel/v

1Adam/lambda_layer_33/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lambda_layer_33/kernel/v* 
_output_shapes
:
¬¬*
dtype0

Adam/lambda_layer_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*,
shared_nameAdam/lambda_layer_33/bias/v

/Adam/lambda_layer_33/bias/v/Read/ReadVariableOpReadVariableOpAdam/lambda_layer_33/bias/v*
_output_shapes	
:¬*
dtype0

Adam/lambda_layer_33/Lambda/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/lambda_layer_33/Lambda/v

1Adam/lambda_layer_33/Lambda/v/Read/ReadVariableOpReadVariableOpAdam/lambda_layer_33/Lambda/v*
_output_shapes
:*
dtype0

Adam/OutputLayer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¬**
shared_nameAdam/OutputLayer/kernel/v

-Adam/OutputLayer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/OutputLayer/kernel/v*
_output_shapes
:	¬*
dtype0

Adam/OutputLayer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/OutputLayer/bias/v

+Adam/OutputLayer/bias/v/Read/ReadVariableOpReadVariableOpAdam/OutputLayer/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ð7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*«7
value¡7B7 B7
´
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
j
w
b
 lbda
!regularization_losses
"	variables
#trainable_variables
$	keras_api
h

%kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api

+iter

,beta_1

-beta_2
	.decay
/learning_ratemSmTmUmVmWmXmYmZ m[%m\&m]v^v_v`vavbvcvdve vf%vg&vh
 
N
0
1
2
3
4
5
6
7
 8
%9
&10
N
0
1
2
3
4
5
6
7
 8
%9
&10
­
0layer_regularization_losses
1metrics

2layers
3non_trainable_variables
4layer_metrics
regularization_losses
	variables
	trainable_variables
 
YW
VARIABLE_VALUELayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUELayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
5layer_regularization_losses
6metrics

7layers
8non_trainable_variables
9layer_metrics
regularization_losses
	variables
trainable_variables
YW
VARIABLE_VALUELayer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUELayer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
:layer_regularization_losses
;metrics

<layers
=non_trainable_variables
>layer_metrics
regularization_losses
	variables
trainable_variables
YW
VARIABLE_VALUELayer3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUELayer3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
?layer_regularization_losses
@metrics

Alayers
Bnon_trainable_variables
Clayer_metrics
regularization_losses
	variables
trainable_variables
][
VARIABLE_VALUElambda_layer_33/kernel1layer_with_weights-3/w/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElambda_layer_33/bias1layer_with_weights-3/b/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUElambda_layer_33/Lambda4layer_with_weights-3/lbda/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 2

0
1
 2
­
Dlayer_regularization_losses
Emetrics

Flayers
Gnon_trainable_variables
Hlayer_metrics
!regularization_losses
"	variables
#trainable_variables
^\
VARIABLE_VALUEOutputLayer/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEOutputLayer/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1

%0
&1
­
Ilayer_regularization_losses
Jmetrics

Klayers
Lnon_trainable_variables
Mlayer_metrics
'regularization_losses
(	variables
)trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

N0
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ototal
	Pcount
Q	variables
R	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

O0
P1

Q	variables
|z
VARIABLE_VALUEAdam/Layer1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/Layer1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Layer2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/Layer2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Layer3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/Layer3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/lambda_layer_33/kernel/mMlayer_with_weights-3/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/lambda_layer_33/bias/mMlayer_with_weights-3/b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/lambda_layer_33/Lambda/mPlayer_with_weights-3/lbda/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/OutputLayer/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/OutputLayer/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Layer1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/Layer1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Layer2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/Layer2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Layer3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/Layer3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/lambda_layer_33/kernel/vMlayer_with_weights-3/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/lambda_layer_33/bias/vMlayer_with_weights-3/b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/lambda_layer_33/Lambda/vPlayer_with_weights-3/lbda/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/OutputLayer/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/OutputLayer/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_34Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_34Layer1/kernelLayer1/biasLayer2/kernelLayer2/biasLayer3/kernelLayer3/biaslambda_layer_33/kernellambda_layer_33/biaslambda_layer_33/LambdaOutputLayer/kernelOutputLayer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_105653
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!Layer1/kernel/Read/ReadVariableOpLayer1/bias/Read/ReadVariableOp!Layer2/kernel/Read/ReadVariableOpLayer2/bias/Read/ReadVariableOp!Layer3/kernel/Read/ReadVariableOpLayer3/bias/Read/ReadVariableOp*lambda_layer_33/kernel/Read/ReadVariableOp(lambda_layer_33/bias/Read/ReadVariableOp*lambda_layer_33/Lambda/Read/ReadVariableOp&OutputLayer/kernel/Read/ReadVariableOp$OutputLayer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/Layer1/kernel/m/Read/ReadVariableOp&Adam/Layer1/bias/m/Read/ReadVariableOp(Adam/Layer2/kernel/m/Read/ReadVariableOp&Adam/Layer2/bias/m/Read/ReadVariableOp(Adam/Layer3/kernel/m/Read/ReadVariableOp&Adam/Layer3/bias/m/Read/ReadVariableOp1Adam/lambda_layer_33/kernel/m/Read/ReadVariableOp/Adam/lambda_layer_33/bias/m/Read/ReadVariableOp1Adam/lambda_layer_33/Lambda/m/Read/ReadVariableOp-Adam/OutputLayer/kernel/m/Read/ReadVariableOp+Adam/OutputLayer/bias/m/Read/ReadVariableOp(Adam/Layer1/kernel/v/Read/ReadVariableOp&Adam/Layer1/bias/v/Read/ReadVariableOp(Adam/Layer2/kernel/v/Read/ReadVariableOp&Adam/Layer2/bias/v/Read/ReadVariableOp(Adam/Layer3/kernel/v/Read/ReadVariableOp&Adam/Layer3/bias/v/Read/ReadVariableOp1Adam/lambda_layer_33/kernel/v/Read/ReadVariableOp/Adam/lambda_layer_33/bias/v/Read/ReadVariableOp1Adam/lambda_layer_33/Lambda/v/Read/ReadVariableOp-Adam/OutputLayer/kernel/v/Read/ReadVariableOp+Adam/OutputLayer/bias/v/Read/ReadVariableOpConst*5
Tin.
,2*	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_106276
ë
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameLayer1/kernelLayer1/biasLayer2/kernelLayer2/biasLayer3/kernelLayer3/biaslambda_layer_33/kernellambda_layer_33/biaslambda_layer_33/LambdaOutputLayer/kernelOutputLayer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/Layer1/kernel/mAdam/Layer1/bias/mAdam/Layer2/kernel/mAdam/Layer2/bias/mAdam/Layer3/kernel/mAdam/Layer3/bias/mAdam/lambda_layer_33/kernel/mAdam/lambda_layer_33/bias/mAdam/lambda_layer_33/Lambda/mAdam/OutputLayer/kernel/mAdam/OutputLayer/bias/mAdam/Layer1/kernel/vAdam/Layer1/bias/vAdam/Layer2/kernel/vAdam/Layer2/bias/vAdam/Layer3/kernel/vAdam/Layer3/bias/vAdam/lambda_layer_33/kernel/vAdam/lambda_layer_33/bias/vAdam/lambda_layer_33/Lambda/vAdam/OutputLayer/kernel/vAdam/OutputLayer/bias/v*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_106406Ü

É!
ü
B__inference_Layer2_layer_call_and_return_conditional_losses_105265

inputs5
!tensordot_readvariableop_resource:
¬¬.
biasadd_readvariableop_resource:	¬
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
¬¬*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:¬2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tanhh
IdentityIdentityTanh:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Æ
¡
I__inference_sequential_33_layer_call_and_return_conditional_losses_105618
input_34 
layer1_105590:	¬
layer1_105592:	¬!
layer2_105595:
¬¬
layer2_105597:	¬!
layer3_105600:
¬¬
layer3_105602:	¬*
lambda_layer_33_105605:
¬¬%
lambda_layer_33_105607:	¬$
lambda_layer_33_105609:%
outputlayer_105612:	¬ 
outputlayer_105614:
identity¢Layer1/StatefulPartitionedCall¢Layer2/StatefulPartitionedCall¢Layer3/StatefulPartitionedCall¢#OutputLayer/StatefulPartitionedCall¢'lambda_layer_33/StatefulPartitionedCall
Layer1/StatefulPartitionedCallStatefulPartitionedCallinput_34layer1_105590layer1_105592*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Layer1_layer_call_and_return_conditional_losses_1052282 
Layer1/StatefulPartitionedCall°
Layer2/StatefulPartitionedCallStatefulPartitionedCall'Layer1/StatefulPartitionedCall:output:0layer2_105595layer2_105597*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Layer2_layer_call_and_return_conditional_losses_1052652 
Layer2/StatefulPartitionedCall°
Layer3/StatefulPartitionedCallStatefulPartitionedCall'Layer2/StatefulPartitionedCall:output:0layer3_105600layer3_105602*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Layer3_layer_call_and_return_conditional_losses_1053022 
Layer3/StatefulPartitionedCall÷
'lambda_layer_33/StatefulPartitionedCallStatefulPartitionedCall'Layer3/StatefulPartitionedCall:output:0lambda_layer_33_105605lambda_layer_33_105607lambda_layer_33_105609*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_lambda_layer_33_layer_call_and_return_conditional_losses_1053222)
'lambda_layer_33/StatefulPartitionedCallÑ
#OutputLayer/StatefulPartitionedCallStatefulPartitionedCall0lambda_layer_33/StatefulPartitionedCall:output:0outputlayer_105612outputlayer_105614*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_OutputLayer_layer_call_and_return_conditional_losses_1053602%
#OutputLayer/StatefulPartitionedCall
IdentityIdentity,OutputLayer/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^Layer1/StatefulPartitionedCall^Layer2/StatefulPartitionedCall^Layer3/StatefulPartitionedCall$^OutputLayer/StatefulPartitionedCall(^lambda_layer_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2@
Layer1/StatefulPartitionedCallLayer1/StatefulPartitionedCall2@
Layer2/StatefulPartitionedCallLayer2/StatefulPartitionedCall2@
Layer3/StatefulPartitionedCallLayer3/StatefulPartitionedCall2J
#OutputLayer/StatefulPartitionedCall#OutputLayer/StatefulPartitionedCall2R
'lambda_layer_33/StatefulPartitionedCall'lambda_layer_33/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_34
À

I__inference_sequential_33_layer_call_and_return_conditional_losses_105504

inputs 
layer1_105476:	¬
layer1_105478:	¬!
layer2_105481:
¬¬
layer2_105483:	¬!
layer3_105486:
¬¬
layer3_105488:	¬*
lambda_layer_33_105491:
¬¬%
lambda_layer_33_105493:	¬$
lambda_layer_33_105495:%
outputlayer_105498:	¬ 
outputlayer_105500:
identity¢Layer1/StatefulPartitionedCall¢Layer2/StatefulPartitionedCall¢Layer3/StatefulPartitionedCall¢#OutputLayer/StatefulPartitionedCall¢'lambda_layer_33/StatefulPartitionedCall
Layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_105476layer1_105478*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Layer1_layer_call_and_return_conditional_losses_1052282 
Layer1/StatefulPartitionedCall°
Layer2/StatefulPartitionedCallStatefulPartitionedCall'Layer1/StatefulPartitionedCall:output:0layer2_105481layer2_105483*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Layer2_layer_call_and_return_conditional_losses_1052652 
Layer2/StatefulPartitionedCall°
Layer3/StatefulPartitionedCallStatefulPartitionedCall'Layer2/StatefulPartitionedCall:output:0layer3_105486layer3_105488*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Layer3_layer_call_and_return_conditional_losses_1053022 
Layer3/StatefulPartitionedCall÷
'lambda_layer_33/StatefulPartitionedCallStatefulPartitionedCall'Layer3/StatefulPartitionedCall:output:0lambda_layer_33_105491lambda_layer_33_105493lambda_layer_33_105495*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_lambda_layer_33_layer_call_and_return_conditional_losses_1053222)
'lambda_layer_33/StatefulPartitionedCallÑ
#OutputLayer/StatefulPartitionedCallStatefulPartitionedCall0lambda_layer_33/StatefulPartitionedCall:output:0outputlayer_105498outputlayer_105500*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_OutputLayer_layer_call_and_return_conditional_losses_1053602%
#OutputLayer/StatefulPartitionedCall
IdentityIdentity,OutputLayer/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^Layer1/StatefulPartitionedCall^Layer2/StatefulPartitionedCall^Layer3/StatefulPartitionedCall$^OutputLayer/StatefulPartitionedCall(^lambda_layer_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2@
Layer1/StatefulPartitionedCallLayer1/StatefulPartitionedCall2@
Layer2/StatefulPartitionedCallLayer2/StatefulPartitionedCall2@
Layer3/StatefulPartitionedCallLayer3/StatefulPartitionedCall2J
#OutputLayer/StatefulPartitionedCall#OutputLayer/StatefulPartitionedCall2R
'lambda_layer_33/StatefulPartitionedCall'lambda_layer_33/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


.__inference_sequential_33_layer_call_fn_105392
input_34
unknown:	¬
	unknown_0:	¬
	unknown_1:
¬¬
	unknown_2:	¬
	unknown_3:
¬¬
	unknown_4:	¬
	unknown_5:
¬¬
	unknown_6:	¬
	unknown_7:
	unknown_8:	¬
	unknown_9:
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinput_34unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_33_layer_call_and_return_conditional_losses_1053672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_34
À

I__inference_sequential_33_layer_call_and_return_conditional_losses_105367

inputs 
layer1_105229:	¬
layer1_105231:	¬!
layer2_105266:
¬¬
layer2_105268:	¬!
layer3_105303:
¬¬
layer3_105305:	¬*
lambda_layer_33_105323:
¬¬%
lambda_layer_33_105325:	¬$
lambda_layer_33_105327:%
outputlayer_105361:	¬ 
outputlayer_105363:
identity¢Layer1/StatefulPartitionedCall¢Layer2/StatefulPartitionedCall¢Layer3/StatefulPartitionedCall¢#OutputLayer/StatefulPartitionedCall¢'lambda_layer_33/StatefulPartitionedCall
Layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_105229layer1_105231*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Layer1_layer_call_and_return_conditional_losses_1052282 
Layer1/StatefulPartitionedCall°
Layer2/StatefulPartitionedCallStatefulPartitionedCall'Layer1/StatefulPartitionedCall:output:0layer2_105266layer2_105268*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Layer2_layer_call_and_return_conditional_losses_1052652 
Layer2/StatefulPartitionedCall°
Layer3/StatefulPartitionedCallStatefulPartitionedCall'Layer2/StatefulPartitionedCall:output:0layer3_105303layer3_105305*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Layer3_layer_call_and_return_conditional_losses_1053022 
Layer3/StatefulPartitionedCall÷
'lambda_layer_33/StatefulPartitionedCallStatefulPartitionedCall'Layer3/StatefulPartitionedCall:output:0lambda_layer_33_105323lambda_layer_33_105325lambda_layer_33_105327*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_lambda_layer_33_layer_call_and_return_conditional_losses_1053222)
'lambda_layer_33/StatefulPartitionedCallÑ
#OutputLayer/StatefulPartitionedCallStatefulPartitionedCall0lambda_layer_33/StatefulPartitionedCall:output:0outputlayer_105361outputlayer_105363*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_OutputLayer_layer_call_and_return_conditional_losses_1053602%
#OutputLayer/StatefulPartitionedCall
IdentityIdentity,OutputLayer/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^Layer1/StatefulPartitionedCall^Layer2/StatefulPartitionedCall^Layer3/StatefulPartitionedCall$^OutputLayer/StatefulPartitionedCall(^lambda_layer_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2@
Layer1/StatefulPartitionedCallLayer1/StatefulPartitionedCall2@
Layer2/StatefulPartitionedCallLayer2/StatefulPartitionedCall2@
Layer3/StatefulPartitionedCallLayer3/StatefulPartitionedCall2J
#OutputLayer/StatefulPartitionedCall#OutputLayer/StatefulPartitionedCall2R
'lambda_layer_33/StatefulPartitionedCall'lambda_layer_33/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


.__inference_sequential_33_layer_call_fn_105922

inputs
unknown:	¬
	unknown_0:	¬
	unknown_1:
¬¬
	unknown_2:	¬
	unknown_3:
¬¬
	unknown_4:	¬
	unknown_5:
¬¬
	unknown_6:	¬
	unknown_7:
	unknown_8:	¬
	unknown_9:
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_33_layer_call_and_return_conditional_losses_1053672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


,__inference_OutputLayer_layer_call_fn_106133

inputs
unknown:	¬
	unknown_0:
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_OutputLayer_layer_call_and_return_conditional_losses_1053602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Ä!
û
B__inference_Layer1_layer_call_and_return_conditional_losses_105228

inputs4
!tensordot_readvariableop_resource:	¬.
biasadd_readvariableop_resource:	¬
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	¬*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:¬2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tanhh
IdentityIdentityTanh:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


.__inference_sequential_33_layer_call_fn_105556
input_34
unknown:	¬
	unknown_0:	¬
	unknown_1:
¬¬
	unknown_2:	¬
	unknown_3:
¬¬
	unknown_4:	¬
	unknown_5:
¬¬
	unknown_6:	¬
	unknown_7:
	unknown_8:	¬
	unknown_9:
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinput_34unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_33_layer_call_and_return_conditional_losses_1055042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_34
¿
¹
0__inference_lambda_layer_33_layer_call_fn_106094

inputs
unknown:
¬¬
	unknown_0:	¬
	unknown_1:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_lambda_layer_33_layer_call_and_return_conditional_losses_1053222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ¬: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
V
ñ
__inference__traced_save_106276
file_prefix,
(savev2_layer1_kernel_read_readvariableop*
&savev2_layer1_bias_read_readvariableop,
(savev2_layer2_kernel_read_readvariableop*
&savev2_layer2_bias_read_readvariableop,
(savev2_layer3_kernel_read_readvariableop*
&savev2_layer3_bias_read_readvariableop5
1savev2_lambda_layer_33_kernel_read_readvariableop3
/savev2_lambda_layer_33_bias_read_readvariableop5
1savev2_lambda_layer_33_lambda_read_readvariableop1
-savev2_outputlayer_kernel_read_readvariableop/
+savev2_outputlayer_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_layer1_kernel_m_read_readvariableop1
-savev2_adam_layer1_bias_m_read_readvariableop3
/savev2_adam_layer2_kernel_m_read_readvariableop1
-savev2_adam_layer2_bias_m_read_readvariableop3
/savev2_adam_layer3_kernel_m_read_readvariableop1
-savev2_adam_layer3_bias_m_read_readvariableop<
8savev2_adam_lambda_layer_33_kernel_m_read_readvariableop:
6savev2_adam_lambda_layer_33_bias_m_read_readvariableop<
8savev2_adam_lambda_layer_33_lambda_m_read_readvariableop8
4savev2_adam_outputlayer_kernel_m_read_readvariableop6
2savev2_adam_outputlayer_bias_m_read_readvariableop3
/savev2_adam_layer1_kernel_v_read_readvariableop1
-savev2_adam_layer1_bias_v_read_readvariableop3
/savev2_adam_layer2_kernel_v_read_readvariableop1
-savev2_adam_layer2_bias_v_read_readvariableop3
/savev2_adam_layer3_kernel_v_read_readvariableop1
-savev2_adam_layer3_bias_v_read_readvariableop<
8savev2_adam_lambda_layer_33_kernel_v_read_readvariableop:
6savev2_adam_lambda_layer_33_bias_v_read_readvariableop<
8savev2_adam_lambda_layer_33_lambda_v_read_readvariableop8
4savev2_adam_outputlayer_kernel_v_read_readvariableop6
2savev2_adam_outputlayer_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÖ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*è
valueÞBÛ)B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-3/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-3/b/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/lbda/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-3/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-3/b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/lbda/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-3/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-3/b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/lbda/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÚ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÇ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_layer1_kernel_read_readvariableop&savev2_layer1_bias_read_readvariableop(savev2_layer2_kernel_read_readvariableop&savev2_layer2_bias_read_readvariableop(savev2_layer3_kernel_read_readvariableop&savev2_layer3_bias_read_readvariableop1savev2_lambda_layer_33_kernel_read_readvariableop/savev2_lambda_layer_33_bias_read_readvariableop1savev2_lambda_layer_33_lambda_read_readvariableop-savev2_outputlayer_kernel_read_readvariableop+savev2_outputlayer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_layer1_kernel_m_read_readvariableop-savev2_adam_layer1_bias_m_read_readvariableop/savev2_adam_layer2_kernel_m_read_readvariableop-savev2_adam_layer2_bias_m_read_readvariableop/savev2_adam_layer3_kernel_m_read_readvariableop-savev2_adam_layer3_bias_m_read_readvariableop8savev2_adam_lambda_layer_33_kernel_m_read_readvariableop6savev2_adam_lambda_layer_33_bias_m_read_readvariableop8savev2_adam_lambda_layer_33_lambda_m_read_readvariableop4savev2_adam_outputlayer_kernel_m_read_readvariableop2savev2_adam_outputlayer_bias_m_read_readvariableop/savev2_adam_layer1_kernel_v_read_readvariableop-savev2_adam_layer1_bias_v_read_readvariableop/savev2_adam_layer2_kernel_v_read_readvariableop-savev2_adam_layer2_bias_v_read_readvariableop/savev2_adam_layer3_kernel_v_read_readvariableop-savev2_adam_layer3_bias_v_read_readvariableop8savev2_adam_lambda_layer_33_kernel_v_read_readvariableop6savev2_adam_lambda_layer_33_bias_v_read_readvariableop8savev2_adam_lambda_layer_33_lambda_v_read_readvariableop4savev2_adam_outputlayer_kernel_v_read_readvariableop2savev2_adam_outputlayer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *7
dtypes-
+2)	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*Í
_input_shapes»
¸: :	¬:¬:
¬¬:¬:
¬¬:¬:
¬¬:¬::	¬:: : : : : : : :	¬:¬:
¬¬:¬:
¬¬:¬:
¬¬:¬::	¬::	¬:¬:
¬¬:¬:
¬¬:¬:
¬¬:¬::	¬:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	¬:!

_output_shapes	
:¬:&"
 
_output_shapes
:
¬¬:!

_output_shapes	
:¬:&"
 
_output_shapes
:
¬¬:!

_output_shapes	
:¬:&"
 
_output_shapes
:
¬¬:!

_output_shapes	
:¬: 	

_output_shapes
::%
!

_output_shapes
:	¬: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	¬:!

_output_shapes	
:¬:&"
 
_output_shapes
:
¬¬:!

_output_shapes	
:¬:&"
 
_output_shapes
:
¬¬:!

_output_shapes	
:¬:&"
 
_output_shapes
:
¬¬:!

_output_shapes	
:¬: 

_output_shapes
::%!

_output_shapes
:	¬: 

_output_shapes
::%!

_output_shapes
:	¬:!

_output_shapes	
:¬:& "
 
_output_shapes
:
¬¬:!!

_output_shapes	
:¬:&""
 
_output_shapes
:
¬¬:!#

_output_shapes	
:¬:&$"
 
_output_shapes
:
¬¬:!%

_output_shapes	
:¬: &

_output_shapes
::%'!

_output_shapes
:	¬: (

_output_shapes
::)

_output_shapes
: 
ã


$__inference_signature_wrapper_105653
input_34
unknown:	¬
	unknown_0:	¬
	unknown_1:
¬¬
	unknown_2:	¬
	unknown_3:
¬¬
	unknown_4:	¬
	unknown_5:
¬¬
	unknown_6:	¬
	unknown_7:
	unknown_8:	¬
	unknown_9:
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinput_34unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_1051902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_34
É!
ü
B__inference_Layer3_layer_call_and_return_conditional_losses_106060

inputs5
!tensordot_readvariableop_resource:
¬¬.
biasadd_readvariableop_resource:	¬
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
¬¬*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:¬2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tanhh
IdentityIdentityTanh:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
É!
ü
B__inference_Layer3_layer_call_and_return_conditional_losses_105302

inputs5
!tensordot_readvariableop_resource:
¬¬.
biasadd_readvariableop_resource:	¬
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
¬¬*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:¬2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tanhh
IdentityIdentityTanh:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Î­
â
"__inference__traced_restore_106406
file_prefix1
assignvariableop_layer1_kernel:	¬-
assignvariableop_1_layer1_bias:	¬4
 assignvariableop_2_layer2_kernel:
¬¬-
assignvariableop_3_layer2_bias:	¬4
 assignvariableop_4_layer3_kernel:
¬¬-
assignvariableop_5_layer3_bias:	¬=
)assignvariableop_6_lambda_layer_33_kernel:
¬¬6
'assignvariableop_7_lambda_layer_33_bias:	¬7
)assignvariableop_8_lambda_layer_33_lambda:8
%assignvariableop_9_outputlayer_kernel:	¬2
$assignvariableop_10_outputlayer_bias:'
assignvariableop_11_adam_iter:	 )
assignvariableop_12_adam_beta_1: )
assignvariableop_13_adam_beta_2: (
assignvariableop_14_adam_decay: 0
&assignvariableop_15_adam_learning_rate: #
assignvariableop_16_total: #
assignvariableop_17_count: ;
(assignvariableop_18_adam_layer1_kernel_m:	¬5
&assignvariableop_19_adam_layer1_bias_m:	¬<
(assignvariableop_20_adam_layer2_kernel_m:
¬¬5
&assignvariableop_21_adam_layer2_bias_m:	¬<
(assignvariableop_22_adam_layer3_kernel_m:
¬¬5
&assignvariableop_23_adam_layer3_bias_m:	¬E
1assignvariableop_24_adam_lambda_layer_33_kernel_m:
¬¬>
/assignvariableop_25_adam_lambda_layer_33_bias_m:	¬?
1assignvariableop_26_adam_lambda_layer_33_lambda_m:@
-assignvariableop_27_adam_outputlayer_kernel_m:	¬9
+assignvariableop_28_adam_outputlayer_bias_m:;
(assignvariableop_29_adam_layer1_kernel_v:	¬5
&assignvariableop_30_adam_layer1_bias_v:	¬<
(assignvariableop_31_adam_layer2_kernel_v:
¬¬5
&assignvariableop_32_adam_layer2_bias_v:	¬<
(assignvariableop_33_adam_layer3_kernel_v:
¬¬5
&assignvariableop_34_adam_layer3_bias_v:	¬E
1assignvariableop_35_adam_lambda_layer_33_kernel_v:
¬¬>
/assignvariableop_36_adam_lambda_layer_33_bias_v:	¬?
1assignvariableop_37_adam_lambda_layer_33_lambda_v:@
-assignvariableop_38_adam_outputlayer_kernel_v:	¬9
+assignvariableop_39_adam_outputlayer_bias_v:
identity_41¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ü
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*è
valueÞBÛ)B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-3/w/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-3/b/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/lbda/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-3/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-3/b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/lbda/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-3/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-3/b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/lbda/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesà
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesû
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*º
_output_shapes§
¤:::::::::::::::::::::::::::::::::::::::::*7
dtypes-
+2)	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_layer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¥
AssignVariableOp_2AssignVariableOp assignvariableop_2_layer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_layer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¥
AssignVariableOp_4AssignVariableOp assignvariableop_4_layer3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5£
AssignVariableOp_5AssignVariableOpassignvariableop_5_layer3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6®
AssignVariableOp_6AssignVariableOp)assignvariableop_6_lambda_layer_33_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¬
AssignVariableOp_7AssignVariableOp'assignvariableop_7_lambda_layer_33_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8®
AssignVariableOp_8AssignVariableOp)assignvariableop_8_lambda_layer_33_lambdaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ª
AssignVariableOp_9AssignVariableOp%assignvariableop_9_outputlayer_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¬
AssignVariableOp_10AssignVariableOp$assignvariableop_10_outputlayer_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_11¥
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_iterIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12§
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13§
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¦
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_decayIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15®
AssignVariableOp_15AssignVariableOp&assignvariableop_15_adam_learning_rateIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¡
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¡
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18°
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_layer1_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19®
AssignVariableOp_19AssignVariableOp&assignvariableop_19_adam_layer1_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20°
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_layer2_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21®
AssignVariableOp_21AssignVariableOp&assignvariableop_21_adam_layer2_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22°
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_layer3_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23®
AssignVariableOp_23AssignVariableOp&assignvariableop_23_adam_layer3_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¹
AssignVariableOp_24AssignVariableOp1assignvariableop_24_adam_lambda_layer_33_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25·
AssignVariableOp_25AssignVariableOp/assignvariableop_25_adam_lambda_layer_33_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¹
AssignVariableOp_26AssignVariableOp1assignvariableop_26_adam_lambda_layer_33_lambda_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27µ
AssignVariableOp_27AssignVariableOp-assignvariableop_27_adam_outputlayer_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28³
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_outputlayer_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29°
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_layer1_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30®
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_layer1_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31°
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_layer2_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32®
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_layer2_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33°
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_layer3_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34®
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_layer3_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¹
AssignVariableOp_35AssignVariableOp1assignvariableop_35_adam_lambda_layer_33_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36·
AssignVariableOp_36AssignVariableOp/assignvariableop_36_adam_lambda_layer_33_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37¹
AssignVariableOp_37AssignVariableOp1assignvariableop_37_adam_lambda_layer_33_lambda_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38µ
AssignVariableOp_38AssignVariableOp-assignvariableop_38_adam_outputlayer_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39³
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_outputlayer_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_399
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÎ
Identity_40Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_40f
Identity_41IdentityIdentity_40:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_41¶
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_41Identity_41:output:0*e
_input_shapesT
R: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
À 
	
I__inference_sequential_33_layer_call_and_return_conditional_losses_105895

inputs;
(layer1_tensordot_readvariableop_resource:	¬5
&layer1_biasadd_readvariableop_resource:	¬<
(layer2_tensordot_readvariableop_resource:
¬¬5
&layer2_biasadd_readvariableop_resource:	¬<
(layer3_tensordot_readvariableop_resource:
¬¬5
&layer3_biasadd_readvariableop_resource:	¬B
.lambda_layer_33_matmul_readvariableop_resource:
¬¬:
+lambda_layer_33_add_readvariableop_resource:	¬9
+lambda_layer_33_sub_readvariableop_resource:@
-outputlayer_tensordot_readvariableop_resource:	¬9
+outputlayer_biasadd_readvariableop_resource:
identity¢Layer1/BiasAdd/ReadVariableOp¢Layer1/Tensordot/ReadVariableOp¢Layer2/BiasAdd/ReadVariableOp¢Layer2/Tensordot/ReadVariableOp¢Layer3/BiasAdd/ReadVariableOp¢Layer3/Tensordot/ReadVariableOp¢"OutputLayer/BiasAdd/ReadVariableOp¢$OutputLayer/Tensordot/ReadVariableOp¢%lambda_layer_33/MatMul/ReadVariableOp¢"lambda_layer_33/add/ReadVariableOp¢"lambda_layer_33/sub/ReadVariableOp¬
Layer1/Tensordot/ReadVariableOpReadVariableOp(layer1_tensordot_readvariableop_resource*
_output_shapes
:	¬*
dtype02!
Layer1/Tensordot/ReadVariableOpx
Layer1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Layer1/Tensordot/axes
Layer1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Layer1/Tensordot/freef
Layer1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Layer1/Tensordot/Shape
Layer1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
Layer1/Tensordot/GatherV2/axisô
Layer1/Tensordot/GatherV2GatherV2Layer1/Tensordot/Shape:output:0Layer1/Tensordot/free:output:0'Layer1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Layer1/Tensordot/GatherV2
 Layer1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Layer1/Tensordot/GatherV2_1/axisú
Layer1/Tensordot/GatherV2_1GatherV2Layer1/Tensordot/Shape:output:0Layer1/Tensordot/axes:output:0)Layer1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Layer1/Tensordot/GatherV2_1z
Layer1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Layer1/Tensordot/Const
Layer1/Tensordot/ProdProd"Layer1/Tensordot/GatherV2:output:0Layer1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Layer1/Tensordot/Prod~
Layer1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Layer1/Tensordot/Const_1¤
Layer1/Tensordot/Prod_1Prod$Layer1/Tensordot/GatherV2_1:output:0!Layer1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Layer1/Tensordot/Prod_1~
Layer1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Layer1/Tensordot/concat/axisÓ
Layer1/Tensordot/concatConcatV2Layer1/Tensordot/free:output:0Layer1/Tensordot/axes:output:0%Layer1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Layer1/Tensordot/concat¨
Layer1/Tensordot/stackPackLayer1/Tensordot/Prod:output:0 Layer1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Layer1/Tensordot/stack¥
Layer1/Tensordot/transpose	Transposeinputs Layer1/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Layer1/Tensordot/transpose»
Layer1/Tensordot/ReshapeReshapeLayer1/Tensordot/transpose:y:0Layer1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Layer1/Tensordot/Reshape»
Layer1/Tensordot/MatMulMatMul!Layer1/Tensordot/Reshape:output:0'Layer1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer1/Tensordot/MatMul
Layer1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:¬2
Layer1/Tensordot/Const_2
Layer1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
Layer1/Tensordot/concat_1/axisà
Layer1/Tensordot/concat_1ConcatV2"Layer1/Tensordot/GatherV2:output:0!Layer1/Tensordot/Const_2:output:0'Layer1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Layer1/Tensordot/concat_1­
Layer1/TensordotReshape!Layer1/Tensordot/MatMul:product:0"Layer1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer1/Tensordot¢
Layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
Layer1/BiasAdd/ReadVariableOp¤
Layer1/BiasAddBiasAddLayer1/Tensordot:output:0%Layer1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer1/BiasAddr
Layer1/TanhTanhLayer1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer1/Tanh­
Layer2/Tensordot/ReadVariableOpReadVariableOp(layer2_tensordot_readvariableop_resource* 
_output_shapes
:
¬¬*
dtype02!
Layer2/Tensordot/ReadVariableOpx
Layer2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Layer2/Tensordot/axes
Layer2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Layer2/Tensordot/freeo
Layer2/Tensordot/ShapeShapeLayer1/Tanh:y:0*
T0*
_output_shapes
:2
Layer2/Tensordot/Shape
Layer2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
Layer2/Tensordot/GatherV2/axisô
Layer2/Tensordot/GatherV2GatherV2Layer2/Tensordot/Shape:output:0Layer2/Tensordot/free:output:0'Layer2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Layer2/Tensordot/GatherV2
 Layer2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Layer2/Tensordot/GatherV2_1/axisú
Layer2/Tensordot/GatherV2_1GatherV2Layer2/Tensordot/Shape:output:0Layer2/Tensordot/axes:output:0)Layer2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Layer2/Tensordot/GatherV2_1z
Layer2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Layer2/Tensordot/Const
Layer2/Tensordot/ProdProd"Layer2/Tensordot/GatherV2:output:0Layer2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Layer2/Tensordot/Prod~
Layer2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Layer2/Tensordot/Const_1¤
Layer2/Tensordot/Prod_1Prod$Layer2/Tensordot/GatherV2_1:output:0!Layer2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Layer2/Tensordot/Prod_1~
Layer2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Layer2/Tensordot/concat/axisÓ
Layer2/Tensordot/concatConcatV2Layer2/Tensordot/free:output:0Layer2/Tensordot/axes:output:0%Layer2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Layer2/Tensordot/concat¨
Layer2/Tensordot/stackPackLayer2/Tensordot/Prod:output:0 Layer2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Layer2/Tensordot/stack¯
Layer2/Tensordot/transpose	TransposeLayer1/Tanh:y:0 Layer2/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer2/Tensordot/transpose»
Layer2/Tensordot/ReshapeReshapeLayer2/Tensordot/transpose:y:0Layer2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Layer2/Tensordot/Reshape»
Layer2/Tensordot/MatMulMatMul!Layer2/Tensordot/Reshape:output:0'Layer2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer2/Tensordot/MatMul
Layer2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:¬2
Layer2/Tensordot/Const_2
Layer2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
Layer2/Tensordot/concat_1/axisà
Layer2/Tensordot/concat_1ConcatV2"Layer2/Tensordot/GatherV2:output:0!Layer2/Tensordot/Const_2:output:0'Layer2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Layer2/Tensordot/concat_1­
Layer2/TensordotReshape!Layer2/Tensordot/MatMul:product:0"Layer2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer2/Tensordot¢
Layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
Layer2/BiasAdd/ReadVariableOp¤
Layer2/BiasAddBiasAddLayer2/Tensordot:output:0%Layer2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer2/BiasAddr
Layer2/TanhTanhLayer2/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer2/Tanh­
Layer3/Tensordot/ReadVariableOpReadVariableOp(layer3_tensordot_readvariableop_resource* 
_output_shapes
:
¬¬*
dtype02!
Layer3/Tensordot/ReadVariableOpx
Layer3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Layer3/Tensordot/axes
Layer3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Layer3/Tensordot/freeo
Layer3/Tensordot/ShapeShapeLayer2/Tanh:y:0*
T0*
_output_shapes
:2
Layer3/Tensordot/Shape
Layer3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
Layer3/Tensordot/GatherV2/axisô
Layer3/Tensordot/GatherV2GatherV2Layer3/Tensordot/Shape:output:0Layer3/Tensordot/free:output:0'Layer3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Layer3/Tensordot/GatherV2
 Layer3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Layer3/Tensordot/GatherV2_1/axisú
Layer3/Tensordot/GatherV2_1GatherV2Layer3/Tensordot/Shape:output:0Layer3/Tensordot/axes:output:0)Layer3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Layer3/Tensordot/GatherV2_1z
Layer3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Layer3/Tensordot/Const
Layer3/Tensordot/ProdProd"Layer3/Tensordot/GatherV2:output:0Layer3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Layer3/Tensordot/Prod~
Layer3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Layer3/Tensordot/Const_1¤
Layer3/Tensordot/Prod_1Prod$Layer3/Tensordot/GatherV2_1:output:0!Layer3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Layer3/Tensordot/Prod_1~
Layer3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Layer3/Tensordot/concat/axisÓ
Layer3/Tensordot/concatConcatV2Layer3/Tensordot/free:output:0Layer3/Tensordot/axes:output:0%Layer3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Layer3/Tensordot/concat¨
Layer3/Tensordot/stackPackLayer3/Tensordot/Prod:output:0 Layer3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Layer3/Tensordot/stack¯
Layer3/Tensordot/transpose	TransposeLayer2/Tanh:y:0 Layer3/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer3/Tensordot/transpose»
Layer3/Tensordot/ReshapeReshapeLayer3/Tensordot/transpose:y:0Layer3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Layer3/Tensordot/Reshape»
Layer3/Tensordot/MatMulMatMul!Layer3/Tensordot/Reshape:output:0'Layer3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer3/Tensordot/MatMul
Layer3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:¬2
Layer3/Tensordot/Const_2
Layer3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
Layer3/Tensordot/concat_1/axisà
Layer3/Tensordot/concat_1ConcatV2"Layer3/Tensordot/GatherV2:output:0!Layer3/Tensordot/Const_2:output:0'Layer3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Layer3/Tensordot/concat_1­
Layer3/TensordotReshape!Layer3/Tensordot/MatMul:product:0"Layer3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer3/Tensordot¢
Layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
Layer3/BiasAdd/ReadVariableOp¤
Layer3/BiasAddBiasAddLayer3/Tensordot:output:0%Layer3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer3/BiasAddr
Layer3/TanhTanhLayer3/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer3/Tanh¿
%lambda_layer_33/MatMul/ReadVariableOpReadVariableOp.lambda_layer_33_matmul_readvariableop_resource* 
_output_shapes
:
¬¬*
dtype02'
%lambda_layer_33/MatMul/ReadVariableOp¸
lambda_layer_33/MatMulBatchMatMulV2Layer3/Tanh:y:0-lambda_layer_33/MatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lambda_layer_33/MatMul±
"lambda_layer_33/add/ReadVariableOpReadVariableOp+lambda_layer_33_add_readvariableop_resource*
_output_shapes	
:¬*
dtype02$
"lambda_layer_33/add/ReadVariableOp·
lambda_layer_33/addAddV2lambda_layer_33/MatMul:output:0*lambda_layer_33/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lambda_layer_33/add°
"lambda_layer_33/sub/ReadVariableOpReadVariableOp+lambda_layer_33_sub_readvariableop_resource*
_output_shapes
:*
dtype02$
"lambda_layer_33/sub/ReadVariableOp­
lambda_layer_33/subSublambda_layer_33/add:z:0*lambda_layer_33/sub/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lambda_layer_33/sub
lambda_layer_33/TanhTanhlambda_layer_33/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lambda_layer_33/Tanh»
$OutputLayer/Tensordot/ReadVariableOpReadVariableOp-outputlayer_tensordot_readvariableop_resource*
_output_shapes
:	¬*
dtype02&
$OutputLayer/Tensordot/ReadVariableOp
OutputLayer/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
OutputLayer/Tensordot/axes
OutputLayer/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
OutputLayer/Tensordot/free
OutputLayer/Tensordot/ShapeShapelambda_layer_33/Tanh:y:0*
T0*
_output_shapes
:2
OutputLayer/Tensordot/Shape
#OutputLayer/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#OutputLayer/Tensordot/GatherV2/axis
OutputLayer/Tensordot/GatherV2GatherV2$OutputLayer/Tensordot/Shape:output:0#OutputLayer/Tensordot/free:output:0,OutputLayer/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
OutputLayer/Tensordot/GatherV2
%OutputLayer/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%OutputLayer/Tensordot/GatherV2_1/axis
 OutputLayer/Tensordot/GatherV2_1GatherV2$OutputLayer/Tensordot/Shape:output:0#OutputLayer/Tensordot/axes:output:0.OutputLayer/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2"
 OutputLayer/Tensordot/GatherV2_1
OutputLayer/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
OutputLayer/Tensordot/Const°
OutputLayer/Tensordot/ProdProd'OutputLayer/Tensordot/GatherV2:output:0$OutputLayer/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
OutputLayer/Tensordot/Prod
OutputLayer/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
OutputLayer/Tensordot/Const_1¸
OutputLayer/Tensordot/Prod_1Prod)OutputLayer/Tensordot/GatherV2_1:output:0&OutputLayer/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
OutputLayer/Tensordot/Prod_1
!OutputLayer/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!OutputLayer/Tensordot/concat/axisì
OutputLayer/Tensordot/concatConcatV2#OutputLayer/Tensordot/free:output:0#OutputLayer/Tensordot/axes:output:0*OutputLayer/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
OutputLayer/Tensordot/concat¼
OutputLayer/Tensordot/stackPack#OutputLayer/Tensordot/Prod:output:0%OutputLayer/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
OutputLayer/Tensordot/stackÇ
OutputLayer/Tensordot/transpose	Transposelambda_layer_33/Tanh:y:0%OutputLayer/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
OutputLayer/Tensordot/transposeÏ
OutputLayer/Tensordot/ReshapeReshape#OutputLayer/Tensordot/transpose:y:0$OutputLayer/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
OutputLayer/Tensordot/ReshapeÎ
OutputLayer/Tensordot/MatMulMatMul&OutputLayer/Tensordot/Reshape:output:0,OutputLayer/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
OutputLayer/Tensordot/MatMul
OutputLayer/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
OutputLayer/Tensordot/Const_2
#OutputLayer/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#OutputLayer/Tensordot/concat_1/axisù
OutputLayer/Tensordot/concat_1ConcatV2'OutputLayer/Tensordot/GatherV2:output:0&OutputLayer/Tensordot/Const_2:output:0,OutputLayer/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2 
OutputLayer/Tensordot/concat_1À
OutputLayer/TensordotReshape&OutputLayer/Tensordot/MatMul:product:0'OutputLayer/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
OutputLayer/Tensordot°
"OutputLayer/BiasAdd/ReadVariableOpReadVariableOp+outputlayer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"OutputLayer/BiasAdd/ReadVariableOp·
OutputLayer/BiasAddBiasAddOutputLayer/Tensordot:output:0*OutputLayer/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
OutputLayer/BiasAdd{
IdentityIdentityOutputLayer/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÒ
NoOpNoOp^Layer1/BiasAdd/ReadVariableOp ^Layer1/Tensordot/ReadVariableOp^Layer2/BiasAdd/ReadVariableOp ^Layer2/Tensordot/ReadVariableOp^Layer3/BiasAdd/ReadVariableOp ^Layer3/Tensordot/ReadVariableOp#^OutputLayer/BiasAdd/ReadVariableOp%^OutputLayer/Tensordot/ReadVariableOp&^lambda_layer_33/MatMul/ReadVariableOp#^lambda_layer_33/add/ReadVariableOp#^lambda_layer_33/sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2>
Layer1/BiasAdd/ReadVariableOpLayer1/BiasAdd/ReadVariableOp2B
Layer1/Tensordot/ReadVariableOpLayer1/Tensordot/ReadVariableOp2>
Layer2/BiasAdd/ReadVariableOpLayer2/BiasAdd/ReadVariableOp2B
Layer2/Tensordot/ReadVariableOpLayer2/Tensordot/ReadVariableOp2>
Layer3/BiasAdd/ReadVariableOpLayer3/BiasAdd/ReadVariableOp2B
Layer3/Tensordot/ReadVariableOpLayer3/Tensordot/ReadVariableOp2H
"OutputLayer/BiasAdd/ReadVariableOp"OutputLayer/BiasAdd/ReadVariableOp2L
$OutputLayer/Tensordot/ReadVariableOp$OutputLayer/Tensordot/ReadVariableOp2N
%lambda_layer_33/MatMul/ReadVariableOp%lambda_layer_33/MatMul/ReadVariableOp2H
"lambda_layer_33/add/ReadVariableOp"lambda_layer_33/add/ReadVariableOp2H
"lambda_layer_33/sub/ReadVariableOp"lambda_layer_33/sub/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä!
û
B__inference_Layer1_layer_call_and_return_conditional_losses_105980

inputs4
!tensordot_readvariableop_resource:	¬.
biasadd_readvariableop_resource:	¬
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	¬*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:¬2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tanhh
IdentityIdentityTanh:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À 
	
I__inference_sequential_33_layer_call_and_return_conditional_losses_105774

inputs;
(layer1_tensordot_readvariableop_resource:	¬5
&layer1_biasadd_readvariableop_resource:	¬<
(layer2_tensordot_readvariableop_resource:
¬¬5
&layer2_biasadd_readvariableop_resource:	¬<
(layer3_tensordot_readvariableop_resource:
¬¬5
&layer3_biasadd_readvariableop_resource:	¬B
.lambda_layer_33_matmul_readvariableop_resource:
¬¬:
+lambda_layer_33_add_readvariableop_resource:	¬9
+lambda_layer_33_sub_readvariableop_resource:@
-outputlayer_tensordot_readvariableop_resource:	¬9
+outputlayer_biasadd_readvariableop_resource:
identity¢Layer1/BiasAdd/ReadVariableOp¢Layer1/Tensordot/ReadVariableOp¢Layer2/BiasAdd/ReadVariableOp¢Layer2/Tensordot/ReadVariableOp¢Layer3/BiasAdd/ReadVariableOp¢Layer3/Tensordot/ReadVariableOp¢"OutputLayer/BiasAdd/ReadVariableOp¢$OutputLayer/Tensordot/ReadVariableOp¢%lambda_layer_33/MatMul/ReadVariableOp¢"lambda_layer_33/add/ReadVariableOp¢"lambda_layer_33/sub/ReadVariableOp¬
Layer1/Tensordot/ReadVariableOpReadVariableOp(layer1_tensordot_readvariableop_resource*
_output_shapes
:	¬*
dtype02!
Layer1/Tensordot/ReadVariableOpx
Layer1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Layer1/Tensordot/axes
Layer1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Layer1/Tensordot/freef
Layer1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Layer1/Tensordot/Shape
Layer1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
Layer1/Tensordot/GatherV2/axisô
Layer1/Tensordot/GatherV2GatherV2Layer1/Tensordot/Shape:output:0Layer1/Tensordot/free:output:0'Layer1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Layer1/Tensordot/GatherV2
 Layer1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Layer1/Tensordot/GatherV2_1/axisú
Layer1/Tensordot/GatherV2_1GatherV2Layer1/Tensordot/Shape:output:0Layer1/Tensordot/axes:output:0)Layer1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Layer1/Tensordot/GatherV2_1z
Layer1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Layer1/Tensordot/Const
Layer1/Tensordot/ProdProd"Layer1/Tensordot/GatherV2:output:0Layer1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Layer1/Tensordot/Prod~
Layer1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Layer1/Tensordot/Const_1¤
Layer1/Tensordot/Prod_1Prod$Layer1/Tensordot/GatherV2_1:output:0!Layer1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Layer1/Tensordot/Prod_1~
Layer1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Layer1/Tensordot/concat/axisÓ
Layer1/Tensordot/concatConcatV2Layer1/Tensordot/free:output:0Layer1/Tensordot/axes:output:0%Layer1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Layer1/Tensordot/concat¨
Layer1/Tensordot/stackPackLayer1/Tensordot/Prod:output:0 Layer1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Layer1/Tensordot/stack¥
Layer1/Tensordot/transpose	Transposeinputs Layer1/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Layer1/Tensordot/transpose»
Layer1/Tensordot/ReshapeReshapeLayer1/Tensordot/transpose:y:0Layer1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Layer1/Tensordot/Reshape»
Layer1/Tensordot/MatMulMatMul!Layer1/Tensordot/Reshape:output:0'Layer1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer1/Tensordot/MatMul
Layer1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:¬2
Layer1/Tensordot/Const_2
Layer1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
Layer1/Tensordot/concat_1/axisà
Layer1/Tensordot/concat_1ConcatV2"Layer1/Tensordot/GatherV2:output:0!Layer1/Tensordot/Const_2:output:0'Layer1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Layer1/Tensordot/concat_1­
Layer1/TensordotReshape!Layer1/Tensordot/MatMul:product:0"Layer1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer1/Tensordot¢
Layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
Layer1/BiasAdd/ReadVariableOp¤
Layer1/BiasAddBiasAddLayer1/Tensordot:output:0%Layer1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer1/BiasAddr
Layer1/TanhTanhLayer1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer1/Tanh­
Layer2/Tensordot/ReadVariableOpReadVariableOp(layer2_tensordot_readvariableop_resource* 
_output_shapes
:
¬¬*
dtype02!
Layer2/Tensordot/ReadVariableOpx
Layer2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Layer2/Tensordot/axes
Layer2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Layer2/Tensordot/freeo
Layer2/Tensordot/ShapeShapeLayer1/Tanh:y:0*
T0*
_output_shapes
:2
Layer2/Tensordot/Shape
Layer2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
Layer2/Tensordot/GatherV2/axisô
Layer2/Tensordot/GatherV2GatherV2Layer2/Tensordot/Shape:output:0Layer2/Tensordot/free:output:0'Layer2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Layer2/Tensordot/GatherV2
 Layer2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Layer2/Tensordot/GatherV2_1/axisú
Layer2/Tensordot/GatherV2_1GatherV2Layer2/Tensordot/Shape:output:0Layer2/Tensordot/axes:output:0)Layer2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Layer2/Tensordot/GatherV2_1z
Layer2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Layer2/Tensordot/Const
Layer2/Tensordot/ProdProd"Layer2/Tensordot/GatherV2:output:0Layer2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Layer2/Tensordot/Prod~
Layer2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Layer2/Tensordot/Const_1¤
Layer2/Tensordot/Prod_1Prod$Layer2/Tensordot/GatherV2_1:output:0!Layer2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Layer2/Tensordot/Prod_1~
Layer2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Layer2/Tensordot/concat/axisÓ
Layer2/Tensordot/concatConcatV2Layer2/Tensordot/free:output:0Layer2/Tensordot/axes:output:0%Layer2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Layer2/Tensordot/concat¨
Layer2/Tensordot/stackPackLayer2/Tensordot/Prod:output:0 Layer2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Layer2/Tensordot/stack¯
Layer2/Tensordot/transpose	TransposeLayer1/Tanh:y:0 Layer2/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer2/Tensordot/transpose»
Layer2/Tensordot/ReshapeReshapeLayer2/Tensordot/transpose:y:0Layer2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Layer2/Tensordot/Reshape»
Layer2/Tensordot/MatMulMatMul!Layer2/Tensordot/Reshape:output:0'Layer2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer2/Tensordot/MatMul
Layer2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:¬2
Layer2/Tensordot/Const_2
Layer2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
Layer2/Tensordot/concat_1/axisà
Layer2/Tensordot/concat_1ConcatV2"Layer2/Tensordot/GatherV2:output:0!Layer2/Tensordot/Const_2:output:0'Layer2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Layer2/Tensordot/concat_1­
Layer2/TensordotReshape!Layer2/Tensordot/MatMul:product:0"Layer2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer2/Tensordot¢
Layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
Layer2/BiasAdd/ReadVariableOp¤
Layer2/BiasAddBiasAddLayer2/Tensordot:output:0%Layer2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer2/BiasAddr
Layer2/TanhTanhLayer2/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer2/Tanh­
Layer3/Tensordot/ReadVariableOpReadVariableOp(layer3_tensordot_readvariableop_resource* 
_output_shapes
:
¬¬*
dtype02!
Layer3/Tensordot/ReadVariableOpx
Layer3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Layer3/Tensordot/axes
Layer3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Layer3/Tensordot/freeo
Layer3/Tensordot/ShapeShapeLayer2/Tanh:y:0*
T0*
_output_shapes
:2
Layer3/Tensordot/Shape
Layer3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
Layer3/Tensordot/GatherV2/axisô
Layer3/Tensordot/GatherV2GatherV2Layer3/Tensordot/Shape:output:0Layer3/Tensordot/free:output:0'Layer3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Layer3/Tensordot/GatherV2
 Layer3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Layer3/Tensordot/GatherV2_1/axisú
Layer3/Tensordot/GatherV2_1GatherV2Layer3/Tensordot/Shape:output:0Layer3/Tensordot/axes:output:0)Layer3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Layer3/Tensordot/GatherV2_1z
Layer3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Layer3/Tensordot/Const
Layer3/Tensordot/ProdProd"Layer3/Tensordot/GatherV2:output:0Layer3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Layer3/Tensordot/Prod~
Layer3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Layer3/Tensordot/Const_1¤
Layer3/Tensordot/Prod_1Prod$Layer3/Tensordot/GatherV2_1:output:0!Layer3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Layer3/Tensordot/Prod_1~
Layer3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Layer3/Tensordot/concat/axisÓ
Layer3/Tensordot/concatConcatV2Layer3/Tensordot/free:output:0Layer3/Tensordot/axes:output:0%Layer3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Layer3/Tensordot/concat¨
Layer3/Tensordot/stackPackLayer3/Tensordot/Prod:output:0 Layer3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Layer3/Tensordot/stack¯
Layer3/Tensordot/transpose	TransposeLayer2/Tanh:y:0 Layer3/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer3/Tensordot/transpose»
Layer3/Tensordot/ReshapeReshapeLayer3/Tensordot/transpose:y:0Layer3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Layer3/Tensordot/Reshape»
Layer3/Tensordot/MatMulMatMul!Layer3/Tensordot/Reshape:output:0'Layer3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer3/Tensordot/MatMul
Layer3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:¬2
Layer3/Tensordot/Const_2
Layer3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
Layer3/Tensordot/concat_1/axisà
Layer3/Tensordot/concat_1ConcatV2"Layer3/Tensordot/GatherV2:output:0!Layer3/Tensordot/Const_2:output:0'Layer3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Layer3/Tensordot/concat_1­
Layer3/TensordotReshape!Layer3/Tensordot/MatMul:product:0"Layer3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer3/Tensordot¢
Layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
Layer3/BiasAdd/ReadVariableOp¤
Layer3/BiasAddBiasAddLayer3/Tensordot:output:0%Layer3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer3/BiasAddr
Layer3/TanhTanhLayer3/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Layer3/Tanh¿
%lambda_layer_33/MatMul/ReadVariableOpReadVariableOp.lambda_layer_33_matmul_readvariableop_resource* 
_output_shapes
:
¬¬*
dtype02'
%lambda_layer_33/MatMul/ReadVariableOp¸
lambda_layer_33/MatMulBatchMatMulV2Layer3/Tanh:y:0-lambda_layer_33/MatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lambda_layer_33/MatMul±
"lambda_layer_33/add/ReadVariableOpReadVariableOp+lambda_layer_33_add_readvariableop_resource*
_output_shapes	
:¬*
dtype02$
"lambda_layer_33/add/ReadVariableOp·
lambda_layer_33/addAddV2lambda_layer_33/MatMul:output:0*lambda_layer_33/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lambda_layer_33/add°
"lambda_layer_33/sub/ReadVariableOpReadVariableOp+lambda_layer_33_sub_readvariableop_resource*
_output_shapes
:*
dtype02$
"lambda_layer_33/sub/ReadVariableOp­
lambda_layer_33/subSublambda_layer_33/add:z:0*lambda_layer_33/sub/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lambda_layer_33/sub
lambda_layer_33/TanhTanhlambda_layer_33/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lambda_layer_33/Tanh»
$OutputLayer/Tensordot/ReadVariableOpReadVariableOp-outputlayer_tensordot_readvariableop_resource*
_output_shapes
:	¬*
dtype02&
$OutputLayer/Tensordot/ReadVariableOp
OutputLayer/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
OutputLayer/Tensordot/axes
OutputLayer/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
OutputLayer/Tensordot/free
OutputLayer/Tensordot/ShapeShapelambda_layer_33/Tanh:y:0*
T0*
_output_shapes
:2
OutputLayer/Tensordot/Shape
#OutputLayer/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#OutputLayer/Tensordot/GatherV2/axis
OutputLayer/Tensordot/GatherV2GatherV2$OutputLayer/Tensordot/Shape:output:0#OutputLayer/Tensordot/free:output:0,OutputLayer/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
OutputLayer/Tensordot/GatherV2
%OutputLayer/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%OutputLayer/Tensordot/GatherV2_1/axis
 OutputLayer/Tensordot/GatherV2_1GatherV2$OutputLayer/Tensordot/Shape:output:0#OutputLayer/Tensordot/axes:output:0.OutputLayer/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2"
 OutputLayer/Tensordot/GatherV2_1
OutputLayer/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
OutputLayer/Tensordot/Const°
OutputLayer/Tensordot/ProdProd'OutputLayer/Tensordot/GatherV2:output:0$OutputLayer/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
OutputLayer/Tensordot/Prod
OutputLayer/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
OutputLayer/Tensordot/Const_1¸
OutputLayer/Tensordot/Prod_1Prod)OutputLayer/Tensordot/GatherV2_1:output:0&OutputLayer/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
OutputLayer/Tensordot/Prod_1
!OutputLayer/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!OutputLayer/Tensordot/concat/axisì
OutputLayer/Tensordot/concatConcatV2#OutputLayer/Tensordot/free:output:0#OutputLayer/Tensordot/axes:output:0*OutputLayer/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
OutputLayer/Tensordot/concat¼
OutputLayer/Tensordot/stackPack#OutputLayer/Tensordot/Prod:output:0%OutputLayer/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
OutputLayer/Tensordot/stackÇ
OutputLayer/Tensordot/transpose	Transposelambda_layer_33/Tanh:y:0%OutputLayer/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
OutputLayer/Tensordot/transposeÏ
OutputLayer/Tensordot/ReshapeReshape#OutputLayer/Tensordot/transpose:y:0$OutputLayer/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
OutputLayer/Tensordot/ReshapeÎ
OutputLayer/Tensordot/MatMulMatMul&OutputLayer/Tensordot/Reshape:output:0,OutputLayer/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
OutputLayer/Tensordot/MatMul
OutputLayer/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
OutputLayer/Tensordot/Const_2
#OutputLayer/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#OutputLayer/Tensordot/concat_1/axisù
OutputLayer/Tensordot/concat_1ConcatV2'OutputLayer/Tensordot/GatherV2:output:0&OutputLayer/Tensordot/Const_2:output:0,OutputLayer/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2 
OutputLayer/Tensordot/concat_1À
OutputLayer/TensordotReshape&OutputLayer/Tensordot/MatMul:product:0'OutputLayer/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
OutputLayer/Tensordot°
"OutputLayer/BiasAdd/ReadVariableOpReadVariableOp+outputlayer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"OutputLayer/BiasAdd/ReadVariableOp·
OutputLayer/BiasAddBiasAddOutputLayer/Tensordot:output:0*OutputLayer/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
OutputLayer/BiasAdd{
IdentityIdentityOutputLayer/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÒ
NoOpNoOp^Layer1/BiasAdd/ReadVariableOp ^Layer1/Tensordot/ReadVariableOp^Layer2/BiasAdd/ReadVariableOp ^Layer2/Tensordot/ReadVariableOp^Layer3/BiasAdd/ReadVariableOp ^Layer3/Tensordot/ReadVariableOp#^OutputLayer/BiasAdd/ReadVariableOp%^OutputLayer/Tensordot/ReadVariableOp&^lambda_layer_33/MatMul/ReadVariableOp#^lambda_layer_33/add/ReadVariableOp#^lambda_layer_33/sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2>
Layer1/BiasAdd/ReadVariableOpLayer1/BiasAdd/ReadVariableOp2B
Layer1/Tensordot/ReadVariableOpLayer1/Tensordot/ReadVariableOp2>
Layer2/BiasAdd/ReadVariableOpLayer2/BiasAdd/ReadVariableOp2B
Layer2/Tensordot/ReadVariableOpLayer2/Tensordot/ReadVariableOp2>
Layer3/BiasAdd/ReadVariableOpLayer3/BiasAdd/ReadVariableOp2B
Layer3/Tensordot/ReadVariableOpLayer3/Tensordot/ReadVariableOp2H
"OutputLayer/BiasAdd/ReadVariableOp"OutputLayer/BiasAdd/ReadVariableOp2L
$OutputLayer/Tensordot/ReadVariableOp$OutputLayer/Tensordot/ReadVariableOp2N
%lambda_layer_33/MatMul/ReadVariableOp%lambda_layer_33/MatMul/ReadVariableOp2H
"lambda_layer_33/add/ReadVariableOp"lambda_layer_33/add/ReadVariableOp2H
"lambda_layer_33/sub/ReadVariableOp"lambda_layer_33/sub/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


'__inference_Layer2_layer_call_fn_106029

inputs
unknown:
¬¬
	unknown_0:	¬
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Layer2_layer_call_and_return_conditional_losses_1052652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
å
·
K__inference_lambda_layer_33_layer_call_and_return_conditional_losses_105322

inputs2
matmul_readvariableop_resource:
¬¬*
add_readvariableop_resource:	¬)
sub_readvariableop_resource:
identity¢MatMul/ReadVariableOp¢add/ReadVariableOp¢sub/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬¬*
dtype02
MatMul/ReadVariableOp
MatMulBatchMatMulV2inputsMatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
MatMul
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
add/ReadVariableOpw
addAddV2MatMul:output:0add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
add
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
:*
dtype02
sub/ReadVariableOpm
subSubadd:z:0sub/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
subT
TanhTanhsub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tanhh
IdentityIdentityTanh:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity
NoOpNoOp^MatMul/ReadVariableOp^add/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ¬: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Æ
¡
I__inference_sequential_33_layer_call_and_return_conditional_losses_105587
input_34 
layer1_105559:	¬
layer1_105561:	¬!
layer2_105564:
¬¬
layer2_105566:	¬!
layer3_105569:
¬¬
layer3_105571:	¬*
lambda_layer_33_105574:
¬¬%
lambda_layer_33_105576:	¬$
lambda_layer_33_105578:%
outputlayer_105581:	¬ 
outputlayer_105583:
identity¢Layer1/StatefulPartitionedCall¢Layer2/StatefulPartitionedCall¢Layer3/StatefulPartitionedCall¢#OutputLayer/StatefulPartitionedCall¢'lambda_layer_33/StatefulPartitionedCall
Layer1/StatefulPartitionedCallStatefulPartitionedCallinput_34layer1_105559layer1_105561*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Layer1_layer_call_and_return_conditional_losses_1052282 
Layer1/StatefulPartitionedCall°
Layer2/StatefulPartitionedCallStatefulPartitionedCall'Layer1/StatefulPartitionedCall:output:0layer2_105564layer2_105566*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Layer2_layer_call_and_return_conditional_losses_1052652 
Layer2/StatefulPartitionedCall°
Layer3/StatefulPartitionedCallStatefulPartitionedCall'Layer2/StatefulPartitionedCall:output:0layer3_105569layer3_105571*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Layer3_layer_call_and_return_conditional_losses_1053022 
Layer3/StatefulPartitionedCall÷
'lambda_layer_33/StatefulPartitionedCallStatefulPartitionedCall'Layer3/StatefulPartitionedCall:output:0lambda_layer_33_105574lambda_layer_33_105576lambda_layer_33_105578*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_lambda_layer_33_layer_call_and_return_conditional_losses_1053222)
'lambda_layer_33/StatefulPartitionedCallÑ
#OutputLayer/StatefulPartitionedCallStatefulPartitionedCall0lambda_layer_33/StatefulPartitionedCall:output:0outputlayer_105581outputlayer_105583*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_OutputLayer_layer_call_and_return_conditional_losses_1053602%
#OutputLayer/StatefulPartitionedCall
IdentityIdentity,OutputLayer/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^Layer1/StatefulPartitionedCall^Layer2/StatefulPartitionedCall^Layer3/StatefulPartitionedCall$^OutputLayer/StatefulPartitionedCall(^lambda_layer_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2@
Layer1/StatefulPartitionedCallLayer1/StatefulPartitionedCall2@
Layer2/StatefulPartitionedCallLayer2/StatefulPartitionedCall2@
Layer3/StatefulPartitionedCallLayer3/StatefulPartitionedCall2J
#OutputLayer/StatefulPartitionedCall#OutputLayer/StatefulPartitionedCall2R
'lambda_layer_33/StatefulPartitionedCall'lambda_layer_33/StatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_34
å
·
K__inference_lambda_layer_33_layer_call_and_return_conditional_losses_106083

inputs2
matmul_readvariableop_resource:
¬¬*
add_readvariableop_resource:	¬)
sub_readvariableop_resource:
identity¢MatMul/ReadVariableOp¢add/ReadVariableOp¢sub/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬¬*
dtype02
MatMul/ReadVariableOp
MatMulBatchMatMulV2inputsMatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
MatMul
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
add/ReadVariableOpw
addAddV2MatMul:output:0add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
add
sub/ReadVariableOpReadVariableOpsub_readvariableop_resource*
_output_shapes
:*
dtype02
sub/ReadVariableOpm
subSubadd:z:0sub/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
subT
TanhTanhsub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tanhh
IdentityIdentityTanh:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity
NoOpNoOp^MatMul/ReadVariableOp^add/ReadVariableOp^sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ¬: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2(
sub/ReadVariableOpsub/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
É!
ü
B__inference_Layer2_layer_call_and_return_conditional_losses_106020

inputs5
!tensordot_readvariableop_resource:
¬¬.
biasadd_readvariableop_resource:	¬
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
¬¬*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:¬2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tanhh
IdentityIdentityTanh:y:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs


'__inference_Layer1_layer_call_fn_105989

inputs
unknown:	¬
	unknown_0:	¬
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Layer1_layer_call_and_return_conditional_losses_1052282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î 
ÿ
G__inference_OutputLayer_layer_call_and_return_conditional_losses_106124

inputs4
!tensordot_readvariableop_resource:	¬-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	¬*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs


'__inference_Layer3_layer_call_fn_106069

inputs
unknown:
¬¬
	unknown_0:	¬
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Layer3_layer_call_and_return_conditional_losses_1053022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs


.__inference_sequential_33_layer_call_fn_105949

inputs
unknown:	¬
	unknown_0:	¬
	unknown_1:
¬¬
	unknown_2:	¬
	unknown_3:
¬¬
	unknown_4:	¬
	unknown_5:
¬¬
	unknown_6:	¬
	unknown_7:
	unknown_8:	¬
	unknown_9:
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_33_layer_call_and_return_conditional_losses_1055042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î 
ÿ
G__inference_OutputLayer_layer_call_and_return_conditional_losses_105360

inputs4
!tensordot_readvariableop_resource:	¬-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	¬*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
ÃÌ

!__inference__wrapped_model_105190
input_34I
6sequential_33_layer1_tensordot_readvariableop_resource:	¬C
4sequential_33_layer1_biasadd_readvariableop_resource:	¬J
6sequential_33_layer2_tensordot_readvariableop_resource:
¬¬C
4sequential_33_layer2_biasadd_readvariableop_resource:	¬J
6sequential_33_layer3_tensordot_readvariableop_resource:
¬¬C
4sequential_33_layer3_biasadd_readvariableop_resource:	¬P
<sequential_33_lambda_layer_33_matmul_readvariableop_resource:
¬¬H
9sequential_33_lambda_layer_33_add_readvariableop_resource:	¬G
9sequential_33_lambda_layer_33_sub_readvariableop_resource:N
;sequential_33_outputlayer_tensordot_readvariableop_resource:	¬G
9sequential_33_outputlayer_biasadd_readvariableop_resource:
identity¢+sequential_33/Layer1/BiasAdd/ReadVariableOp¢-sequential_33/Layer1/Tensordot/ReadVariableOp¢+sequential_33/Layer2/BiasAdd/ReadVariableOp¢-sequential_33/Layer2/Tensordot/ReadVariableOp¢+sequential_33/Layer3/BiasAdd/ReadVariableOp¢-sequential_33/Layer3/Tensordot/ReadVariableOp¢0sequential_33/OutputLayer/BiasAdd/ReadVariableOp¢2sequential_33/OutputLayer/Tensordot/ReadVariableOp¢3sequential_33/lambda_layer_33/MatMul/ReadVariableOp¢0sequential_33/lambda_layer_33/add/ReadVariableOp¢0sequential_33/lambda_layer_33/sub/ReadVariableOpÖ
-sequential_33/Layer1/Tensordot/ReadVariableOpReadVariableOp6sequential_33_layer1_tensordot_readvariableop_resource*
_output_shapes
:	¬*
dtype02/
-sequential_33/Layer1/Tensordot/ReadVariableOp
#sequential_33/Layer1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_33/Layer1/Tensordot/axes
#sequential_33/Layer1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_33/Layer1/Tensordot/free
$sequential_33/Layer1/Tensordot/ShapeShapeinput_34*
T0*
_output_shapes
:2&
$sequential_33/Layer1/Tensordot/Shape
,sequential_33/Layer1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_33/Layer1/Tensordot/GatherV2/axisº
'sequential_33/Layer1/Tensordot/GatherV2GatherV2-sequential_33/Layer1/Tensordot/Shape:output:0,sequential_33/Layer1/Tensordot/free:output:05sequential_33/Layer1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_33/Layer1/Tensordot/GatherV2¢
.sequential_33/Layer1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_33/Layer1/Tensordot/GatherV2_1/axisÀ
)sequential_33/Layer1/Tensordot/GatherV2_1GatherV2-sequential_33/Layer1/Tensordot/Shape:output:0,sequential_33/Layer1/Tensordot/axes:output:07sequential_33/Layer1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_33/Layer1/Tensordot/GatherV2_1
$sequential_33/Layer1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_33/Layer1/Tensordot/ConstÔ
#sequential_33/Layer1/Tensordot/ProdProd0sequential_33/Layer1/Tensordot/GatherV2:output:0-sequential_33/Layer1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_33/Layer1/Tensordot/Prod
&sequential_33/Layer1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_33/Layer1/Tensordot/Const_1Ü
%sequential_33/Layer1/Tensordot/Prod_1Prod2sequential_33/Layer1/Tensordot/GatherV2_1:output:0/sequential_33/Layer1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_33/Layer1/Tensordot/Prod_1
*sequential_33/Layer1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_33/Layer1/Tensordot/concat/axis
%sequential_33/Layer1/Tensordot/concatConcatV2,sequential_33/Layer1/Tensordot/free:output:0,sequential_33/Layer1/Tensordot/axes:output:03sequential_33/Layer1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_33/Layer1/Tensordot/concatà
$sequential_33/Layer1/Tensordot/stackPack,sequential_33/Layer1/Tensordot/Prod:output:0.sequential_33/Layer1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_33/Layer1/Tensordot/stackÑ
(sequential_33/Layer1/Tensordot/transpose	Transposeinput_34.sequential_33/Layer1/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_33/Layer1/Tensordot/transposeó
&sequential_33/Layer1/Tensordot/ReshapeReshape,sequential_33/Layer1/Tensordot/transpose:y:0-sequential_33/Layer1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&sequential_33/Layer1/Tensordot/Reshapeó
%sequential_33/Layer1/Tensordot/MatMulMatMul/sequential_33/Layer1/Tensordot/Reshape:output:05sequential_33/Layer1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2'
%sequential_33/Layer1/Tensordot/MatMul
&sequential_33/Layer1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:¬2(
&sequential_33/Layer1/Tensordot/Const_2
,sequential_33/Layer1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_33/Layer1/Tensordot/concat_1/axis¦
'sequential_33/Layer1/Tensordot/concat_1ConcatV20sequential_33/Layer1/Tensordot/GatherV2:output:0/sequential_33/Layer1/Tensordot/Const_2:output:05sequential_33/Layer1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_33/Layer1/Tensordot/concat_1å
sequential_33/Layer1/TensordotReshape/sequential_33/Layer1/Tensordot/MatMul:product:00sequential_33/Layer1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
sequential_33/Layer1/TensordotÌ
+sequential_33/Layer1/BiasAdd/ReadVariableOpReadVariableOp4sequential_33_layer1_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02-
+sequential_33/Layer1/BiasAdd/ReadVariableOpÜ
sequential_33/Layer1/BiasAddBiasAdd'sequential_33/Layer1/Tensordot:output:03sequential_33/Layer1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
sequential_33/Layer1/BiasAdd
sequential_33/Layer1/TanhTanh%sequential_33/Layer1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
sequential_33/Layer1/Tanh×
-sequential_33/Layer2/Tensordot/ReadVariableOpReadVariableOp6sequential_33_layer2_tensordot_readvariableop_resource* 
_output_shapes
:
¬¬*
dtype02/
-sequential_33/Layer2/Tensordot/ReadVariableOp
#sequential_33/Layer2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_33/Layer2/Tensordot/axes
#sequential_33/Layer2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_33/Layer2/Tensordot/free
$sequential_33/Layer2/Tensordot/ShapeShapesequential_33/Layer1/Tanh:y:0*
T0*
_output_shapes
:2&
$sequential_33/Layer2/Tensordot/Shape
,sequential_33/Layer2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_33/Layer2/Tensordot/GatherV2/axisº
'sequential_33/Layer2/Tensordot/GatherV2GatherV2-sequential_33/Layer2/Tensordot/Shape:output:0,sequential_33/Layer2/Tensordot/free:output:05sequential_33/Layer2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_33/Layer2/Tensordot/GatherV2¢
.sequential_33/Layer2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_33/Layer2/Tensordot/GatherV2_1/axisÀ
)sequential_33/Layer2/Tensordot/GatherV2_1GatherV2-sequential_33/Layer2/Tensordot/Shape:output:0,sequential_33/Layer2/Tensordot/axes:output:07sequential_33/Layer2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_33/Layer2/Tensordot/GatherV2_1
$sequential_33/Layer2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_33/Layer2/Tensordot/ConstÔ
#sequential_33/Layer2/Tensordot/ProdProd0sequential_33/Layer2/Tensordot/GatherV2:output:0-sequential_33/Layer2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_33/Layer2/Tensordot/Prod
&sequential_33/Layer2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_33/Layer2/Tensordot/Const_1Ü
%sequential_33/Layer2/Tensordot/Prod_1Prod2sequential_33/Layer2/Tensordot/GatherV2_1:output:0/sequential_33/Layer2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_33/Layer2/Tensordot/Prod_1
*sequential_33/Layer2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_33/Layer2/Tensordot/concat/axis
%sequential_33/Layer2/Tensordot/concatConcatV2,sequential_33/Layer2/Tensordot/free:output:0,sequential_33/Layer2/Tensordot/axes:output:03sequential_33/Layer2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_33/Layer2/Tensordot/concatà
$sequential_33/Layer2/Tensordot/stackPack,sequential_33/Layer2/Tensordot/Prod:output:0.sequential_33/Layer2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_33/Layer2/Tensordot/stackç
(sequential_33/Layer2/Tensordot/transpose	Transposesequential_33/Layer1/Tanh:y:0.sequential_33/Layer2/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2*
(sequential_33/Layer2/Tensordot/transposeó
&sequential_33/Layer2/Tensordot/ReshapeReshape,sequential_33/Layer2/Tensordot/transpose:y:0-sequential_33/Layer2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&sequential_33/Layer2/Tensordot/Reshapeó
%sequential_33/Layer2/Tensordot/MatMulMatMul/sequential_33/Layer2/Tensordot/Reshape:output:05sequential_33/Layer2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2'
%sequential_33/Layer2/Tensordot/MatMul
&sequential_33/Layer2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:¬2(
&sequential_33/Layer2/Tensordot/Const_2
,sequential_33/Layer2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_33/Layer2/Tensordot/concat_1/axis¦
'sequential_33/Layer2/Tensordot/concat_1ConcatV20sequential_33/Layer2/Tensordot/GatherV2:output:0/sequential_33/Layer2/Tensordot/Const_2:output:05sequential_33/Layer2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_33/Layer2/Tensordot/concat_1å
sequential_33/Layer2/TensordotReshape/sequential_33/Layer2/Tensordot/MatMul:product:00sequential_33/Layer2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
sequential_33/Layer2/TensordotÌ
+sequential_33/Layer2/BiasAdd/ReadVariableOpReadVariableOp4sequential_33_layer2_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02-
+sequential_33/Layer2/BiasAdd/ReadVariableOpÜ
sequential_33/Layer2/BiasAddBiasAdd'sequential_33/Layer2/Tensordot:output:03sequential_33/Layer2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
sequential_33/Layer2/BiasAdd
sequential_33/Layer2/TanhTanh%sequential_33/Layer2/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
sequential_33/Layer2/Tanh×
-sequential_33/Layer3/Tensordot/ReadVariableOpReadVariableOp6sequential_33_layer3_tensordot_readvariableop_resource* 
_output_shapes
:
¬¬*
dtype02/
-sequential_33/Layer3/Tensordot/ReadVariableOp
#sequential_33/Layer3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_33/Layer3/Tensordot/axes
#sequential_33/Layer3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_33/Layer3/Tensordot/free
$sequential_33/Layer3/Tensordot/ShapeShapesequential_33/Layer2/Tanh:y:0*
T0*
_output_shapes
:2&
$sequential_33/Layer3/Tensordot/Shape
,sequential_33/Layer3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_33/Layer3/Tensordot/GatherV2/axisº
'sequential_33/Layer3/Tensordot/GatherV2GatherV2-sequential_33/Layer3/Tensordot/Shape:output:0,sequential_33/Layer3/Tensordot/free:output:05sequential_33/Layer3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_33/Layer3/Tensordot/GatherV2¢
.sequential_33/Layer3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_33/Layer3/Tensordot/GatherV2_1/axisÀ
)sequential_33/Layer3/Tensordot/GatherV2_1GatherV2-sequential_33/Layer3/Tensordot/Shape:output:0,sequential_33/Layer3/Tensordot/axes:output:07sequential_33/Layer3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_33/Layer3/Tensordot/GatherV2_1
$sequential_33/Layer3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_33/Layer3/Tensordot/ConstÔ
#sequential_33/Layer3/Tensordot/ProdProd0sequential_33/Layer3/Tensordot/GatherV2:output:0-sequential_33/Layer3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_33/Layer3/Tensordot/Prod
&sequential_33/Layer3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_33/Layer3/Tensordot/Const_1Ü
%sequential_33/Layer3/Tensordot/Prod_1Prod2sequential_33/Layer3/Tensordot/GatherV2_1:output:0/sequential_33/Layer3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_33/Layer3/Tensordot/Prod_1
*sequential_33/Layer3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_33/Layer3/Tensordot/concat/axis
%sequential_33/Layer3/Tensordot/concatConcatV2,sequential_33/Layer3/Tensordot/free:output:0,sequential_33/Layer3/Tensordot/axes:output:03sequential_33/Layer3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_33/Layer3/Tensordot/concatà
$sequential_33/Layer3/Tensordot/stackPack,sequential_33/Layer3/Tensordot/Prod:output:0.sequential_33/Layer3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_33/Layer3/Tensordot/stackç
(sequential_33/Layer3/Tensordot/transpose	Transposesequential_33/Layer2/Tanh:y:0.sequential_33/Layer3/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2*
(sequential_33/Layer3/Tensordot/transposeó
&sequential_33/Layer3/Tensordot/ReshapeReshape,sequential_33/Layer3/Tensordot/transpose:y:0-sequential_33/Layer3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&sequential_33/Layer3/Tensordot/Reshapeó
%sequential_33/Layer3/Tensordot/MatMulMatMul/sequential_33/Layer3/Tensordot/Reshape:output:05sequential_33/Layer3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2'
%sequential_33/Layer3/Tensordot/MatMul
&sequential_33/Layer3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:¬2(
&sequential_33/Layer3/Tensordot/Const_2
,sequential_33/Layer3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_33/Layer3/Tensordot/concat_1/axis¦
'sequential_33/Layer3/Tensordot/concat_1ConcatV20sequential_33/Layer3/Tensordot/GatherV2:output:0/sequential_33/Layer3/Tensordot/Const_2:output:05sequential_33/Layer3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_33/Layer3/Tensordot/concat_1å
sequential_33/Layer3/TensordotReshape/sequential_33/Layer3/Tensordot/MatMul:product:00sequential_33/Layer3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
sequential_33/Layer3/TensordotÌ
+sequential_33/Layer3/BiasAdd/ReadVariableOpReadVariableOp4sequential_33_layer3_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02-
+sequential_33/Layer3/BiasAdd/ReadVariableOpÜ
sequential_33/Layer3/BiasAddBiasAdd'sequential_33/Layer3/Tensordot:output:03sequential_33/Layer3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
sequential_33/Layer3/BiasAdd
sequential_33/Layer3/TanhTanh%sequential_33/Layer3/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
sequential_33/Layer3/Tanhé
3sequential_33/lambda_layer_33/MatMul/ReadVariableOpReadVariableOp<sequential_33_lambda_layer_33_matmul_readvariableop_resource* 
_output_shapes
:
¬¬*
dtype025
3sequential_33/lambda_layer_33/MatMul/ReadVariableOpð
$sequential_33/lambda_layer_33/MatMulBatchMatMulV2sequential_33/Layer3/Tanh:y:0;sequential_33/lambda_layer_33/MatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2&
$sequential_33/lambda_layer_33/MatMulÛ
0sequential_33/lambda_layer_33/add/ReadVariableOpReadVariableOp9sequential_33_lambda_layer_33_add_readvariableop_resource*
_output_shapes	
:¬*
dtype022
0sequential_33/lambda_layer_33/add/ReadVariableOpï
!sequential_33/lambda_layer_33/addAddV2-sequential_33/lambda_layer_33/MatMul:output:08sequential_33/lambda_layer_33/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!sequential_33/lambda_layer_33/addÚ
0sequential_33/lambda_layer_33/sub/ReadVariableOpReadVariableOp9sequential_33_lambda_layer_33_sub_readvariableop_resource*
_output_shapes
:*
dtype022
0sequential_33/lambda_layer_33/sub/ReadVariableOpå
!sequential_33/lambda_layer_33/subSub%sequential_33/lambda_layer_33/add:z:08sequential_33/lambda_layer_33/sub/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!sequential_33/lambda_layer_33/sub®
"sequential_33/lambda_layer_33/TanhTanh%sequential_33/lambda_layer_33/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2$
"sequential_33/lambda_layer_33/Tanhå
2sequential_33/OutputLayer/Tensordot/ReadVariableOpReadVariableOp;sequential_33_outputlayer_tensordot_readvariableop_resource*
_output_shapes
:	¬*
dtype024
2sequential_33/OutputLayer/Tensordot/ReadVariableOp
(sequential_33/OutputLayer/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_33/OutputLayer/Tensordot/axes¥
(sequential_33/OutputLayer/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2*
(sequential_33/OutputLayer/Tensordot/free¬
)sequential_33/OutputLayer/Tensordot/ShapeShape&sequential_33/lambda_layer_33/Tanh:y:0*
T0*
_output_shapes
:2+
)sequential_33/OutputLayer/Tensordot/Shape¨
1sequential_33/OutputLayer/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_33/OutputLayer/Tensordot/GatherV2/axisÓ
,sequential_33/OutputLayer/Tensordot/GatherV2GatherV22sequential_33/OutputLayer/Tensordot/Shape:output:01sequential_33/OutputLayer/Tensordot/free:output:0:sequential_33/OutputLayer/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,sequential_33/OutputLayer/Tensordot/GatherV2¬
3sequential_33/OutputLayer/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3sequential_33/OutputLayer/Tensordot/GatherV2_1/axisÙ
.sequential_33/OutputLayer/Tensordot/GatherV2_1GatherV22sequential_33/OutputLayer/Tensordot/Shape:output:01sequential_33/OutputLayer/Tensordot/axes:output:0<sequential_33/OutputLayer/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:20
.sequential_33/OutputLayer/Tensordot/GatherV2_1 
)sequential_33/OutputLayer/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_33/OutputLayer/Tensordot/Constè
(sequential_33/OutputLayer/Tensordot/ProdProd5sequential_33/OutputLayer/Tensordot/GatherV2:output:02sequential_33/OutputLayer/Tensordot/Const:output:0*
T0*
_output_shapes
: 2*
(sequential_33/OutputLayer/Tensordot/Prod¤
+sequential_33/OutputLayer/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_33/OutputLayer/Tensordot/Const_1ð
*sequential_33/OutputLayer/Tensordot/Prod_1Prod7sequential_33/OutputLayer/Tensordot/GatherV2_1:output:04sequential_33/OutputLayer/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2,
*sequential_33/OutputLayer/Tensordot/Prod_1¤
/sequential_33/OutputLayer/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_33/OutputLayer/Tensordot/concat/axis²
*sequential_33/OutputLayer/Tensordot/concatConcatV21sequential_33/OutputLayer/Tensordot/free:output:01sequential_33/OutputLayer/Tensordot/axes:output:08sequential_33/OutputLayer/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2,
*sequential_33/OutputLayer/Tensordot/concatô
)sequential_33/OutputLayer/Tensordot/stackPack1sequential_33/OutputLayer/Tensordot/Prod:output:03sequential_33/OutputLayer/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2+
)sequential_33/OutputLayer/Tensordot/stackÿ
-sequential_33/OutputLayer/Tensordot/transpose	Transpose&sequential_33/lambda_layer_33/Tanh:y:03sequential_33/OutputLayer/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2/
-sequential_33/OutputLayer/Tensordot/transpose
+sequential_33/OutputLayer/Tensordot/ReshapeReshape1sequential_33/OutputLayer/Tensordot/transpose:y:02sequential_33/OutputLayer/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2-
+sequential_33/OutputLayer/Tensordot/Reshape
*sequential_33/OutputLayer/Tensordot/MatMulMatMul4sequential_33/OutputLayer/Tensordot/Reshape:output:0:sequential_33/OutputLayer/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_33/OutputLayer/Tensordot/MatMul¤
+sequential_33/OutputLayer/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_33/OutputLayer/Tensordot/Const_2¨
1sequential_33/OutputLayer/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_33/OutputLayer/Tensordot/concat_1/axis¿
,sequential_33/OutputLayer/Tensordot/concat_1ConcatV25sequential_33/OutputLayer/Tensordot/GatherV2:output:04sequential_33/OutputLayer/Tensordot/Const_2:output:0:sequential_33/OutputLayer/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2.
,sequential_33/OutputLayer/Tensordot/concat_1ø
#sequential_33/OutputLayer/TensordotReshape4sequential_33/OutputLayer/Tensordot/MatMul:product:05sequential_33/OutputLayer/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#sequential_33/OutputLayer/TensordotÚ
0sequential_33/OutputLayer/BiasAdd/ReadVariableOpReadVariableOp9sequential_33_outputlayer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0sequential_33/OutputLayer/BiasAdd/ReadVariableOpï
!sequential_33/OutputLayer/BiasAddBiasAdd,sequential_33/OutputLayer/Tensordot:output:08sequential_33/OutputLayer/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_33/OutputLayer/BiasAdd
IdentityIdentity*sequential_33/OutputLayer/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityì
NoOpNoOp,^sequential_33/Layer1/BiasAdd/ReadVariableOp.^sequential_33/Layer1/Tensordot/ReadVariableOp,^sequential_33/Layer2/BiasAdd/ReadVariableOp.^sequential_33/Layer2/Tensordot/ReadVariableOp,^sequential_33/Layer3/BiasAdd/ReadVariableOp.^sequential_33/Layer3/Tensordot/ReadVariableOp1^sequential_33/OutputLayer/BiasAdd/ReadVariableOp3^sequential_33/OutputLayer/Tensordot/ReadVariableOp4^sequential_33/lambda_layer_33/MatMul/ReadVariableOp1^sequential_33/lambda_layer_33/add/ReadVariableOp1^sequential_33/lambda_layer_33/sub/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : 2Z
+sequential_33/Layer1/BiasAdd/ReadVariableOp+sequential_33/Layer1/BiasAdd/ReadVariableOp2^
-sequential_33/Layer1/Tensordot/ReadVariableOp-sequential_33/Layer1/Tensordot/ReadVariableOp2Z
+sequential_33/Layer2/BiasAdd/ReadVariableOp+sequential_33/Layer2/BiasAdd/ReadVariableOp2^
-sequential_33/Layer2/Tensordot/ReadVariableOp-sequential_33/Layer2/Tensordot/ReadVariableOp2Z
+sequential_33/Layer3/BiasAdd/ReadVariableOp+sequential_33/Layer3/BiasAdd/ReadVariableOp2^
-sequential_33/Layer3/Tensordot/ReadVariableOp-sequential_33/Layer3/Tensordot/ReadVariableOp2d
0sequential_33/OutputLayer/BiasAdd/ReadVariableOp0sequential_33/OutputLayer/BiasAdd/ReadVariableOp2h
2sequential_33/OutputLayer/Tensordot/ReadVariableOp2sequential_33/OutputLayer/Tensordot/ReadVariableOp2j
3sequential_33/lambda_layer_33/MatMul/ReadVariableOp3sequential_33/lambda_layer_33/MatMul/ReadVariableOp2d
0sequential_33/lambda_layer_33/add/ReadVariableOp0sequential_33/lambda_layer_33/add/ReadVariableOp2d
0sequential_33/lambda_layer_33/sub/ReadVariableOp0sequential_33/lambda_layer_33/sub/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_34"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¸
serving_default¤
A
input_345
serving_default_input_34:0ÿÿÿÿÿÿÿÿÿC
OutputLayer4
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:úm
©
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
*i&call_and_return_all_conditional_losses
j__call__
k_default_save_signature"
_tf_keras_sequential
»

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*l&call_and_return_all_conditional_losses
m__call__"
_tf_keras_layer
»

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*n&call_and_return_all_conditional_losses
o__call__"
_tf_keras_layer
»

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*p&call_and_return_all_conditional_losses
q__call__"
_tf_keras_layer
½
w
b
 lbda
!regularization_losses
"	variables
#trainable_variables
$	keras_api
*r&call_and_return_all_conditional_losses
s__call__"
_tf_keras_layer
»

%kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
*t&call_and_return_all_conditional_losses
u__call__"
_tf_keras_layer

+iter

,beta_1

-beta_2
	.decay
/learning_ratemSmTmUmVmWmXmYmZ m[%m\&m]v^v_v`vavbvcvdve vf%vg&vh"
	optimizer
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
 8
%9
&10"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
 8
%9
&10"
trackable_list_wrapper
Ê
0layer_regularization_losses
1metrics

2layers
3non_trainable_variables
4layer_metrics
regularization_losses
	variables
	trainable_variables
j__call__
k_default_save_signature
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
,
vserving_default"
signature_map
 :	¬2Layer1/kernel
:¬2Layer1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
5layer_regularization_losses
6metrics

7layers
8non_trainable_variables
9layer_metrics
regularization_losses
	variables
trainable_variables
m__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
!:
¬¬2Layer2/kernel
:¬2Layer2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
:layer_regularization_losses
;metrics

<layers
=non_trainable_variables
>layer_metrics
regularization_losses
	variables
trainable_variables
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
!:
¬¬2Layer3/kernel
:¬2Layer3/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
?layer_regularization_losses
@metrics

Alayers
Bnon_trainable_variables
Clayer_metrics
regularization_losses
	variables
trainable_variables
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
*:(
¬¬2lambda_layer_33/kernel
#:!¬2lambda_layer_33/bias
$:"2lambda_layer_33/Lambda
 "
trackable_list_wrapper
5
0
1
 2"
trackable_list_wrapper
5
0
1
 2"
trackable_list_wrapper
­
Dlayer_regularization_losses
Emetrics

Flayers
Gnon_trainable_variables
Hlayer_metrics
!regularization_losses
"	variables
#trainable_variables
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
%:#	¬2OutputLayer/kernel
:2OutputLayer/bias
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
­
Ilayer_regularization_losses
Jmetrics

Klayers
Lnon_trainable_variables
Mlayer_metrics
'regularization_losses
(	variables
)trainable_variables
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
'
N0"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ototal
	Pcount
Q	variables
R	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
O0
P1"
trackable_list_wrapper
-
Q	variables"
_generic_user_object
%:#	¬2Adam/Layer1/kernel/m
:¬2Adam/Layer1/bias/m
&:$
¬¬2Adam/Layer2/kernel/m
:¬2Adam/Layer2/bias/m
&:$
¬¬2Adam/Layer3/kernel/m
:¬2Adam/Layer3/bias/m
/:-
¬¬2Adam/lambda_layer_33/kernel/m
(:&¬2Adam/lambda_layer_33/bias/m
):'2Adam/lambda_layer_33/Lambda/m
*:(	¬2Adam/OutputLayer/kernel/m
#:!2Adam/OutputLayer/bias/m
%:#	¬2Adam/Layer1/kernel/v
:¬2Adam/Layer1/bias/v
&:$
¬¬2Adam/Layer2/kernel/v
:¬2Adam/Layer2/bias/v
&:$
¬¬2Adam/Layer3/kernel/v
:¬2Adam/Layer3/bias/v
/:-
¬¬2Adam/lambda_layer_33/kernel/v
(:&¬2Adam/lambda_layer_33/bias/v
):'2Adam/lambda_layer_33/Lambda/v
*:(	¬2Adam/OutputLayer/kernel/v
#:!2Adam/OutputLayer/bias/v
ò2ï
I__inference_sequential_33_layer_call_and_return_conditional_losses_105774
I__inference_sequential_33_layer_call_and_return_conditional_losses_105895
I__inference_sequential_33_layer_call_and_return_conditional_losses_105587
I__inference_sequential_33_layer_call_and_return_conditional_losses_105618À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
.__inference_sequential_33_layer_call_fn_105392
.__inference_sequential_33_layer_call_fn_105922
.__inference_sequential_33_layer_call_fn_105949
.__inference_sequential_33_layer_call_fn_105556À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÍBÊ
!__inference__wrapped_model_105190input_34"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_Layer1_layer_call_and_return_conditional_losses_105980¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_Layer1_layer_call_fn_105989¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_Layer2_layer_call_and_return_conditional_losses_106020¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_Layer2_layer_call_fn_106029¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_Layer3_layer_call_and_return_conditional_losses_106060¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_Layer3_layer_call_fn_106069¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_lambda_layer_33_layer_call_and_return_conditional_losses_106083¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ú2×
0__inference_lambda_layer_33_layer_call_fn_106094¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_OutputLayer_layer_call_and_return_conditional_losses_106124¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_OutputLayer_layer_call_fn_106133¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÌBÉ
$__inference_signature_wrapper_105653input_34"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 «
B__inference_Layer1_layer_call_and_return_conditional_losses_105980e3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¬
 
'__inference_Layer1_layer_call_fn_105989X3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¬¬
B__inference_Layer2_layer_call_and_return_conditional_losses_106020f4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¬
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¬
 
'__inference_Layer2_layer_call_fn_106029Y4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¬
ª "ÿÿÿÿÿÿÿÿÿ¬¬
B__inference_Layer3_layer_call_and_return_conditional_losses_106060f4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¬
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¬
 
'__inference_Layer3_layer_call_fn_106069Y4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¬
ª "ÿÿÿÿÿÿÿÿÿ¬°
G__inference_OutputLayer_layer_call_and_return_conditional_losses_106124e%&4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¬
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_OutputLayer_layer_call_fn_106133X%&4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¬
ª "ÿÿÿÿÿÿÿÿÿ©
!__inference__wrapped_model_105190 %&5¢2
+¢(
&#
input_34ÿÿÿÿÿÿÿÿÿ
ª "=ª:
8
OutputLayer)&
OutputLayerÿÿÿÿÿÿÿÿÿ¶
K__inference_lambda_layer_33_layer_call_and_return_conditional_losses_106083g 4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¬
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¬
 
0__inference_lambda_layer_33_layer_call_fn_106094Z 4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¬
ª "ÿÿÿÿÿÿÿÿÿ¬Ä
I__inference_sequential_33_layer_call_and_return_conditional_losses_105587w %&=¢:
3¢0
&#
input_34ÿÿÿÿÿÿÿÿÿ
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Ä
I__inference_sequential_33_layer_call_and_return_conditional_losses_105618w %&=¢:
3¢0
&#
input_34ÿÿÿÿÿÿÿÿÿ
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Â
I__inference_sequential_33_layer_call_and_return_conditional_losses_105774u %&;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Â
I__inference_sequential_33_layer_call_and_return_conditional_losses_105895u %&;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_sequential_33_layer_call_fn_105392j %&=¢:
3¢0
&#
input_34ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_33_layer_call_fn_105556j %&=¢:
3¢0
&#
input_34ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_33_layer_call_fn_105922h %&;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_33_layer_call_fn_105949h %&;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¸
$__inference_signature_wrapper_105653 %&A¢>
¢ 
7ª4
2
input_34&#
input_34ÿÿÿÿÿÿÿÿÿ"=ª:
8
OutputLayer)&
OutputLayerÿÿÿÿÿÿÿÿÿ