
X
Placeholder/inputs_placeholderPlaceholder*
dtype0*
shape:?????????

^
NN/W1Const*
dtype0*A
value8B6
"(2	>?U?<`??9?B???>?:?<$???l?????Ľ`?0?
@

NN/W1/readIdentityNN/W1*
T0*
_class

loc:@NN/W1
6
NN/b1Const*
dtype0*
valueB*E7?;
@

NN/b1/readIdentityNN/b1*
T0*
_class

loc:@NN/b1
^
NN/W2Const*
dtype0*A
value8B6
"(SJd=??o=?"]=?h=)`=??m=HG=zzM=??i=??Z=
@

NN/W2/readIdentityNN/W2*
T0*
_class

loc:@NN/W2
6
NN/b2Const*
dtype0*
valueB*?j??
@

NN/b2/readIdentityNN/b2*
T0*
_class

loc:@NN/b2
n
	NN/MatMulMatMulPlaceholder/inputs_placeholder
NN/W1/read*
T0*
transpose_a( *
transpose_b( 
-
NN/addAdd	NN/MatMul
NN/b1/read*
T0
 
NN/ReluReluNN/add*
T0
p
NN/MatMul_1MatMulPlaceholder/inputs_placeholder
NN/W2/read*
T0*
transpose_a( *
transpose_b( 
1
NN/add_1AddNN/MatMul_1
NN/b2/read*
T0
$
	NN/Relu_1ReluNN/add_1*
T0
*
NN/AddAddNN/Relu	NN/Relu_1*
T0
9
NN/truediv/yConst*
dtype0*
valueB
 *   @
4

NN/truedivRealDivNN/AddNN/truediv/y*
T0
C
Accuracy/predictions/yConst*
dtype0*
valueB
 *   ?
L
Accuracy/predictionsGreater
NN/truedivAccuracy/predictions/y*
T0 