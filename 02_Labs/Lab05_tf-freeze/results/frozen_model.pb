
X
Placeholder/inputs_placeholderPlaceholder*
dtype0*
shape:?????????

^
NN/W1Const*A
value8B6
"(?RG=Z?W=?sW=?a=3L=?V=[aE=9?`=ˇP=C\=*
dtype0
@

NN/W1/readIdentityNN/W1*
T0*
_class

loc:@NN/W1
6
NN/b1Const*
valueB*׶??*
dtype0
@

NN/b1/readIdentityNN/b1*
T0*
_class

loc:@NN/b1
^
NN/W2Const*A
value8B6
"(l?仅뛽Q]F??aν???;??=?@??+Kƽ?~??z??*
dtype0
@

NN/W2/readIdentityNN/W2*
T0*
_class

loc:@NN/W2
6
NN/b2Const*
valueB*P??=*
dtype0
@

NN/b2/readIdentityNN/b2*
T0*
_class

loc:@NN/b2
n
	NN/MatMulMatMulPlaceholder/inputs_placeholder
NN/W1/read*
transpose_b( *
T0*
transpose_a( 
-
NN/addAdd	NN/MatMul
NN/b1/read*
T0
 
NN/ReluReluNN/add*
T0
p
NN/MatMul_1MatMulPlaceholder/inputs_placeholder
NN/W2/read*
transpose_b( *
T0*
transpose_a( 
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
NN/truediv/yConst*
valueB
 *   @*
dtype0
4

NN/truedivRealDivNN/AddNN/truediv/y*
T0
C
Accuracy/predictions/yConst*
valueB
 *   ?*
dtype0
L
Accuracy/predictionsGreater
NN/truedivAccuracy/predictions/y*
T0 