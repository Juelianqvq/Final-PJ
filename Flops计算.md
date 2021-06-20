<img src="C:\Users\Kwx\AppData\Roaming\Typora\typora-user-images\image-20210612164753731.png" alt="image-20210612164753731" style="zoom:80%;" />

CvT原文：http://arxiv.org/abs/2103.15808

大致流程：

每个stage包含 token embedding 与 transformer，共 3 个 stage ，stage3嵌入类别向量后经mlp输出分类结果

1.token embedding ：卷积层 + LN

2.transformer：LN(卷积self-attention) + LN(全连接) , 残差连接

   关于 卷积self-attenton ： 通过depthwise卷积 + BN + pointwise卷积 获得三个向量 q，k，v ，之后无不同



FLOPs 计算：

BN,LN,GELU,Softmax 的 FLOPs 相较之下可忽略

卷积层 ：$C_{in}\times K^2\times C_{out}\times H\times W$

深度分离 ：$C_{in}\times H\times W\times (K^2+C_{out})$

全连接层 ：$I\times J$

Cifar-10 输入 $3\times 32\times32$



Stage 1: 

te:$3\times7^2\times64\times8^2=602112$

tr:$3\times64\times8^2\times(3^2+64)=897024$  &  $64\times64=4096$

​    $2\times64\times256=32768$



stage 2:

te:$64\times3^2\times192\times4^2=1769472$

tr:$2\times3\times192\times4^2\times(3^2+192)=3704832$  &  $2\times192\times192=73728$

​    $2\times2\times192\times768=589824$



stage 3:

te:$192\times3^2\times384\times2^2=2654208$

tr:$10\times3\times2^2\times384\times(3^2+384)=18109440$ & $10\times384\times384=1474560$

​    $10\times2\times384\times1536=11796480$

 

4:

$384\times10=3840$



###### **Mac=41712384=0.042GMac=0.08GFlops**



