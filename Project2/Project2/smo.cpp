//C-Support Vector Classification(Binary Case)
//The advantage of SMO(Sequential Minimal Optimization) Algorithm lies in the fact that solving for
//two Lagrange multipliers can be done analytically.

/*
There are three components to SMO:
(1)an analytic method to solve for the two Lagrange multipliers.
(2)a heuristic(启发式的)for choosing which multipliers to optimize:
    SMO will always optimize two Lagrange multipliers at every step, with one of the Lagrange multipliers 
having previously violated the KKT conditions before the step. That is, SMO will always alter two Lagrange
multipliers to move uphill in the objective function projected into the one-dimentional feasible subspace.
    There are tow separate choice heuristics: one for the first Lagrange multiplier and one for the second.
The choice of the first heuristic provides the outer loop of the SMO algorithm. The outer loop first iterates
over the entire training set, determing whether each example violates the KKT conditions. If an example violates
the KKT conditions, it is then eligible for immediate optimzation. Once a violated example is found, a second
multiplier is chosen using the second choice heuristic, and the two multiplers are jointly optimized.
    Once a first Lagrange multipler is chosen, SMO chooses the second Lagrange multiplier to maximize the 
size of the step taken during joint optimization.
(3)a method for computing b:
    After each step, b is re-computed.
*/
//The kernel function is RBF(Radial-Basis Function).

#include "smo.h"

int main()
{
	ofstream outClientFile("data_result.txt", ios::out);//如果指定的文件data_result.txt不存在，ofstream就用该文件名建立它。
	int numChanged=0;
	int examineAll=1;
	//srand((unsigned int)time(NULL));
	initialize();
	//以下两层循环，开始时检查所有样本，选择不符合KKT条件的两个乘子进行优化，选择成功，返回1，否则，返回0
	//所以成功了，numChanged必然>0，从第二遍循环时，不从整个样本中去寻找不符合KKT条件的两个乘子进行优化，
	//而是从非边界乘子中去寻找，因为非边界样本需要调整的可能性更大，边界样本往往不被调整而始终停留在边界上。
	//如果没找到，再从整个样本中去找，直到整个样本中再也找不到需要改变的乘子为止，此时算法结束。
	while(numChanged>0||examineAll)
	{
		numChanged=0;
		if(examineAll)
		{
			for(int k=0;k<end_support_i;k++)
			{
				numChanged+=examineExample(k);//examine all examples
			}
		}
		else
		{
			for(int k=0;k<end_support_i;k++)
			{
				if(alph[k]!=0&&alph[k]!=C)
					numChanged+=examineExample(k);//loop k over all non-bound Lagrange multipliers
			}
		}
		if(examineAll==1)
		{
			examineAll=0;
		}
		else if(numChanged==0)
		{
			examineAll=1;
		}
		/*
		The examples in the non-bound subset are most likely to violate the KKT conditions.
		*/
	}
	//存放训练后的参数
	{
		outClientFile<<"d="<<d<<endl;//维数
		outClientFile<<"b="<<b<<endl;//threshold
		outClientFile<<"two_sigma_squared="<<two_sigma_squared<<endl;
		int n_support_vectors=0;
		for(int i=0;i<end_support_i;i++)
		{
			if(alph[i]>0)//此地方是否可以改为alph[i]>tolerance?????????????????
			{
				n_support_vectors++;
			}
		}
		outClientFile<<"n_support_vectors="<<n_support_vectors<<endl;
		outClientFile<<"rate="<<(double)n_support_vectors/first_test_i<<endl;
		for(int i=0;i<end_support_i;i++)
		{
			if(alph[i]>0)
			{
				outClientFile<<"alph["<<i<<"]="<<alph[i]<<endl;
			}
		}
		outClientFile<<endl;
	}
	//输出，以及测试
	cout<<error_rate()<<endl;
	while (1);
	return 0;
}

/////////examineExample程序
//假定第一个乘子ai(位置为i1)，examineExample(i1)首先检查，如果它超出tolerance而违背KKT条件，那么它就成为第一个乘子；
//然后，寻找第二个乘子(位置为i2),通过调用takeStep(i1,i2)来优化这两个乘子。
int examineExample(int i1)
{
	int y1;
	double alph1,E1,r1;
	y1=target[i1];
	alph1=alph[i1];
	//alphi在initialize();中的初始化为0.0。
	if(alph1>0&&alph1<C)
		E1=error_cache[i1];
		/*
		A cached error value E is kept for every non-bound training example from 
		which example can be chosed for maximizing the step size.
		*/
	else
		E1=learned_func(i1)-y1;//learned_func为计算输出函数
	/*
	When an error E is required by SMO, it will look up the error in the error cache if the 
	corresponding Lagrange multiplier is not at bound. Otherwise, it will evaluate the current
	SVM decision function based on the current alph vector.
	*/
	r1=y1*E1;
	//违反KKT(Karush-Kuhn-Tucker)条件的判断
	/*
	KKT condition:
	    if alphi == 0, yi*Ei >= 0;
        if 0 < alphi < C, yi*Ei == 0;
		if alphi == C, yi*Ei <= 0;
	*/
	/*
	The SMO algorithm is based on the evaluation of the KKT conditions. When every multiplier 
	fulfils the KKT conditions of the problem, the algorithm terminates.
	*/
	if((r1>tolerance&&alph1>0)||(r1<-tolerance&&alph1<C))
	{
		/*使用三种方法选择第二个乘子:
		hierarchy one：在non-bound乘子中寻找maximum fabs(E1-E2)的样本
		hierarchy two：如果上面没取得进展,那么从随机位置查找non-boundary 样本
		hierarchy three：如果上面也失败，则从随机位置查找整个样本,改为bound样本
		以上方法 非边界值优先 其次再随机选取一个边界值
		*/
		if (examineFirstChoice(i1,E1))//hierarchy one
		{
			return 1;
      	}
        /*
		The hierarchy of second choice heuristics consists of the following:
		(A)If the above heuristic, i.e. examineFirstChoice(i1,E1), does not make positive progress, 
		then SMO starts iterating through the non-bound examples, searching for a second example that
		can make positive progress;
		(B)If none of the non-bound examples make positive progress, then SMO starts iterating through
		the entire training set until an example is found that makes positive progress.
		    Both the iteration through the non-bound examples (A) and the iteration through the entire 
		training set (B) are started at random locations in order not to bias SMO towards the examples at
		the beginning of the training set.
		*/
     	if (examineNonBound(i1))//hierarchy two
     	{
        	return 1;
     	}

      	if (examineBound(i1))//hierarchy three
     	{
        	return 1;
		}
        /*
		Once a first Lagrange multiplier is chosen, SMO chooses the second Lagrange
		multiplier to maximize the size of the step taken during joint optimization.
		SMO approximates the step size by |E1-E2|.
		*/
	}
	///没有进展
	return 0;
}

//hierarchy one：在non-bound乘子中寻找maximum fabs(E1-E2)的样本
//There are cases when there is no positive progress; for instance when both input vectors are identical,
//which causes the objective function to become flat along the direction of optimiztion.
int examineFirstChoice(int i1,double E1)
{
   int k,i2;
   double tmax;
   double E2,temp;
   for(i2=-1,tmax=0.0,k=0;k<end_support_i;k++)//end_support_i
   {
	   if(alph[k]>0&&alph[k]<C)//choose non-bound multiplier
	   {
   		   E2=error_cache[k];
		   /*
		   A cached error value E is kept for every non-bound training example from 
		   which example can be chosed for maximizing the step size.
		   */
   		   temp=fabs(E1-E2);//fabs:returns the absolute value of its argument.
   		   if(temp>tmax)
		   {
			   tmax=temp;
   			   i2=k;
		   }
	   }
   }
   if(i2>=0)//如果没有non-bound multiplier，i2==-1。
   {
	   if(takeStep(i1,i2))//If there has a positive progress, return 1.
	   {
		   return 1;
	   }
   }
   return 0;
}
//hierarchy two：如果上面没取得进展,那么从随机位置查找non-boundary 样本
int examineNonBound(int i1)
{
  	 int k;
	 int k0 = rand()%end_support_i;
	 //The result of the modulus operator (%) is the remainder when the first operand is divided by the second. 

	 //If there is no positive progress in hierarchy one, hierarchy two will iterate through the 
	 //non-bound example starting at a random position.
   	 int i2;
  	 for (k = 0; k < end_support_i; k++)
  	 {
		 i2 = (k + k0) % end_support_i;//从随机位开始
		 if (alph[i2] > 0.0 && alph[i2] < C)//查找non-bound样本
		 {
			 if (takeStep(i1, i2))//As soon as there has positive progress, return 1.
			 {
				 return 1;
        	 }
		 }
	 }
	 return 0;
}
//hierarchy three：如果上面也失败，则从随机位置查找整个样本,(改为bound样本)
int examineBound(int i1)
{
  	 int k;
	 int k0 = rand()%end_support_i;
	 //If none of the non-bound example make positive progress, then hierarchy three starts at a random
	 //position in the entire training set and iterates through the entire set in finding the alph2 that
	 //will make positive progress in joint optimization.
   	 int i2;
  	 for (k = 0; k < end_support_i; k++)
  	 {
		 i2 = (k + k0) % end_support_i;//从随机位开始
		 //下面作为边界值的判定 被注释 但我认为本身的边界判断就是有误的
		 //if (alph[i2]= 0.0 || alph[i2]=C)//修改****************************************************
		 {
			 if (takeStep(i1, i2))//As soon as there has positive progress, return 1.
			 {
				 return 1;
			 }
		 }
	 }
	 return 0;	
}
//takeStep()
//用于优化两个乘子，成功，返回1，否则，返回0
//At every step, SMO chooses two Lagrange multipliers to jointly(共同地) optimize, 
//finds the optimal values for these multipliers, and updates the SVM to
//reflect the new optimal values.
int takeStep(int i1,int i2)
{
	int y1,y2,s;
	double alph1,alph2;//两个乘子的旧值
	double a1,a2;//两个乘子的新值
	double k11,k22,k12;
	double E1,E2,L,H,eta,Lobj,Hobj,delta_b;
	
	if(i1==i2) 
		return 0;//当两个样本相同，不进行优化。
	//给变量赋值
	alph1=alph[i1];
	alph2=alph[i2];
	y1=target[i1];
	y2=target[i2];
	if(alph1>0&&alph1<C)
		E1=error_cache[i1];//当作为非边界值 即他在支持向量上 与标签值的误差就是为0
	else
		E1=learned_func(i1)-y1;//learned_func(int)为非线性的评价函数，即输出函数
	if(alph2>0&&alph2<C)
		E2=error_cache[i2];
	else
		E2=learned_func(i2)-y2;
	s=y1*y2;
	//计算乘子的上下限
	//y1或y2的取值为1或-1(Binary Case)
	if(y1==y2)
	{
		double gamma=alph1+alph2;
		if(gamma>C)
		{
			L=gamma-C;
			H=C;
		}
		else
		{
			L=0;
			H=gamma;
		}
	}
	else
	{
		double gamma=alph1-alph2;
		if(gamma>0)
		{
			L=0;
			H=C-gamma;
		}
		else
		{
			L=-gamma;
			H=C;
		}
	}//计算乘子的上下限
	if(fabs(L-H) < eps)//L equals H
	{
		return 0;
	}
	//计算eta
	k11=kernel_func(i1,i1);//kernel_func(int,int)为核函数
	k22=kernel_func(i2,i2);
	k12=kernel_func(i1,i2);
	eta=2*k12-k11-k22;//eta是<=0的。
	if(eta<0)
	{
		a2=alph2-y2*(E1-E2)/eta;//计算新的alph2
		//调整a2，使其处于可行域
		if(a2<L)
		{
			a2=L;
		}
		if(a2>H)
		{
			a2=H;
		}
	}
	else//此时eta==0，得分别从端点H,L求目标函数值Lobj,Hobj，然后设a2为求得最大目标函数值的端点值
	{
		double c1=eta/2;
		double c2=y2*(E1-E2)-eta*alph2;
		Lobj=c1*L*L+c2*L;
		Hobj=c1*H*H+c2*H;
		if(Lobj>Hobj+eps)//eps==1e-3，是一个近似0的小数。
			a2=L;
		else if(Lobj<Hobj-eps)
			a2=H;
		else
			a2=alph2;//加eps的目的在于，使得Lobj与Hobj尽量分开，如果，很接近，就认为没有改进(make progress)
	}
	/*
	Under unusual circumstances, eta will not be negative. A zero eta can occur if more than one training
	example has the same input vector X. If eta==0, we need to evaluate the objective function at the two 
	endpoints, i.e. at L and H, and set a2(第二个乘子的新值) to be the one with larger objective function 
	value. The objective function is: obj=eta*a2^2/2 + (y2*(E1-E2)-eta*alph2)*a2+const
	*/
	if(fabs(a2-alph2)<eps*(a2+alph2+eps))
	{
		return 0;
	}
	/***********************************
    计算新的a1
	***********************************/
	a1=alph1-s*(a2-alph2);
	if(a1<0)//调整a1,使其符合条件*****??????????????????????????????????????????
	{
		a2+=s*a1;
		a1=0;
	}
	else if(a1>C)
	{
		double t=a1-C;
		a2+=s*t;
		a1=C;
	}
	//更新阀值b
	//After each step, b is re-computed.
	{
		double b1,b2,bnew;
		if(a1>0&&a1<C)
		{
			bnew=b+E1 + y1*(a1-alph1)*k11 + y2*(a2-alph2)*k12;
		    //The above threshold b is valid when the new alph1 is not at the bounds,
		    //because it forces the output of the SVM to be y1 when the input is vector X1.
		}
		else
		{
			if(a2>0&&a2<C)
				bnew=b+E2 + y1*(a1-alph1)*k12 + y2*(a2-alph2)*k22;
			    //The above threshold b is valis when the new alph2 is not at the bounds,
			    //because it forces the output of the SVM to be y2 when the input is vector x2.
			    //When both threshold of alph1 and alph2 are valid, they are equal.
			else
			{
				b1=b+E1+y1*(a1-alph1)*k11+y2*(a2-alph2)*k12;
				b2=b+E2+y1*(a1-alph1)*k12+y2*(a2-alph2)*k22;
				bnew=(b1+b2)/2;
				//When both new Lagrange multipliers are at bound and if L is not equal to H, 
				//then the interval between b1 and b2 are all thresholds that are consistent with
				//the KKT conditions. In this case, SMO chooses the threshold to be halfway in between
				//b1 and b2.
			}
		}
		delta_b=bnew-b;
		b=bnew;
	}
	
	//对于线性情况，要更新权向量，这里不用了
	//更新error_cache，对取得进展的a1,a2,所对应的i1,i2的error_cache[i1]=error_cache[i2]=0
	{
		double t1=y1*(a1-alph1);//两个乘子(新值-旧值)*标签值
		double t2=y2*(a2-alph2);
		for(int i=0;i<end_support_i;i++)//对于所有的alph中的非边界位置 进行误差更新
		{
			if(0<alph[i]&&alph[i]<C)//非边界值 对应的E 看不懂 所谓的更新
			{
				/*
				Whenever a joint optimization occurs, the cached errors for all non-bound
				multipliers alph[i] that are not involved in the optimization are updated.
				*/
				error_cache[i]+=t1*kernel_func(i1,i)+t2*(kernel_func(i2,i))-delta_b;
			}
		}
		error_cache[i1]=0.0;
		error_cache[i2]=0.0;
		/*
		When a Lagrange multiplier is non-bound and is unvolved in a joint optimization,
		its cached error is set to zero.
		*/
	}
	alph[i1]=a1;//store a1 in the alpha array
	alph[i2]=a2;//store a2 in the alpha array
	return 1;//说明已经取得进展
}

//learned_func(int)
//评价分类学习函数
double learned_func(int k)
{
	double s=0.0;
	for(int i=0;i<end_support_i;i++)
	{
		if(alph[i]>0)//alph[i]是属于[0, C]的。此行可省略。
		{
			s+=alph[i]*target[i]*kernel_func(i,k);
		}
	}
	s-=b;
	return s;
}
//计算点积函数dot_product_func(int,int)
double dot_product_func(int i1,int i2)
{
	double dot=0;
	for(int i=0;i<d;i++)
	{
		dot+=dense_points[i1][i]*dense_points[i2][i];
	}
	return dot;
}
//The kernel_func(int, int) is RBF(Radial-Basis Function).
//K(Xi, Xj)=exp(-||Xi-Xj||^2/(2*sigma^2))
double kernel_func(int i1,int i2)
{
	double s=dot_product_func(i1,i2);
	s*=-2;
	//s+=precomputed_self_dot_product[i1]+precomputed_self_dot_product[i2];//应用余弦定理
	s+=dot_product_func(i1,i1)+dot_product_func(i2,i2);
	return exp(-s/two_sigma_squared);
}
//初始化initialize()
void initialize()
{
	int i;
	//初始化阀值b为0
	b=0.0;
	//初始化alph[]为0
	for(i=0;i<end_support_i;i++)
	{
		alph[i]=0.0;
	}
	//设置样本值矩阵
	setX();
	//设置目标值向量
	setT();
	//设置预计算点积
	/*
	for(i=0;i<N;i++)//N为样品总数，包括训练样本与测试样本
	{
		precomputed_self_dot_product[i]=dot_product_func(i,i);
	}
	*/
}
//计算误差率error_rate()
double error_rate()
{	
	ofstream to("smo_test.txt");
	int tp=0,tn=0,fp=0,fn=0;
	double ming=0,te=0,total_q=0,temp=0;
	for(int i=first_test_i;i<N;i++)
	{	
		temp=learned_func(i);
		if(temp>0&&target[i]>0)
			tp++;
		else if(temp>0&&target[i]<0)
			fp++;
		else if(temp<0&&target[i]>0)
			fn++;
		else if(temp<0&&target[i]<0)
			tn++;
		to<<i<<"  实际输出"<<temp<<endl;
	}
	total_q=(double)(tp+tn)/(double)(tp+tn+fp+fn);//总精度
	ming=(double)tp/(double)(tp+fn);
	te=(double)tp/(double)(tp+fp);
	to<<"---------------测试结果-----------------"<<endl;
	to<<"tp="<<tp<<"   tn="<<tn<<"  fp="<<fp<<"  fn="<<fn<<endl;
	to<<"ming="<<ming<<"  te="<<te<<"  total_q="<<total_q<<endl;
	return (1-total_q);
}
//设置样本X[]
/*
void setX()
{
	//为了在需要时方便地检索要处理的数据，数据应保存在文件中。
	ifstream ff("17_smo.txt", ios::in);//ifstream用于从指定文件输入

	//exit program if ifstream could not open file
	if(!ff)//用!ff条件判断文件是否打开成功
	{
		cerr<<"File could not be opened!"<<endl;
		exit(1);//exit的作用为终止程序。
	}//end if

	int i=0,j=0;
	char ch;//ch为每次读到的字符
	while(ff.get(ch))
	{
		if(isspace(ch))//If ch is a white-space character, do nothing.
		{
			continue;
		}

		if(isdigit(ch))//isdigit returns a non-zero value if ch is a decimal digit(0-9).
		{
			dense_points[i][j]=(int)ch-48;//把输入的字符转换为数字
			j++;
			if(j==d)
			{
				j=0;
				i++;
			}
		}
	}
	ff.close();
}
*/
void setX()
{
	//为了在需要时方便地检索要处理的数据，数据应保存在文件中。
	ifstream inClientFile("data2713_adjusted.txt", ios::in);//ifstream用于从指定文件输入

	//exit program if ifstream could not open file
	if(!inClientFile)//用!inClientFile条件判断文件是否打开成功
	{
		cerr<<"File could not be opened!"<<endl;
		exit(1);//exit的作用为终止程序。
	}//end if

	int i=0,j=0;
	double a_data;//a_data为每次读到的数据, 默认为6位有效数字。
	while(inClientFile>>a_data)
	{
		dense_points[i][j]=a_data;
		j++;
		if(j==d)
		{
			j=0;
			i++;
		}
	}
	inClientFile.close();//显式关闭不再引用的文件。
}
//set targetT[]
void setT()
{
	//训练样本目标值
	for(int i=0;i<17;i++)
		target[i]=1;
	for(int i=17;i<27;i++)
		target[i]=-1;

	/*
	//测试样本目标值
	for(i=12090;i<13097;i++)
		target[i]=1;
	for(i=13097;i<14104;i++)
		target[i]=-1;
	*/
}