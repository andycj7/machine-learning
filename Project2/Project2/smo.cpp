//C-Support Vector Classification(Binary Case)
//The advantage of SMO(Sequential Minimal Optimization) Algorithm lies in the fact that solving for
//two Lagrange multipliers can be done analytically.

/*
There are three components to SMO:
(1)an analytic method to solve for the two Lagrange multipliers.
(2)a heuristic(����ʽ��)for choosing which multipliers to optimize:
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
	ofstream outClientFile("data_result.txt", ios::out);//���ָ�����ļ�data_result.txt�����ڣ�ofstream���ø��ļ�����������
	int numChanged=0;
	int examineAll=1;
	//srand((unsigned int)time(NULL));
	initialize();
	//��������ѭ������ʼʱ�������������ѡ�񲻷���KKT�������������ӽ����Ż���ѡ��ɹ�������1�����򣬷���0
	//���Գɹ��ˣ�numChanged��Ȼ>0���ӵڶ���ѭ��ʱ����������������ȥѰ�Ҳ�����KKT�������������ӽ����Ż���
	//���ǴӷǱ߽������ȥѰ�ң���Ϊ�Ǳ߽�������Ҫ�����Ŀ����Ը��󣬱߽�������������������ʼ��ͣ���ڱ߽��ϡ�
	//���û�ҵ����ٴ�����������ȥ�ң�ֱ��������������Ҳ�Ҳ�����Ҫ�ı�ĳ���Ϊֹ����ʱ�㷨������
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
	//���ѵ����Ĳ���
	{
		outClientFile<<"d="<<d<<endl;//ά��
		outClientFile<<"b="<<b<<endl;//threshold
		outClientFile<<"two_sigma_squared="<<two_sigma_squared<<endl;
		int n_support_vectors=0;
		for(int i=0;i<end_support_i;i++)
		{
			if(alph[i]>0)//�˵ط��Ƿ���Ը�Ϊalph[i]>tolerance?????????????????
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
	//������Լ�����
	cout<<error_rate()<<endl;
	while (1);
	return 0;
}

/////////examineExample����
//�ٶ���һ������ai(λ��Ϊi1)��examineExample(i1)���ȼ�飬���������tolerance��Υ��KKT��������ô���ͳ�Ϊ��һ�����ӣ�
//Ȼ��Ѱ�ҵڶ�������(λ��Ϊi2),ͨ������takeStep(i1,i2)���Ż����������ӡ�
int examineExample(int i1)
{
	int y1;
	double alph1,E1,r1;
	y1=target[i1];
	alph1=alph[i1];
	//alphi��initialize();�еĳ�ʼ��Ϊ0.0��
	if(alph1>0&&alph1<C)
		E1=error_cache[i1];
		/*
		A cached error value E is kept for every non-bound training example from 
		which example can be chosed for maximizing the step size.
		*/
	else
		E1=learned_func(i1)-y1;//learned_funcΪ�����������
	/*
	When an error E is required by SMO, it will look up the error in the error cache if the 
	corresponding Lagrange multiplier is not at bound. Otherwise, it will evaluate the current
	SVM decision function based on the current alph vector.
	*/
	r1=y1*E1;
	//Υ��KKT(Karush-Kuhn-Tucker)�������ж�
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
		/*ʹ�����ַ���ѡ��ڶ�������:
		hierarchy one����non-bound������Ѱ��maximum fabs(E1-E2)������
		hierarchy two���������ûȡ�ý�չ,��ô�����λ�ò���non-boundary ����
		hierarchy three���������Ҳʧ�ܣ�������λ�ò�����������,��Ϊbound����
		���Ϸ��� �Ǳ߽�ֵ���� ��������ѡȡһ���߽�ֵ
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
	///û�н�չ
	return 0;
}

//hierarchy one����non-bound������Ѱ��maximum fabs(E1-E2)������
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
   if(i2>=0)//���û��non-bound multiplier��i2==-1��
   {
	   if(takeStep(i1,i2))//If there has a positive progress, return 1.
	   {
		   return 1;
	   }
   }
   return 0;
}
//hierarchy two���������ûȡ�ý�չ,��ô�����λ�ò���non-boundary ����
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
		 i2 = (k + k0) % end_support_i;//�����λ��ʼ
		 if (alph[i2] > 0.0 && alph[i2] < C)//����non-bound����
		 {
			 if (takeStep(i1, i2))//As soon as there has positive progress, return 1.
			 {
				 return 1;
        	 }
		 }
	 }
	 return 0;
}
//hierarchy three���������Ҳʧ�ܣ�������λ�ò�����������,(��Ϊbound����)
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
		 i2 = (k + k0) % end_support_i;//�����λ��ʼ
		 //������Ϊ�߽�ֵ���ж� ��ע�� ������Ϊ����ı߽��жϾ��������
		 //if (alph[i2]= 0.0 || alph[i2]=C)//�޸�****************************************************
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
//�����Ż��������ӣ��ɹ�������1�����򣬷���0
//At every step, SMO chooses two Lagrange multipliers to jointly(��ͬ��) optimize, 
//finds the optimal values for these multipliers, and updates the SVM to
//reflect the new optimal values.
int takeStep(int i1,int i2)
{
	int y1,y2,s;
	double alph1,alph2;//�������ӵľ�ֵ
	double a1,a2;//�������ӵ���ֵ
	double k11,k22,k12;
	double E1,E2,L,H,eta,Lobj,Hobj,delta_b;
	
	if(i1==i2) 
		return 0;//������������ͬ���������Ż���
	//��������ֵ
	alph1=alph[i1];
	alph2=alph[i2];
	y1=target[i1];
	y2=target[i2];
	if(alph1>0&&alph1<C)
		E1=error_cache[i1];//����Ϊ�Ǳ߽�ֵ ������֧�������� ���ǩֵ��������Ϊ0
	else
		E1=learned_func(i1)-y1;//learned_func(int)Ϊ�����Ե����ۺ��������������
	if(alph2>0&&alph2<C)
		E2=error_cache[i2];
	else
		E2=learned_func(i2)-y2;
	s=y1*y2;
	//������ӵ�������
	//y1��y2��ȡֵΪ1��-1(Binary Case)
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
	}//������ӵ�������
	if(fabs(L-H) < eps)//L equals H
	{
		return 0;
	}
	//����eta
	k11=kernel_func(i1,i1);//kernel_func(int,int)Ϊ�˺���
	k22=kernel_func(i2,i2);
	k12=kernel_func(i1,i2);
	eta=2*k12-k11-k22;//eta��<=0�ġ�
	if(eta<0)
	{
		a2=alph2-y2*(E1-E2)/eta;//�����µ�alph2
		//����a2��ʹ�䴦�ڿ�����
		if(a2<L)
		{
			a2=L;
		}
		if(a2>H)
		{
			a2=H;
		}
	}
	else//��ʱeta==0���÷ֱ�Ӷ˵�H,L��Ŀ�꺯��ֵLobj,Hobj��Ȼ����a2Ϊ������Ŀ�꺯��ֵ�Ķ˵�ֵ
	{
		double c1=eta/2;
		double c2=y2*(E1-E2)-eta*alph2;
		Lobj=c1*L*L+c2*L;
		Hobj=c1*H*H+c2*H;
		if(Lobj>Hobj+eps)//eps==1e-3����һ������0��С����
			a2=L;
		else if(Lobj<Hobj-eps)
			a2=H;
		else
			a2=alph2;//��eps��Ŀ�����ڣ�ʹ��Lobj��Hobj�����ֿ���������ܽӽ�������Ϊû�иĽ�(make progress)
	}
	/*
	Under unusual circumstances, eta will not be negative. A zero eta can occur if more than one training
	example has the same input vector X. If eta==0, we need to evaluate the objective function at the two 
	endpoints, i.e. at L and H, and set a2(�ڶ������ӵ���ֵ) to be the one with larger objective function 
	value. The objective function is: obj=eta*a2^2/2 + (y2*(E1-E2)-eta*alph2)*a2+const
	*/
	if(fabs(a2-alph2)<eps*(a2+alph2+eps))
	{
		return 0;
	}
	/***********************************
    �����µ�a1
	***********************************/
	a1=alph1-s*(a2-alph2);
	if(a1<0)//����a1,ʹ���������*****??????????????????????????????????????????
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
	//���·�ֵb
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
	
	//�������������Ҫ����Ȩ���������ﲻ����
	//����error_cache����ȡ�ý�չ��a1,a2,����Ӧ��i1,i2��error_cache[i1]=error_cache[i2]=0
	{
		double t1=y1*(a1-alph1);//��������(��ֵ-��ֵ)*��ǩֵ
		double t2=y2*(a2-alph2);
		for(int i=0;i<end_support_i;i++)//�������е�alph�еķǱ߽�λ�� ����������
		{
			if(0<alph[i]&&alph[i]<C)//�Ǳ߽�ֵ ��Ӧ��E ������ ��ν�ĸ���
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
	return 1;//˵���Ѿ�ȡ�ý�չ
}

//learned_func(int)
//���۷���ѧϰ����
double learned_func(int k)
{
	double s=0.0;
	for(int i=0;i<end_support_i;i++)
	{
		if(alph[i]>0)//alph[i]������[0, C]�ġ����п�ʡ�ԡ�
		{
			s+=alph[i]*target[i]*kernel_func(i,k);
		}
	}
	s-=b;
	return s;
}
//����������dot_product_func(int,int)
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
	//s+=precomputed_self_dot_product[i1]+precomputed_self_dot_product[i2];//Ӧ�����Ҷ���
	s+=dot_product_func(i1,i1)+dot_product_func(i2,i2);
	return exp(-s/two_sigma_squared);
}
//��ʼ��initialize()
void initialize()
{
	int i;
	//��ʼ����ֵbΪ0
	b=0.0;
	//��ʼ��alph[]Ϊ0
	for(i=0;i<end_support_i;i++)
	{
		alph[i]=0.0;
	}
	//��������ֵ����
	setX();
	//����Ŀ��ֵ����
	setT();
	//����Ԥ������
	/*
	for(i=0;i<N;i++)//NΪ��Ʒ����������ѵ���������������
	{
		precomputed_self_dot_product[i]=dot_product_func(i,i);
	}
	*/
}
//���������error_rate()
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
		to<<i<<"  ʵ�����"<<temp<<endl;
	}
	total_q=(double)(tp+tn)/(double)(tp+tn+fp+fn);//�ܾ���
	ming=(double)tp/(double)(tp+fn);
	te=(double)tp/(double)(tp+fp);
	to<<"---------------���Խ��-----------------"<<endl;
	to<<"tp="<<tp<<"   tn="<<tn<<"  fp="<<fp<<"  fn="<<fn<<endl;
	to<<"ming="<<ming<<"  te="<<te<<"  total_q="<<total_q<<endl;
	return (1-total_q);
}
//��������X[]
/*
void setX()
{
	//Ϊ������Ҫʱ����ؼ���Ҫ��������ݣ�����Ӧ�������ļ��С�
	ifstream ff("17_smo.txt", ios::in);//ifstream���ڴ�ָ���ļ�����

	//exit program if ifstream could not open file
	if(!ff)//��!ff�����ж��ļ��Ƿ�򿪳ɹ�
	{
		cerr<<"File could not be opened!"<<endl;
		exit(1);//exit������Ϊ��ֹ����
	}//end if

	int i=0,j=0;
	char ch;//chΪÿ�ζ������ַ�
	while(ff.get(ch))
	{
		if(isspace(ch))//If ch is a white-space character, do nothing.
		{
			continue;
		}

		if(isdigit(ch))//isdigit returns a non-zero value if ch is a decimal digit(0-9).
		{
			dense_points[i][j]=(int)ch-48;//��������ַ�ת��Ϊ����
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
	//Ϊ������Ҫʱ����ؼ���Ҫ��������ݣ�����Ӧ�������ļ��С�
	ifstream inClientFile("data2713_adjusted.txt", ios::in);//ifstream���ڴ�ָ���ļ�����

	//exit program if ifstream could not open file
	if(!inClientFile)//��!inClientFile�����ж��ļ��Ƿ�򿪳ɹ�
	{
		cerr<<"File could not be opened!"<<endl;
		exit(1);//exit������Ϊ��ֹ����
	}//end if

	int i=0,j=0;
	double a_data;//a_dataΪÿ�ζ���������, Ĭ��Ϊ6λ��Ч���֡�
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
	inClientFile.close();//��ʽ�رղ������õ��ļ���
}
//set targetT[]
void setT()
{
	//ѵ������Ŀ��ֵ
	for(int i=0;i<17;i++)
		target[i]=1;
	for(int i=17;i<27;i++)
		target[i]=-1;

	/*
	//��������Ŀ��ֵ
	for(i=12090;i<13097;i++)
		target[i]=1;
	for(i=13097;i<14104;i++)
		target[i]=-1;
	*/
}