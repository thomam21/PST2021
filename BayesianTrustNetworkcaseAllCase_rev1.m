%* *****************************************************************************
%  *  Name:   Mini Thomas
%  *
%  *  Title:  Trust Inference using Bayesian Network
%  *  Description:  
%  *  This program computes the trust score using Bayesian Network(BN). 
%  *  The BN is represented by a DAG showing the relationship of each node
%  *  in the trust network. We generate conditional probability distribution
%  *  at each node (descendant and non-descendant node) using rand function. 
%  *  We consider three cases based on the pdf of the non-descendant 
%  *  node(x1 to x12):(1) worst case- the pdf of the nodes are wide spread with
%  *  standard deviation(SD) between 100-300.(2)best case when SD is between 1-3.
%  *  and (3)average case when pdfs of 6 non-descendant nodes have SD between 100-300
%  *  and remaining 6 have SD between 1-3. Two states are considered to evaluate 
%  *  trust: High and Low to answer questions such as 
%  *  what is probability that trust is high when its parents are high/low.
%  *
%  *  Written:       20March2021
%  *  Last updated:  17September2021
%  *
%  **************************************************************************** */
tic
clear;
n = input('Enter the case: -1 for Worst Case; : 0 for average, 1 for Best case ');

switch n
    case -1
       trustscore= trustcaseworst()
    case 0
        trustscore= trustcaseaverage()
    case 1
        trustscore= trustcasebest()
    otherwise
      trustscore= trustcaseaverage()
end

toc;

function trustscore = trustcaseaverage()
%  *
%  * A Bayesian network consists of a direct-acyclic graph (DAG) in which
%  * every node represents a variable and every edge represents a dependency
%  * between variables. We construct this graph by specifying an adjacency
%  * matrix where the element on row _i_ and column _j_ contains the number of
%  * edges directed from node _i_ to node _j_. 
%  *

adj = [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ;% adjacency matrix
       0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 ; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
nodeNames = {'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12','x13', 'x14', 'x15', 'x16', 'x17', 'x18','x19', 'x20', 'x21', 'x22'};   % nodes 
x1 = 1; x2 = 2; x3 = 3; x4 = 4; x5 = 5; x6=6;x7 = 7; x8 = 8; x9 = 9; x10 = 10; x11 = 11; x12=12;  x13 = 13; x14 = 14; x15 = 15; x16 = 16; x17 = 17; x18=18; x19 = 19; x20 = 20; x21=21;  x22=22;                 % node identifiers
n = numel(nodeNames);                       % number of nodes
t = 1; f  = 2;                              % true and false
values = cell(1,n);                         % values assumed by variables
for i = 1:numel(nodeNames)
    values{i} = [t f];                      
end

%  *
%  *draw the network
%  *
nodeLabels = {'x1', 'x2', 'x3','x4', 'x5', 'x6','x7', 'x8', 'x9','x10', 'x11', 'x12','x13', 'x14', 'x15','x16', 'x17', 'x18','x19', 'x20', 'x21','x22'};
bg = biograph(adj, nodeLabels, 'arrowsize', 6);
set(bg.Nodes, 'shape', 'ellipse');
bgInViewer = view(bg);

% %  *
% %  * save as figure
% %  *
% bgFig = figure;
% copyobj(bgInViewer.hgAxes,bgFig)
% %  *

%  *
%  * annotate using the CPT
%  *
[xp, xn] = find(adj);     % xp = parent id, xn = node id
pa(xn) = xp;              % parents
pa(1) = 1;                % root is parent of itself


%  *
%  * Generate mean and SD for all the non-descendant nodes within 
%  * specific range for each of the k iterations using rand function to
%  * define the pdfs of the non-descendant nodes
%  *
dx=0.01;x=0:dx:10; %scale
pp=0;
k=5; % number of iterations
for i=1:1:k 
%set range for each component using random command
a1=5;a2=4;a3=3;a4=2;a5=1;a6=3;a7=6;a8=7;a9=4;a10=6;a11=3;a12=2;
b1=10;b2=8;b3=7;b4=8;b5=5;b6=6;b7=10;b8=10;b9=9;b10=8;b11=6;b12=4;
mx1 = (b1-a1).*rand(1,1) + a1;
d1=5*mx1;
mx2 = (b2-a2).*rand(1,1) + a2;
d2=5*mx2;
mx3 = (b3-a3).*rand(1,1) + a3;
d3=5*mx3;
mx4 = (b4-a4).*rand(1,1) + a4;
d4=5*mx4;
mx5 = (b5-a5).*rand(1,1) + a5;
d5=5*mx5;
mx6 = (b6-a6).*rand(1,1) + a6;
d6=5*mx6;
mx7 = (b7-a7).*rand(1,1) + a7;
d7=5*mx7;
mx8 = (b8-a8).*rand(1,1) + a8;
d8=5*mx8;
mx9 = (b9-a9).*rand(1,1) + a9;
d9=5*mx9;
mx10 = (b10-a10).*rand(1,1) + a10;
d10=5*mx10;
mx11 = (b11-a11).*rand(1,1) + a11;
d11=5*mx11;
mx12 = (b12-a12).*rand(1,1) + a12;
d12=5*mx12;



%  *
%  * Generate pdfs of all non-descendant nodes x1 to x12 with the mean & SD 
%  *
p_x1=normpdf(x,mx1,d1);         %normpdf(x, mean, Std Dev)
p_x2=normpdf(x,mx2,d2);
p_x3=normpdf(x,mx3,d3);        
p_x4=normpdf(x,mx4,d4);
p_x5=normpdf(x,mx5,d5);
p_x6=normpdf(x,mx6,d6);
p_x7=normpdf(x,mx7,d7);
p_x8=normpdf(x,mx8,d8);
p_x9=normpdf(x,mx9,d9);
p_x10=normpdf(x,mx10,d10);
p_x11=normpdf(x,mx11,d11);
p_x12=normpdf(x,mx12,d12);
%figure(1)% visualize the plots
% plot(x,p_x1,'bo',x,p_x4,'ro',x,p_x6,'go',x,p_x8,'mo',x,p_x10,'co',x,p_x12,'yo')
% xlabel('x');
% ylabel('pdf');

%  *
%  * Generate conditional pdfs of all descendant nodes x13 to x22. Here we
%  * are multiplying by a scalar of 3 for first level pdfs and scalar of 5
%  * for second level pdfs and sclar of 7 for 3rd level.
%  *
%  * Conditional probabilities for all determinants 
%-----------------------
p_x13_x1_x2=3.*p_x1.*p_x2; 
p_x14_x3_x4=3.*p_x3.*p_x4; 
p_x15_x5_x6=3.*p_x5.*p_x6; 
p_x16_x7_x8=3.*p_x7.*p_x8; 
p_x17_x9_x10=3.*p_x9.*p_x10; 
p_x18_x11_x12=3.*p_x11.*p_x12; 
% * Conditional probabilities for all determinants 
%-----------------------
p_x19_x13_x14=5.*p_x13_x1_x2.*p_x14_x3_x4; 
p_x20_x15_x16=5.*p_x15_x5_x6.*p_x16_x7_x8; 
p_x21_x17_x18=5.*p_x17_x9_x10.*p_x18_x11_x12; 
% * Conditional Probability of Overall Trust 
%-----------------------
p_x22_x19_x20_x21=7.*p_x19_x13_x14.*p_x20_x15_x16.*p_x21_x17_x18;
% plot(x,p_x22_x19_x21_x22/max(p_x22_x19_x21_x22),'r-','LineWidth',2);%PLOT TRUST POSTERIOR
% hold on
% pause(3)

%  *
%  * Compute and plot posterior probability for selected nodes
%  *
Performance= p_x13_x1_x2.*p_x1.*p_x2; 
Reliability=p_x14_x3_x4.*p_x3.*p_x4.*p_x5; 
Operation=p_x15_x5_x6.*p_x6.*p_x7;
Standard= p_x16_x7_x8.*p_x7.*p_x8.*p_x9;
Reuse=p_x17_x9_x10.*p_x9.*p_x10; 
Protection=p_x18_x11_x12.*p_x11.*p_x12;
Robustness= p_x19_x13_x14.*p_x13_x1_x2.*p_x14_x3_x4.*p_x1.*p_x2.*p_x3.*p_x4.*p_x5; 
Security=p_x20_x15_x16.*p_x15_x5_x6.*p_x16_x7_x8.*p_x5.*p_x6.*p_x7.*p_x8.*p_x9; 
Privacy=p_x21_x17_x18.*p_x17_x9_x10.*p_x18_x11_x12.*p_x9.*p_x10.*p_x11.*p_x12; 
OverallTrust=p_x22_x19_x20_x21.*p_x19_x13_x14.*p_x20_x15_x16.*p_x21_x17_x18.*p_x13_x1_x2.*p_x14_x3_x4.*p_x15_x5_x6.*p_x16_x7_x8.*p_x17_x9_x10.*p_x18_x11_x12.*p_x1.*p_x2.*p_x3.*p_x4.*p_x5.*p_x6.*p_x7.*p_x8.*p_x9.*p_x10.*p_x11.*p_x12; 
% TrustWorstCase=OverallTrust;
% y=TrustWorstCase;
end 


%  *
%  * Plot posterior probability for selected nodes
%  *
figure 
plot(x,OverallTrust/max(OverallTrust),'r-','LineWidth',2);%plot the posteriors
hold on
plot(x,Robustness/max(Robustness),'b-','LineWidth',2);%PLOT TRUST POSTERIOR
hold on
plot(x,Security/max(Security),'m-','LineWidth',2);%PLOT TRUST POSTERIOR
hold on
plot(x,Privacy/max(Privacy),'c-','LineWidth',2);%PLOT TRUST POSTERIOR
title('Posterior Probability of selected nodes','FontSize', 16);
xlabel('Number of samples','FontSize', 16);
ylabel('Overall Trust probability','FontSize', 16);
title('Posterior Probability of selected nodes for Average Case');
    %title( sprintf('Mean value of Trust Posterior is %d .., with variance=%d', E_Pos, Var_Pos));
    legend(    'OverallTrust',...
               'Robustness',...
              'Security',...
              'Privacy'); 

%  *
%  * Compute the expected mean of the posterior pdf to get a score
%  *
X=1:1001;
E_x13=(trapz(X.*Performance)*dx)/k;
E_x14=(trapz(X.*Reliability)*dx)/k;
E_x15=(trapz(X.*Operation)*dx)/k;
E_x16=(trapz(X.*Standard)*dx)/k;
E_x17=(trapz(X.*Reuse)*dx)/k;
E_x18=(trapz(X.*Protection)*dx)/k;
E_x19=(trapz(X.*Robustness)*dx)/k;
E_x20=(trapz(X.*Security)*dx)/k;
E_x21=(trapz(X.*Privacy)*dx)/k;
E_x22=(trapz(X.*OverallTrust)*dx)/k;
trustscore= E_x22;

%  *
%  * Display the expected mean of the posterior pdfs
%  *
disp  ('Average Case Scores');
disp  ('--------------------');
fprintf('Performance Score is: %d\n',E_x13)
fprintf('Reliability Score is: %d\n',E_x14)
fprintf('Confidentiality Score is: %d\n',E_x15)
fprintf('Standards Score is: %d\n',E_x16)
fprintf('Data Protection Score is: %d\n',E_x17)
fprintf('Data Reuse Score is: %d\n',E_x18)
fprintf('Robustness Score is: %d\n',E_x19)
fprintf('Security Score is: %d\n',E_x20)
fprintf('Privacy Score is: %d\n',E_x21)
fprintf('Overall Trust Score is: %d\n',E_x22)


%  *
%  * Compute AUC of the trust posterior
%  *
AUCCase=trapz(X,OverallTrust);
AUCCaseLog=log(AUCCase);
disp  ('AUC '); 
disp  ('----');
fprintf('AUC of Overall Trust is: %d\n',AUCCaseLog)
end 



function trustscore = trustcaseworst()
%  *
%  * A Bayesian network consists of a direct-acyclic graph (DAG) in which
%  * every node represents a variable and every edge represents a dependency
%  * between variables. We construct this graph by specifying an adjacency
%  * matrix where the element on row _i_ and column _j_ contains the number of
%  * edges directed from node _i_ to node _j_. 
%  *

adj = [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ;% adjacency matrix
       0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 ; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
nodeNames = {'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12','x13', 'x14', 'x15', 'x16', 'x17', 'x18','x19', 'x20', 'x21', 'x22'};   % nodes 
x1 = 1; x2 = 2; x3 = 3; x4 = 4; x5 = 5; x6=6;x7 = 7; x8 = 8; x9 = 9; x10 = 10; x11 = 11; x12=12;  x13 = 13; x14 = 14; x15 = 15; x16 = 16; x17 = 17; x18=18; x19 = 19; x20 = 20; x21=21;  x22=22;                 % node identifiers
n = numel(nodeNames);                       % number of nodes
t = 1; f  = 2;                              % true and false
values = cell(1,n);                         % values assumed by variables
for i = 1:numel(nodeNames)
    values{i} = [t f];                      
end

%  *
%  *draw the network
%  *
nodeLabels = {'x1', 'x2', 'x3','x4', 'x5', 'x6','x7', 'x8', 'x9','x10', 'x11', 'x12','x13', 'x14', 'x15','x16', 'x17', 'x18','x19', 'x20', 'x21','x22'};
bg = biograph(adj, nodeLabels, 'arrowsize', 6);
set(bg.Nodes, 'shape', 'ellipse');
bgInViewer = view(bg);

% %  *
% %  * save as figure
% %  *
% bgFig = figure;
% copyobj(bgInViewer.hgAxes,bgFig)
% %  *

%  *
%  * annotate using the CPT
%  *
[xp, xn] = find(adj);     % xp = parent id, xn = node id
pa(xn) = xp;              % parents
pa(1) = 1;                % root is parent of itself

%  *
%  * Generate mean and SD for all the non-descendant nodes within 
%  * specific range for each of the k iterations using rand function to
%  * define the pdfs of the non-descendant nodes
%  *
dx=0.01;x=0:dx:10; %scale
pp=0;
k=5; % number of iterations
for i=1:1:k 
%set range for each component using random command
a1=5;a2=4;a3=3;a4=2;a5=1;a6=3;a7=6;a8=7;a9=4;a10=6;a11=3;a12=2;
b1=10;b2=8;b3=7;b4=8;b5=5;b6=6;b7=10;b8=10;b9=9;b10=8;b11=6;b12=4;
mx1 = (b1-a1).*rand(1,1) + a1;
d1=100*mx1;
mx2 = (b2-a2).*rand(1,1) + a2;
d2=100*mx2;
mx3 = (b3-a3).*rand(1,1) + a3;
d3=100*mx3;
mx4 = (b4-a4).*rand(1,1) + a4;
d4=100*mx4;
mx5 = (b5-a5).*rand(1,1) + a5;
d5=100*mx5;
mx6 = (b6-a6).*rand(1,1) + a6;
d6=100*mx6;
mx7 = (b7-a7).*rand(1,1) + a7;
d7=100*mx7;
mx8 = (b8-a8).*rand(1,1) + a8;
d8=100*mx8;
mx9 = (b9-a9).*rand(1,1) + a9;
d9=100*mx9;
mx10 = (b10-a10).*rand(1,1) + a10;
d10=100*mx10;
mx11 = (b11-a11).*rand(1,1) + a11;
d11=100*mx11;
mx12 = (b12-a12).*rand(1,1) + a12;
d12=100*mx12;


%  *
%  * Generate pdfs of all non-descendant nodes x1 to x12 with the mean & SD 
%  *
p_x1=normpdf(x,mx1,d1);         %normpdf(x, mean, Std Dev)
p_x2=normpdf(x,mx2,d2);
p_x3=normpdf(x,mx3,d3);        
p_x4=normpdf(x,mx4,d4);
p_x5=normpdf(x,mx5,d5);
p_x6=normpdf(x,mx6,d6);
p_x7=normpdf(x,mx7,d7);
p_x8=normpdf(x,mx8,d8);
p_x9=normpdf(x,mx9,d9);
p_x10=normpdf(x,mx10,d10);
p_x11=normpdf(x,mx11,d11);
p_x12=normpdf(x,mx12,d12);
%figure(1)% visualize the plots
% plot(x,p_x1,'bo',x,p_x4,'ro',x,p_x6,'go',x,p_x8,'mo',x,p_x10,'co',x,p_x12,'yo')
% xlabel('x');
% ylabel('pdf');

%  *
%  * Generate conditional pdfs of all descendant nodes x13 to x22. Here we
%  * are multiplying by a scalar of 3 for first level pdfs and scalar of 5
%  * for second level pdfs and sclar of 7 for 3rd level.
%  *
%  * Conditional probabilities for all determinants 
%-----------------------
p_x13_x1_x2=3.*p_x1.*p_x2; 
p_x14_x3_x4=3.*p_x3.*p_x4; 
p_x15_x5_x6=3.*p_x5.*p_x6; 
p_x16_x7_x8=3.*p_x7.*p_x8; 
p_x17_x9_x10=3.*p_x9.*p_x10; 
p_x18_x11_x12=3.*p_x11.*p_x12; 
% * Conditional probabilities for all determinants 
%-----------------------
p_x19_x13_x14=5.*p_x13_x1_x2.*p_x14_x3_x4; 
p_x20_x15_x16=5.*p_x15_x5_x6.*p_x16_x7_x8; 
p_x21_x17_x18=5.*p_x17_x9_x10.*p_x18_x11_x12; 
% * Conditional Probability of Overall Trust 
%-----------------------
p_x22_x19_x20_x21=7.*p_x19_x13_x14.*p_x20_x15_x16.*p_x21_x17_x18;
% plot(x,p_x22_x19_x21_x22/max(p_x22_x19_x21_x22),'r-','LineWidth',2);%PLOT TRUST POSTERIOR
% hold on
% pause(3)

%  *
%  * Compute posterior probability for selected nodes
%  *
Performance= p_x13_x1_x2.*p_x1.*p_x2; 
Reliability=p_x14_x3_x4.*p_x3.*p_x4.*p_x5; 
Operation=p_x15_x5_x6.*p_x6.*p_x7;
Standard= p_x16_x7_x8.*p_x7.*p_x8.*p_x9;
Reuse=p_x17_x9_x10.*p_x9.*p_x10; 
Protection=p_x18_x11_x12.*p_x11.*p_x12;
Robustness= p_x19_x13_x14.*p_x13_x1_x2.*p_x14_x3_x4.*p_x1.*p_x2.*p_x3.*p_x4.*p_x5; 
Security=p_x20_x15_x16.*p_x15_x5_x6.*p_x16_x7_x8.*p_x5.*p_x6.*p_x7.*p_x8.*p_x9; 
Privacy=p_x21_x17_x18.*p_x17_x9_x10.*p_x18_x11_x12.*p_x9.*p_x10.*p_x11.*p_x12; 
OverallTrust=p_x22_x19_x20_x21.*p_x19_x13_x14.*p_x20_x15_x16.*p_x21_x17_x18.*p_x13_x1_x2.*p_x14_x3_x4.*p_x15_x5_x6.*p_x16_x7_x8.*p_x17_x9_x10.*p_x18_x11_x12.*p_x1.*p_x2.*p_x3.*p_x4.*p_x5.*p_x6.*p_x7.*p_x8.*p_x9.*p_x10.*p_x11.*p_x12; 
% TrustWorstCase=OverallTrust;
% y=TrustWorstCase;
end 

%  *
%  * Plot posterior probability for selected nodes
%  *
figure 
plot(x,OverallTrust/max(OverallTrust),'r-','LineWidth',2);%plot the posteriors
hold on
plot(x,Robustness/max(Robustness),'b-','LineWidth',2);%PLOT TRUST POSTERIOR
hold on
plot(x,Security/max(Security),'m-','LineWidth',2);%PLOT TRUST POSTERIOR
hold on
plot(x,Privacy/max(Privacy),'c-','LineWidth',2);%PLOT TRUST POSTERIOR
title('Posterior Probability of selected nodes for Worst Case','FontSize', 16);
xlabel('Number of samples','FontSize', 16);
ylabel('Overall Trust probability','FontSize', 16);
title('PDF of Trust Quantification and Aggregation');
    %title( sprintf('Mean value of Trust Posterior is %d .., with variance=%d', E_Pos, Var_Pos));
    legend(    'OverallTrust',...
               'Robustness',...
              'Security',...
              'Privacy'); 


%  *
%  * Compute the expected mean of the posterior pdf to get a score
%  *
X=1:1001;
E_x13=(trapz(X.*Performance)*dx)/k;
E_x14=(trapz(X.*Reliability)*dx)/k;
E_x15=(trapz(X.*Operation)*dx)/k;
E_x16=(trapz(X.*Standard)*dx)/k;
E_x17=(trapz(X.*Reuse)*dx)/k;
E_x18=(trapz(X.*Protection)*dx)/k;
E_x19=(trapz(X.*Robustness)*dx)/k;
E_x20=(trapz(X.*Security)*dx)/k;
E_x21=(trapz(X.*Privacy)*dx)/k;
E_x22=(trapz(X.*OverallTrust)*dx)/k;
trustscore= E_x22;

%  *
%  * Display the expected mean of the posterior pdfs
%  *
disp  ('Worst Case Scores');
disp  ('--------------------');
fprintf('Performance Score is: %d\n',E_x13)
fprintf('Reliability Score is: %d\n',E_x14)
fprintf('Confidentiality Score is: %d\n',E_x15)
fprintf('Standards Score is: %d\n',E_x16)
fprintf('Data Protection Score is: %d\n',E_x17)
fprintf('Data Reuse Score is: %d\n',E_x18)
fprintf('Robustness Score is: %d\n',E_x19)
fprintf('Security Score is: %d\n',E_x20)
fprintf('Privacy Score is: %d\n',E_x21)
fprintf('Overall Trust Score is: %d\n',E_x22)


%  *
%  * Compute AUC of the trust posterior
%  *
AUCCase=trapz(X,OverallTrust);
AUCCaseLog=log(AUCCase);
disp  ('AUC '); 
disp  ('----');
fprintf('AUC of Overall Trust is: %d\n',AUCCaseLog)
end 





function trustscore = trustcasebest()
%  *
%  * A Bayesian network consists of a direct-acyclic graph (DAG) in which
%  * every node represents a variable and every edge represents a dependency
%  * between variables. We construct this graph by specifying an adjacency
%  * matrix where the element on row _i_ and column _j_ contains the number of
%  * edges directed from node _i_ to node _j_. 
%  *

adj = [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 ;% adjacency matrix
       0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 ; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1; 
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1;
       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
nodeNames = {'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12','x13', 'x14', 'x15', 'x16', 'x17', 'x18','x19', 'x20', 'x21', 'x22'};   % nodes 
x1 = 1; x2 = 2; x3 = 3; x4 = 4; x5 = 5; x6=6;x7 = 7; x8 = 8; x9 = 9; x10 = 10; x11 = 11; x12=12;  x13 = 13; x14 = 14; x15 = 15; x16 = 16; x17 = 17; x18=18; x19 = 19; x20 = 20; x21=21;  x22=22;                 % node identifiers
n = numel(nodeNames);                       % number of nodes
t = 1; f  = 2;                              % true and false
values = cell(1,n);                         % values assumed by variables
for i = 1:numel(nodeNames)
    values{i} = [t f];                      
end

%  *
%  *draw the network
%  *
nodeLabels = {'x1', 'x2', 'x3','x4', 'x5', 'x6','x7', 'x8', 'x9','x10', 'x11', 'x12','x13', 'x14', 'x15','x16', 'x17', 'x18','x19', 'x20', 'x21','x22'};
bg = biograph(adj, nodeLabels, 'arrowsize', 6);
set(bg.Nodes, 'shape', 'ellipse');
bgInViewer = view(bg);

% %  *
% %  * save as figure
% %  *
% bgFig = figure;
% copyobj(bgInViewer.hgAxes,bgFig)
%  *

%  *
%  * annotate using the CPT
%  *
[xp, xn] = find(adj);     % xp = parent id, xn = node id
pa(xn) = xp;              % parents
pa(1) = 1;                % root is parent of itself

%  *
%  * Generate mean and SD for all the non-descendant nodes within 
%  * specific range for each of the k iterations using rand function to
%  * define the pdfs of the non-descendant nodes
%  *
dx=0.01;x=0:dx:10; %scale
pp=0;
k=5; % number of iterations
for i=1:1:k 
%set range for each component using random command
a1=5;a2=4;a3=3;a4=5;a5=2;a6=4;a7=6;a8=7;a9=6;a10=6;a11=3;a12=2;
b1=8;b2=6;b3=5;b4=8;b5=3;b6=6;b7=8;b8=9;b9=8;b10=8;b11=6;b12=4;
mx1 = (b1-a1).*rand(1,1) + a1;
d1=0.5*mx1;
mx2 = (b2-a2).*rand(1,1) + a2;
d2=0.5*mx2;
mx3 = (b3-a3).*rand(1,1) + a3;
d3=0.5*mx3;
mx4 = (b4-a4).*rand(1,1) + a4;
d4=0.5*mx4;
mx5 = (b5-a5).*rand(1,1) + a5;
d5=0.5*mx5;
mx6 = (b6-a6).*rand(1,1) + a6;
d6=0.5*mx6;
mx7 = (b7-a7).*rand(1,1) + a7;
d7=0.5*mx7;
mx8 = (b8-a8).*rand(1,1) + a8;
d8=0.5*mx8;
mx9 = (b9-a9).*rand(1,1) + a9;
d9=0.5*mx9;
mx10 = (b10-a10).*rand(1,1) + a10;
d10=0.5*mx10;
mx11 = (b11-a11).*rand(1,1) + a11;
d11=0.5*mx11;
mx12 = (b12-a12).*rand(1,1) + a12;
d12=0.5*mx12;


%  *
%  * Generate pdfs of all non-descendant nodes x1 to x12 with the mean & SD 
%  *
p_x1=normpdf(x,mx1,d1);         %normpdf(x, mean, Std Dev)
p_x2=normpdf(x,mx2,d2);
p_x3=normpdf(x,mx3,d3);        
p_x4=normpdf(x,mx4,d4);
p_x5=normpdf(x,mx5,d5);
p_x6=normpdf(x,mx6,d6);
p_x7=normpdf(x,mx7,d7);
p_x8=normpdf(x,mx8,d8);
p_x9=normpdf(x,mx9,d9);
p_x10=normpdf(x,mx10,d10);
p_x11=normpdf(x,mx11,d11);
p_x12=normpdf(x,mx12,d12);
%figure(1)% visualize the plots
% plot(x,p_x1,'bo',x,p_x4,'ro',x,p_x6,'go',x,p_x8,'mo',x,p_x10,'co',x,p_x12,'yo')
% xlabel('x');
% ylabel('pdf');

%  *
%  * Generate conditional pdfs of all descendant nodes x13 to x22. Here we
%  * are multiplying by a scalar of 3 for first level pdfs and scalar of 5
%  * for second level pdfs and sclar of 7 for 3rd level.
%  *
%  * Conditional probabilities for all determinants 
%-----------------------
p_x13_x1_x2=3.*p_x1.*p_x2; 
p_x14_x3_x4=3.*p_x3.*p_x4; 
p_x15_x5_x6=3.*p_x5.*p_x6; 
p_x16_x7_x8=3.*p_x7.*p_x8; 
p_x17_x9_x10=3.*p_x9.*p_x10; 
p_x18_x11_x12=3.*p_x11.*p_x12; 
% * Conditional probabilities for all determinants 
%-----------------------
p_x19_x13_x14=5.*p_x13_x1_x2.*p_x14_x3_x4; 
p_x20_x15_x16=5.*p_x15_x5_x6.*p_x16_x7_x8; 
p_x21_x17_x18=5.*p_x17_x9_x10.*p_x18_x11_x12; 
% * Conditional Probability of Overall Trust 
%-----------------------
p_x22_x19_x20_x21=7.*p_x19_x13_x14.*p_x20_x15_x16.*p_x21_x17_x18;
% plot(x,p_x22_x19_x21_x22/max(p_x22_x19_x21_x22),'r-','LineWidth',2);%PLOT TRUST POSTERIOR
% hold on
% pause(3)

%  *
%  * Compute posterior probability for selected nodes
%  *
Performance= p_x13_x1_x2.*p_x1.*p_x2; 
Reliability=p_x14_x3_x4.*p_x3.*p_x4.*p_x5; 
Operation=p_x15_x5_x6.*p_x6.*p_x7;
Standard= p_x16_x7_x8.*p_x7.*p_x8.*p_x9;
Reuse=p_x17_x9_x10.*p_x9.*p_x10; 
Protection=p_x18_x11_x12.*p_x11.*p_x12;
Robustness= p_x19_x13_x14.*p_x13_x1_x2.*p_x14_x3_x4.*p_x1.*p_x2.*p_x3.*p_x4.*p_x5; 
Security=p_x20_x15_x16.*p_x15_x5_x6.*p_x16_x7_x8.*p_x5.*p_x6.*p_x7.*p_x8.*p_x9; 
Privacy=p_x21_x17_x18.*p_x17_x9_x10.*p_x18_x11_x12.*p_x9.*p_x10.*p_x11.*p_x12; 
OverallTrust=p_x22_x19_x20_x21.*p_x19_x13_x14.*p_x20_x15_x16.*p_x21_x17_x18.*p_x13_x1_x2.*p_x14_x3_x4.*p_x15_x5_x6.*p_x16_x7_x8.*p_x17_x9_x10.*p_x18_x11_x12.*p_x1.*p_x2.*p_x3.*p_x4.*p_x5.*p_x6.*p_x7.*p_x8.*p_x9.*p_x10.*p_x11.*p_x12; 
% TrustWorstCase=OverallTrust;
% y=TrustWorstCase;
end 

%  *
%  * Plot posterior probability for selected nodes
%  *
figure
plot(x,OverallTrust/max(OverallTrust),'r-','LineWidth',2);%plot the posteriors
hold on
plot(x,Robustness/max(Robustness),'b-','LineWidth',2);%PLOT TRUST POSTERIOR
hold on
plot(x,Security/max(Security),'m-','LineWidth',2);%PLOT TRUST POSTERIOR
hold on
plot(x,Privacy/max(Privacy),'c-','LineWidth',2);%PLOT TRUST POSTERIOR
title('Posterior Probability of selected nodes for Worst Case','FontSize', 16);
xlabel('Number of samples','FontSize', 16);
ylabel('Overall Trust probability','FontSize', 16);
title('PDF of Trust Quantification and Aggregation');
    %title( sprintf('Mean value of Trust Posterior is %d .., with variance=%d', E_Pos, Var_Pos));
    legend(    'OverallTrust',...
               'Robustness',...
              'Security',...
              'Privacy'); 


%  *
%  * Compute the expected mean of the posterior pdf to get a score
%  *
X=1:1001;
E_x13=(trapz(X.*Performance)*dx)/k;
E_x14=(trapz(X.*Reliability)*dx)/k;
E_x15=(trapz(X.*Operation)*dx)/k;
E_x16=(trapz(X.*Standard)*dx)/k;
E_x17=(trapz(X.*Reuse)*dx)/k;
E_x18=(trapz(X.*Protection)*dx)/k;
E_x19=(trapz(X.*Robustness)*dx)/k;
E_x20=(trapz(X.*Security)*dx)/k;
E_x21=(trapz(X.*Privacy)*dx)/k;
E_x22=(trapz(X.*OverallTrust)*dx)/k;
trustscore= E_x22;

%  *
%  * Display the expected mean of the posterior pdfs
%  *
disp  ('Best Case Scores');
disp  ('--------------------');
fprintf('Performance Score is: %d\n',E_x13)
fprintf('Reliability Score is: %d\n',E_x14)
fprintf('Confidentiality Score is: %d\n',E_x15)
fprintf('Standards Score is: %d\n',E_x16)
fprintf('Data Protection Score is: %d\n',E_x17)
fprintf('Data Reuse Score is: %d\n',E_x18)
fprintf('Robustness Score is: %d\n',E_x19)
fprintf('Security Score is: %d\n',E_x20)
fprintf('Privacy Score is: %d\n',E_x21)
fprintf('Overall Trust Score is: %d\n',E_x22)


%  *
%  * Compute AUC of the trust posterior
%  *
AUCCase=trapz(X,OverallTrust);
AUCCaseLog=log(AUCCase);
disp  ('AUC '); 
disp  ('----');
fprintf('AUC of Overall Trust is: %d\n',AUCCaseLog)
end 







