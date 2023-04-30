%2021 A Distance-Constrained Semantic Autoencoder for Zero-Shot Remote Sensing Scene Classification 
clear
clc
options = [];
options.PCARatio = 0.99;
%
%load('/Users/debm830/WorkSpace/matlab_ws/DSAE-master/gnd_UCM19.mat');
load('/Users/debm830/WorkSpace/matlab_ws/DSAE-master/ResNet152_Place360/gnd_UCM19.mat');

gnd = gnd_21n;
nnClass = length(unique(gnd_21n)); 

load('/Users/debm830/WorkSpace/matlab_ws/DSAE-master/Features/w2v_ucm21_300att.mat');
F_attribute = w2v_300_att;

%load('/Users/debm830/WorkSpace/matlab_ws/DSAE-master/feature_resnet152_place365_ucm19.mat');
load('/Users/debm830/WorkSpace/matlab_ws/DSAE-master/ResNet152_Place360/feature_resnet152.mat');

[~,~,~,new_data2] = PCA(featuresResNet152UCM19,options);
F_feature = normcols(new_data2');
clear features new_data2 gnd_n
 
accy1 = zeros(25,1);
accy2 = zeros(25,1);
for t = 1:25
%% Randomly select train samples and test samples
fprintf('Iteration %i:', t);
randIdx = randperm(nnClass);
seenClass = 16; 
X_tr_cl_id = randIdx(1:seenClass);
X_te_cl_id = randIdx(seenClass+1:end);
Ftr_feature = [];
Fte_feature = [];
Ftr_attributelabel = [];
Ftr_attributetabel = [];
Fte_attributetabel = [];
tr_label = [];
te_label = [];
for i = 1:length(X_tr_cl_id)
    idx = find(gnd == X_tr_cl_id(i));
    Ftr_feature = [Ftr_feature, F_feature(:,idx)];
    Ftr_attributetabel = [Ftr_attributetabel, F_attribute(:,X_tr_cl_id(i))];
    Ftr_attributelabel = [Ftr_attributelabel, repmat(F_attribute(:,X_tr_cl_id(i)), 1, length(idx))];
    tr_label = [tr_label; gnd(idx)]; 
end
for i = 1:length(X_te_cl_id)
    idx = find(gnd == X_te_cl_id(i));
    Fte_feature = [Fte_feature, F_feature(:,idx)];
    Fte_attributetabel = [Fte_attributetabel, F_attribute(:,X_te_cl_id(i))];
    te_label = [te_label; gnd(idx)]; 
end

%% Distance-Constrained Semantic Autoencoder
n = size(Ftr_feature,2);
sumw = 0;
sumb = 0;
for i = 1 : n
    sumw = sumw + length(find(tr_label == tr_label(i)));
    sumb = sumb + length(find(tr_label ~= tr_label(i)));
end
sw =[];
for i = 1:seenClass
    sw = blkdiag(sw, ones(length(find(tr_label == X_tr_cl_id(i)))));
end
lw = diag(sum(sw,2)) - sw;
sb = ones(n)- sw;
lb = diag(sum(sb,2)) - sb;
Rw = 2*Ftr_feature*lw*Ftr_feature'/sumw;
Rb = 2*Ftr_feature*lb*Ftr_feature'/sumb;
% Initialize
alpha = 100; 
beta = 10;  
lambda1 = 10000; 
lambda2 = 100;
lambda3 = 1000; 
As = Ftr_attributelabel;
Xs = Ftr_feature;
Xt = Fte_feature;
A = As * As';
B = alpha * Xs * Xs'+lambda1*(Rw-Rb);
C = (1+alpha) * As * Xs';
Ws = sylvester(A,B,C);
Wt = Ws;
% Update
for iteration = 1:50
    At = (1+beta)*inv(Wt*Wt'+beta*eye(size(Wt,1)))*(Wt*Xt);
    Wt = sylvester(lambda3*eye(size(At,1))+lambda2*At*At', lambda2*beta*Xt*Xt', lambda3*Ws+lambda2*(1+beta)*At*Xt');
    Ws = sylvester(lambda3*eye(size(As,1))+As*As', alpha*Xs*Xs'+lambda1*(Rw-Rb), lambda3*Wt+(1+alpha)*As*Xs');
end

%% Classification
% In the visual space
Fte_pre_feature = Wt' * Fte_attributetabel;
dis = zeros(size(Fte_pre_feature,2),size(Fte_feature,2));
for i = 1 : size(Fte_pre_feature,2)
    for j = 1 : size(Fte_feature,2)
         temp = dot(Fte_pre_feature(:,i),Fte_feature(:,j)) / (norm(Fte_pre_feature(:,i)) * norm(Fte_feature(:,j))); 
         dis(i,j) = 1 - temp;
    end
end
[~,index] = min(dis',[],2);
summ = 0;
for i = 1: length(index)
    if X_te_cl_id(index(i)) == te_label(i)
        summ = summ + 1;
    end
end
accy1(t) = summ / length(index); 

fprintf('\t accy1: %f,', accy1(t));

% In the semantic space
Fte_pre_attribute = Wt * Fte_feature;
dis = zeros(size(Fte_pre_attribute,2),size(Fte_attributetabel,2));
for i = 1 : size(Fte_pre_attribute,2)
    for j = 1 : size(Fte_attributetabel,2)
         temp = dot(Fte_pre_attribute(:,i),Fte_attributetabel(:,j)) / (norm(Fte_pre_attribute(:,i)) * norm(Fte_attributetabel(:,j))); 
         dis(i,j) = 1 - temp;
    end
end
[~,index] = min(dis,[],2);
summ = 0;
for i = 1: length(index)
    if X_te_cl_id(index(i)) == te_label(i)
        summ = summ + 1;
    end
end
accy2(t) = summ / length(index); 

fprintf('\t accy2: %f\n', accy2(t));
end
fprintf('---------------------------------------------------------\n')
fprintf('Average Accy:');
fprintf('\t accy1: %f,', mean(accy1));
fprintf('\t accy2: %f\n', mean(accy2));

