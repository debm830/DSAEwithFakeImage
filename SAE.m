clear
clc
options = [];
options.PCARatio = 0.99;
%
load('D:\images\UCM_21\gnd_21');
gnd = gnd_21;
clear gnd_21
nnClass = length(unique(gnd));
%
load('D:\untitled\attribute21');
F_attribute = feature'; 
%
load('D:\images\UCM_21\\resnet152_21');
[~,~,~,new_data2] = PCA(resnet152_21',options);
F_feature = normcols(new_data2');
clear resnet152_21 new_data2

accy1 = zeros(25,1);
accy2 = zeros(25,1);
for t = 1:25
%随机选取训练类别和测试类别
randIdx = randperm(nnClass);
seenClass = 16; 
X_tr_cl_id = randIdx(1:seenClass);
X_te_cl_id = randIdx(seenClass+1:end);
%
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

% SAE
lamda = 1000;  
S = Ftr_attributelabel;
X = Ftr_feature;
A = S * S';
B = lamda * X * X';
C = (1+lamda) * S * X'; 
Wt = sylvester(A,B,C);

% visual space
Fte_pre_feature = Wt' * Fte_attributetabel;
dis = zeros(size(Fte_pre_feature,2),size(Fte_feature,2));
for i = 1 : size(Fte_pre_feature,2)
    for j = 1 : size(Fte_feature,2)
         temp = dot(Fte_pre_feature(:,i),Fte_feature(:,j)) / (norm(Fte_pre_feature(:,i)) * norm(Fte_feature(:,j))); 
         dis(i,j) = 1 - temp;
    end
end
[~,index] = min(dis',[],2);
sum = 0;
for i = 1: length(index)
    if X_te_cl_id(index(i)) == te_label(i)
        sum = sum + 1;
    end
end
accy1(t) = sum / length(index); 

% semantic space
Fte_pre_attribute = Wt * Fte_feature;
dis = zeros(size(Fte_pre_attribute,2),size(Fte_attributetabel,2));
for i = 1 : size(Fte_pre_attribute,2)
    for j = 1 : size(Fte_attributetabel,2)
         temp = dot(Fte_pre_attribute(:,i),Fte_attributetabel(:,j)) / (norm(Fte_pre_attribute(:,i)) * norm(Fte_attributetabel(:,j))); 
         dis(i,j) = 1 - temp;
    end
end
[~,index] = min(dis,[],2);
sum = 0;
for i = 1: length(index)
    if X_te_cl_id(index(i)) == te_label(i)
        sum = sum + 1;
    end
end
accy2(t) = sum / length(index); 
end

