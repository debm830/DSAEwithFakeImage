% Extract Image Features Using Pretrained Network

%% Load Data UCM19 Dataset
%dataPath = '/Users/debm830/WorkSpace/Datasets/Remote_Sensing_Image/UCM_19/Images/train';
dataPath = '/Users/debm830/WorkSpace/Datasets/Remote_Sensing_Image/AID_30_220';

%% Image processing
imds = imageDatastore(dataPath,'IncludeSubfolders',true,'LabelSource','foldernames');

[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
    title(imdsTrain.Labels(idx(i)));
end

%% Load Pretrained Network rsnet101
net = resnet101;
inputSize = net.Layers(1).InputSize;
analyzeNetwork(net)

%augimdsUCM19 = augmentedImageDatastore(inputSize(1:2),imds);
augimds = augmentedImageDatastore(inputSize(1:2),imds);
layer = 'pool5';


features = activations(net,augimds,layer,OutputAs='rows');

%% Save featuresUCM19

save('featuresAID30','features')
gnd = imds.Labels;
whos gnd


%% analyzze
Class = unique(gnd); 
nnClass = length(Class);
gnd_n = [];
for i = 1 : length(gnd)
    gnd_n(end+1) = find(Class==gnd(i));
end
save('gnd_AID30',"gnd","gnd_n","Class")

