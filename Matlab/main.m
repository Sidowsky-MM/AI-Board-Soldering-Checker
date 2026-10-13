%% MAIN.M — Detekcja wad PCB (CNN od zera + SVM(HOG)+KFold CV + ResNet50)

clc; clear; close all;
fprintf('Źródło danych: Solder Joint Dataset (Kaggle)\n');
fprintf('https://www.kaggle.com/datasets/akshaydattatraykhare/solder-joint-defect-detection\n\n');

%% Informacja o bazie danych
% Nazwa bazy: Solder Joint Dataset
% Źródło: Kaggle
% Link: https://www.kaggle.com/datasets/akshaydattatraykhare/solder-joint-defect-detection


%% ===== USTAWIENIA (pod plakat) =====
EXPORT_DPI      = 450;
BIG_FONTSIZE    = 22;
LABEL_FONTSIZE  = 24;
TITLE_FONTSIZE  = 30;

% Walidacja krzyżowa SVM:
K_FOLDS_SVM     = 5;

fprintf('Źródło danych: Solder Joint Dataset (Kaggle)\n');
fprintf('https://www.kaggle.com/datasets/akshaydattatraykhare/solder-joint-defect-detection\n\n');

%% Informacja o bazie danych
% Nazwa bazy: Solder Joint Dataset
% Źródło: Kaggle
% Link: https://www.kaggle.com/datasets/akshaydattatraykhare/solder-joint-defect-detection


%% 1. Konfiguracja danych
fprintf('[1/7] Wczytanie danych...\n');

glownaSciezka = 'H:/ai_projekt/SolderJointDataset';
imds = imageDatastore(glownaSciezka, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[imdsTrain, imdsVal, imdsTest] = splitEachLabel(imds, 0.7, 0.15, 0.15, 'randomized');

classes = categories(imdsTrain.Labels);
numClasses = numel(classes);

fprintf('[1/7] Klasy: %d | Train: %d | Val: %d | Test: %d\n', ...
    numClasses, numel(imdsTrain.Files), numel(imdsVal.Files), numel(imdsTest.Files));

fprintf('[INFO] WSPÓLNY TEST dla wszystkich metod: %d obrazów.\n', numel(imdsTest.Files));

%% 2. Augmentacja danych
fprintf('[2/7] Augmentacja...\n');

augmenter = imageDataAugmenter( ...
    'RandRotation',[-10 10], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5]);

augTrain = augmentedImageDatastore([64 64 3], imdsTrain, 'DataAugmentation', augmenter);
augVal   = augmentedImageDatastore([64 64 3], imdsVal);
augTest  = augmentedImageDatastore([64 64 3], imdsTest);

fprintf('[2/7] OK\n');

%% 3. Własna CNN (od zera)
fprintf('[3/7] Trening własnej CNN...\n');

layersOwn = [
    imageInputLayer([64 64 3], 'Name','input')

    convolution2dLayer(3, 32, 'Padding','same', 'Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','pool1')

    convolution2dLayer(3, 64, 'Padding','same', 'Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2,'Stride',2,'Name','pool2')

    convolution2dLayer(3, 128, 'Padding','same', 'Name','conv3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')
    maxPooling2dLayer(2,'Stride',2,'Name','pool3')

    dropoutLayer(0.5,'Name','drop')
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','sm')
    classificationLayer('Name','out')
];

optionsOwn = trainingOptions('adam', ...
    'MiniBatchSize',128, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augVal, ...
    'ValidationFrequency',30, ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'OutputFcn',@(info)saveTrainingPlotCNN(info,'cnn_own',EXPORT_DPI,BIG_FONTSIZE), ...
    'ExecutionEnvironment','gpu');

netOwn = trainNetwork(augTrain, layersOwn, optionsOwn);
fprintf('[3/7] CNN zakończona.\n');

%% 4. Ewaluacja CNN (na WSPÓLNYM teście)
fprintf('[4/7] Ewaluacja CNN...\n');

yTrue = imdsTest.Labels;
yPredCNN = classify(netOwn, augTest);

accCNN = mean(yPredCNN == yTrue);
f1CNN_macro = f1_macro(yTrue, yPredCNN);

% Confusion Matrix CNN
fig = figure('Units','pixels','Position',[50 50 2400 1500]);
cm = confusionchart(yTrue, yPredCNN);
cm.Title = 'Macierz pomyłek - Własna CNN';
cm.FontSize = BIG_FONTSIZE;
cm.XLabel = 'Predykcja';
cm.YLabel = 'Klasa rzeczywista';
exportgraphics(fig, 'confmat_cnn_own.png', 'Resolution', EXPORT_DPI);
close(fig);

fprintf('[4/7] CNN Acc=%.2f%% | Macro-F1=%.3f\n', accCNN*100, f1CNN_macro);

%% 5. SVM (HOG) + Walidacja krzyżowa (K-Fold) + test WSPÓLNY
fprintf('[5/7] SVM (HOG) + %d-fold CV...\n', K_FOLDS_SVM);

hogSize = [64 64];

% --- HOG TRAIN (imdsTrain) ---
XTrain = [];
for i = 1:numel(imdsTrain.Files)
    img = readimage(imdsTrain, i);
    img = imresize(img, hogSize);
    feat = extractHOGFeatures(rgb2gray(img));
    XTrain = [XTrain; feat]; %#ok<AGROW>
end
YTrain = imdsTrain.Labels;

% --- HOG TEST (imdsTest) => wspólny test ---
XTest = [];
for i = 1:numel(imdsTest.Files)
    img = readimage(imdsTest, i);
    img = imresize(img, hogSize);
    feat = extractHOGFeatures(rgb2gray(img));
    XTest = [XTest; feat]; %#ok<AGROW>
end
YTest = yTrue;

t = templateSVM('KernelFunction','rbf','Standardize',true);
Mdl = fitcecoc(XTrain, YTrain, 'Learners', t, 'Coding','onevsall');

CV = crossval(Mdl, 'KFold', K_FOLDS_SVM);
cvLoss = kfoldLoss(CV);
cvAcc  = 1 - cvLoss;

fprintf('[5/7] SVM CV Accuracy=%.2f%% | CV Loss=%.4f\n', cvAcc*100, cvLoss);

yPredSVM = predict(Mdl, XTest);
accSVM = mean(yPredSVM == YTest);
f1SVM_macro = f1_macro(YTest, yPredSVM);

% Confusion Matrix SVM
fig = figure('Units','pixels','Position',[50 50 2400 1500]);
cm = confusionchart(YTest, yPredSVM);
cm.Title = sprintf('Macierz pomyłek - SVM (HOG) | %d-fold CV Acc=%.2f%%', K_FOLDS_SVM, cvAcc*100);
cm.FontSize = BIG_FONTSIZE;
cm.XLabel = 'Predykcja';
cm.YLabel = 'Klasa rzeczywista';
exportgraphics(fig, 'confmat_svm.png', 'Resolution', EXPORT_DPI);
close(fig);

fprintf('[5/7] SVM Test Acc=%.2f%% | Macro-F1=%.3f\n', accSVM*100, f1SVM_macro);

%% 6. ResNet-50 (transfer learning) + test WSPÓLNY
fprintf('[6/7] Transfer learning ResNet-50...\n');

try
    baseNet = resnet50;
catch
    error('Zainstaluj: Deep Learning Toolbox Model for ResNet-50 Network.');
end

inputSize = baseNet.Layers(1).InputSize;
augTrainRes = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', augmenter);
augValRes   = augmentedImageDatastore(inputSize, imdsVal);
augTestRes  = augmentedImageDatastore(inputSize, imdsTest);

lgraph = layerGraph(baseNet);
[learnableLayer, classLayer] = findLayersToReplaceLocal(lgraph);

newLearnable = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);

newClass = classificationLayer('Name','new_classoutput');

lgraph = replaceLayer(lgraph, learnableLayer.Name, newLearnable);
lgraph = replaceLayer(lgraph, classLayer.Name, newClass);

optionsRes = trainingOptions('adam', ...
    'MiniBatchSize',64, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augValRes, ...
    'ValidationFrequency',30, ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ExecutionEnvironment','gpu');

netRes = trainNetwork(augTrainRes, lgraph, optionsRes);

yPredRes = classify(netRes, augTestRes);
accRes = mean(yPredRes == yTrue);
f1Res_macro = f1_macro(yTrue, yPredRes);

% Confusion Matrix ResNet
fig = figure('Units','pixels','Position',[50 50 2400 1500]);
cm = confusionchart(yTrue, yPredRes);
cm.Title = 'Macierz pomyłek - ResNet-50';
cm.FontSize = BIG_FONTSIZE;
cm.XLabel = 'Predykcja';
cm.YLabel = 'Klasa rzeczywista';
exportgraphics(fig, 'confmat_resnet.png', 'Resolution', EXPORT_DPI);
close(fig);

fprintf('[6/7] ResNet Acc=%.2f%% | Macro-F1=%.3f\n', accRes*100, f1Res_macro);

%% 7. Wyniki per klasa (recall i F1 dla SHORT/itp.) + CSV + wykresy zbiorcze
fprintf('[7/7] Wyniki per-klasa + CSV + wykresy...\n');

% --- PER KLASA: recall i F1 ---
[recCNN, f1CNN] = per_class_metrics(yTrue, yPredCNN, classes);
[recSVM, f1SVM] = per_class_metrics(yTrue, yPredSVM, classes);
[recRES, f1RES] = per_class_metrics(yTrue, yPredRes, classes);

% Wypis w terminalu (format jak chciałeś)
fprintf('\n======= SKUTECZNOŚĆ PER KLASA (RECALL) [1.00=100%%] =======\n');
for i = 1:numClasses
    fprintf('%s: SVM=%.2f | CNN=%.2f | ResNet=%.2f\n', classes{i}, recSVM(i), recCNN(i), recRES(i));
end
fprintf('=========================================================\n');

fprintf('\n=========== F1 PER KLASA [1.00=100%%] ====================\n');
for i = 1:numClasses
    fprintf('%s: SVM=%.2f | CNN=%.2f | ResNet=%.2f\n', classes{i}, f1SVM(i), f1CNN(i), f1RES(i));
end
fprintf('=========================================================\n');

% Zapis CSV per klasa
T = table(string(classes), recSVM, recCNN, recRES, f1SVM, f1CNN, f1RES, ...
    'VariableNames', {'Class','Recall_SVM','Recall_CNN','Recall_ResNet','F1_SVM','F1_CNN','F1_ResNet'});
writetable(T, 'per_class_metrics.csv');

% Zapis zbiorczych accuracy/F1 macro do CSV
accuracyVals = [accCNN, accSVM, accRes]*100;
macroF1Vals  = [f1CNN_macro, f1SVM_macro, f1Res_macro];
writematrix(accuracyVals, 'accuracy_porownanie.csv');
writematrix(macroF1Vals,  'f1_macro_porownanie.csv');

% Wykres porównania accuracy (PNG HD)
fig = figure('Units','pixels','Position',[50 50 2400 1500]);
bar(accuracyVals);
set(gca,'XTickLabel',{'CNN','SVM','ResNet50'}, 'FontSize', BIG_FONTSIZE);
ylabel('Dokładność (%)','FontSize',LABEL_FONTSIZE);
title('Porównanie dokładności (wspólny test)','FontSize',TITLE_FONTSIZE);
grid on;
for i=1:numel(accuracyVals)
    text(i, accuracyVals(i)+0.5, sprintf('%.2f%%', accuracyVals(i)), ...
        'HorizontalAlignment','center', 'FontSize', BIG_FONTSIZE, 'FontWeight','bold');
end
exportgraphics(fig, 'bar_accuracies.png', 'Resolution', EXPORT_DPI);
close(fig);

% Wykres porównania macro-F1 (PNG HD)
fig = figure('Units','pixels','Position',[50 50 2400 1500]);
bar(macroF1Vals);
set(gca,'XTickLabel',{'CNN','SVM','ResNet50'}, 'FontSize', BIG_FONTSIZE);
ylabel('Macro F1-score','FontSize',LABEL_FONTSIZE);
title('Porównanie Macro F1-score (wspólny test)','FontSize',TITLE_FONTSIZE);
grid on;
for i=1:numel(macroF1Vals)
    text(i, macroF1Vals(i)+0.02, sprintf('%.3f', macroF1Vals(i)), ...
        'HorizontalAlignment','center', 'FontSize', BIG_FONTSIZE, 'FontWeight','bold');
end
exportgraphics(fig, 'bar_f1_score.png', 'Resolution', EXPORT_DPI);
close(fig);

% Podsumowanie
fprintf('\n================ PODSUMOWANIE (WSPÓLNY TEST) ================\n');
fprintf('CNN        | Acc: %.2f%% | Macro-F1: %.3f\n', accCNN*100, f1CNN_macro);
fprintf('SVM (HOG)  | CV Acc(%d-fold): %.2f%% | Test Acc: %.2f%% | Macro-F1: %.3f\n', ...
    K_FOLDS_SVM, cvAcc*100, accSVM*100, f1SVM_macro);
fprintf('ResNet-50  | Acc: %.2f%% | Macro-F1: %.3f\n', accRes*100, f1Res_macro);
fprintf('Per-klasa zapisano do: per_class_metrics.csv\n');
fprintf('=============================================================\n');

%% ================== FUNKCJE ==================

function f1 = f1_macro(trueLabels, predLabels)
    classes = unique(trueLabels);
    f1s = zeros(numel(classes),1);
    for i = 1:numel(classes)
        c = classes(i);
        tp = sum((predLabels == c) & (trueLabels == c));
        fp = sum((predLabels == c) & (trueLabels ~= c));
        fn = sum((predLabels ~= c) & (trueLabels == c));
        precision = tp / (tp + fp + eps);
        recall    = tp / (tp + fn + eps);
        f1s(i) = 2 * (precision * recall) / (precision + recall + eps);
    end
    f1 = mean(f1s);
end

function [recalls, f1s] = per_class_metrics(trueLabels, predLabels, classes)
% recalls(i) = TP/(TP+FN) dla klasy i
% f1s(i)     = F1-score dla klasy i
    n = numel(classes);
    recalls = zeros(n,1);
    f1s = zeros(n,1);
    for i = 1:n
        c = categorical(classes(i));
        tp = sum((predLabels == c) & (trueLabels == c));
        fp = sum((predLabels == c) & (trueLabels ~= c));
        fn = sum((predLabels ~= c) & (trueLabels == c));
        precision = tp / (tp + fp + eps);
        recall    = tp / (tp + fn + eps);
        recalls(i) = recall;
        f1s(i) = 2 * (precision * recall) / (precision + recall + eps);
    end
end

function stop = saveTrainingPlotCNN(info, prefix, dpi, fontSize)
    stop = false;
    if isempty(info.TrainingAccuracy); return; end

    fig = findall(0,'Type','Figure','Tag','NNET_CNN_TRAININGPLOT_UIFIGURE');
    if isempty(fig); return; end

    ax = findall(fig, 'Type', 'Axes');
    if numel(ax) < 2; return; end

    areas = arrayfun(@(a) a.Position(3)*a.Position(4), ax);
    [~, idx] = sort(areas, 'descend');
    ax1 = ax(idx(1));
    ax2 = ax(idx(2));

    try
        ax1.FontSize = fontSize;
        ax2.FontSize = fontSize;
    catch
    end

    f1name = [prefix '_training_plot_1.png'];
    f2name = [prefix '_training_plot_2.png'];

    exportgraphics(ax1, f1name, 'Resolution', dpi);
    exportgraphics(ax2, f2name, 'Resolution', dpi);

    try
        yl1 = lower(string(ax1.YLabel.String));
        yl2 = lower(string(ax2.YLabel.String));

        if contains(yl1,'acc');  movefile(f1name, [prefix '_training_plot_acc.png'], 'f'); end
        if contains(yl1,'loss'); movefile(f1name, [prefix '_training_plot_loss.png'], 'f'); end

        if contains(yl2,'acc');  movefile(f2name, [prefix '_training_plot_acc.png'], 'f'); end
        if contains(yl2,'loss'); movefile(f2name, [prefix '_training_plot_loss.png'], 'f'); end
    catch
    end
end

function [learnableLayer, classLayer] = findLayersToReplaceLocal(lgraph)
    layers = lgraph.Layers;
    classLayerIdx = find(arrayfun(@(l) isa(l,'nnet.cnn.layer.ClassificationOutputLayer'), layers), 1, 'last');
    if isempty(classLayerIdx)
        error('Nie znaleziono warstwy klasyfikacyjnej w sieci.');
    end
    classLayer = layers(classLayerIdx);

    for i = classLayerIdx-1:-1:1
        if isa(layers(i),'nnet.cnn.layer.FullyConnectedLayer') || isa(layers(i),'nnet.cnn.layer.Convolution2DLayer')
            learnableLayer = layers(i);
            return;
        end
    end
    error('Nie znaleziono warstwy uczącej się przed warstwą klasyfikacji.');
end
