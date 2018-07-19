

path = '..\assign2_dataset\';
train_feature = importdata([path, 'page_blocks_train_feature.txt']);
train_label = importdata([path, 'page_blocks_train_label.txt']);
test_feature = importdata([path, 'page_blocks_test_feature.txt']);
test_label = importdata([path, 'page_blocks_test_label.txt']);

n_train = size(train_feature, 1);
n_test = size(test_feature, 1);
train_feature = [train_feature ones(n_train ,1)];
test_feature = [test_feature ones(n_test, 1)];


beta = zeros(11, 5);
for i=1 : 5
    fprintf('make smote %d...\n', i);
    [X, Y] = make_smote(train_feature, train_label, i, 5, 5);
    fprintf('make train %d...\n', i);
    beta(:,i) = make_train(X, Y);
    %disp(beta(:,i));
end
disp(beta);

%test
fprintf('make test...\n');
test_rlt = make_test(test_feature, beta);

%compare rlt
n_correct = 0;
TP = zeros(5);
FP = zeros(5);
FN = zeros(5);
%tot = zeros(5);
for i=1 : n_test
    %tot(test_label(i)) = tot(test_label(i)) + 1;
    if(test_rlt(i) == test_label(i))
        n_correct = n_correct + 1;
        TP(test_rlt(i)) = TP(test_rlt(i)) + 1;
    else
        FP(test_rlt(i)) = FP(test_rlt(i)) + 1;
        FN(test_label(i)) = FN(test_label(i)) + 1;
    end
end
for i=1 : 5
    if(TP(i) == 0)
        P = 0; R = 0;
    else
        P = TP(i)/(TP(i)+FP(i));
        R = TP(i)/(TP(i)+FN(i));
    end
   fprintf('class %d:查准率=%d/%d=%f,  查全率=%d/%d=%f\n', i, TP(i), TP(i)+FP(i), P,TP(i),TP(i)+FN(i), R);
end
fprintf('准确率=%d/%d=%f\n', n_correct, n_test ,n_correct/n_test);


