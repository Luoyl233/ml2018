function [smote_feature, smote_label] = make_smote(X, Y,label, k, rate)
    [m, d] = size(X);
    index_positive = find(Y == label);
    index_negative = find(Y ~= label);
    Y(index_positive) = 1;
    Y(index_negative) = 0;
    n_positive = length(index_positive);
    n_negative = m - n_positive;
    ratio = n_negative / n_positive;
    
    if(ratio > 1.)
        index_target = index_positive;
        %rate = floor(ratio-1);
        rate = min(floor(ratio-1), rate);
        n_new_sample = n_positive * rate;
        smote_label = [Y; ones(n_new_sample, 1)];
        %label = 1;
    else
        index_target = index_negative;
        %rate = floor(1/ratio - 1);
        rate = min(floor(1/ratio - 1), rate);
        n_new_sample = n_negative * rate;
        smote_label = [Y; zeros(n_new_sample, 1)];
        %label = 0;
    end
    n_target = size(index_target, 1);
    new_sample = zeros(n_new_sample, d);
    
    dist_X = zeros(m, n_target);
    for i=1 : n_target
        dist_X(:, i) = dist(X, X(index_target(i), :)');
    end
    
    for i=1 : n_target
        index_k_nearest = knn(i, k, dist_X);
        for j=1 : rate
            nn = randi(k);
            diff = X(index_k_nearest(nn),:) - X(index_target(i),:);
            new_sample((i-1)*rate+j,:) = X(index_target(i),:) +  rand(1)* diff;
        end
    end
    
    smote_feature = [X ;new_sample];
    
    function index_k_nearest = knn(sample_index, k, dist_matrix)
       [~, index] = sort(dist_matrix(:, sample_index));
       index_k_nearest = index(2:k+1, 1);
%        idx = 2;
%        index_k_nearest = zeros(k, 1);
%        while(k > 0)
%            if(Y((index(idx, :))) == label)
%                index_k_nearest(k, :) = index(idx, :);
%                k = k - 1;
%            end
%            idx = idx + 1;
%        end
     
    end
end