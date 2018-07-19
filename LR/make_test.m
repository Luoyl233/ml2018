function rlt = make_test(X, beta)
    m = size(X, 1);
    rlt = zeros(m);
    rescaling = [504./4431., 4643./292., 4910./25., 4851./84., 4831./103.];
    rescaling = rescaling / 5.0;
    rescaling(1) = rescaling(1) * 25.;
    %rescaling(2) = rescaling(2);
    %rescaling(5) = rescaling(5) * 2.0;

    
    for i=1 : m
        max_prob = 0;
        max_label = 0;
        for j=1 : 5
            beta_T_x = beta(:,j)'* X(i,:)';
            cur_prob = exp(beta_T_x) / (1.);
            %使用再缩放
            %cur_prob =  exp(rescaling(j)*beta_T_x );
            if(cur_prob > max_prob)
                max_prob = cur_prob;
                max_label = j;
            end 
        end
        rlt(i) = max_label;
    end
end