function beta = make_train(X, Y)
    [m, d] = size(X);
    
    % ��������Ϊ0/1
%     index = (Y ~= label);
%     Y(index) = 0;
%     Y(~index) = 1;

    % betaΪ����Ĳ���
    beta = zeros(d, 1);
    beta(d, 1) = 1;

    % LΪ������Ȼ
    L = 0; last_L = 0;
    
    turn = 0; e_max = 700.0;
    % ţ�ٵ��������
    while(1)
        turn = turn + 1;
        % ����beta^T * X�Ľ���������ظ�����
        beta_T_x = beta' * X';
        % ����L(beta)
        L = 0;
        for i=1 : m
            % e^1024�����
            if(beta_T_x(:,i) < e_max)
                L = L - Y(i,:)*beta_T_x(:,i) + log(1 + exp(beta_T_x(:,i)));
            else
                %fprintf('%d > e_max:1\n', beta_T_x(:,i));
                L = L - Y(i,:)*beta_T_x(:,i) + log(1 + exp(e_max));
            end
        end
    
        % ��ֹ����
        margin = abs(L - last_L);
        % disp(abs(margin));
        if((margin <= 1e-4) || (turn > 100))
            break; 
        end
    
        % �����������, d_1, d_2�ֱ�Ϊһ�����׵���
        last_L = L; d_1 = 0.; d_2 = 0.;
        for i=1 : m
            % ����p1
            if(beta_T_x(:,i) > 128.)
                %fprintf('%d > e_max:2\n', beta_T_x(:,i));
                p1 = 1.- (1. / (1 + exp(e_max)));
            else
                p1 = 1.- (1. / (1 + exp(beta_T_x(:,i))));
            end   
            d_1 = d_1 - X(i,:)'*(Y(i,:)-p1);
            d_2 = d_2 + X(i,:)'*X(i,:)*p1*(1-p1);
        end
        if(det(d_2) <= 1e-2)
            %fprintf('use pinv\n');
            beta = beta - pinv(d_2) * d_1;
        else
            beta = beta - d_2\d_1;
        end
    end
end