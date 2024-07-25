function [H_normalized, alpha, Ws, C, Z, objHistory, Hs] = Ours_v3(Bs, missInd, nCluster, nOrders, nProjection, nAnchor, mu)

nView = length(Bs);
nSmp = size(missInd,1);
nFeas = cell2mat(cellfun(@(x) size(x, 2), Bs, 'UniformOutput',false));

AXss = cell(1, nView);
Ms = cell(1, nView);
Betas = cell(1, nView);
for iView = 1:nView
    %*******************************************
    % Normalize Bi => B D^-.5
    %*******************************************
    Bi = Bs{iView};
    tmp = mean(sum(Bi, 2));
    if (tmp < 1.01) && (tmp > 0.99)
        Bi = bsxfun(@times, Bi, 1./sqrt(max(sum(Bi, 1), eps))); %  Bi => B D^-.5
    end
    Bs{iView} = Bi;
    %*******************************************
    % Precompute B L^t
    %*******************************************
    nOrder = nOrders(iView);
    AXs = cell(nOrder+1, 1);
    if nOrder == 0
        AXs{1} = Bi;
    elseif nOrder == 1
        AXs{1} = Bi; % 0-order
        BB = Bi' * Bi;     %S

        Lm = (1-mu) * eye(size(Bi, 2)) + mu * (BB);       
        AXs{2} = Bi * Lm; % 1-order
        
    elseif nOrder >= 2
        AXs{1} = Bi; % 0-order   ni*mi
        BB = Bi' * Bi;     %nimi^2
        Lm = (1-mu) * eye(size(Bi, 2)) + mu * (BB);    
        AXs{2} = Bi * Lm; % 1-order  
        for iOrder = 2:nOrder
            AXs{iOrder + 1} = AXs{iOrder} * Lm;
        end
    end
    
    for iOrder = 1:length(AXs)
        AXs{iOrder} = AXs{iOrder}/iOrder;
    end
    AXss{iView} = AXs;
    
    %*******************************************
    % Precompute Mi
    %*******************************************
    Mi = zeros(nOrder + 1);
    for iOrder_1 = 1:size(Mi, 1)
        for iOrder_2 = iOrder_1:size(Mi, 1)
            Mi(iOrder_1, iOrder_2) = sum(sum(AXs{iOrder_1} .* AXs{iOrder_2}));
            Mi(iOrder_2, iOrder_1) = Mi(iOrder_1, iOrder_2);
        end
    end
    Ms{iView} = Mi ;
    
    %*******************************************
    % Init Beta
    %*******************************************
    beta = ones(nOrder + 1, 1)/(nOrder + 1);
    Betas{iView} = beta;
end

missInd2 = mat2cell(missInd>0, nSmp, ones(1, nView)); % logical
%**********************************************
% Initialize
%**********************************************
Ws = cell(nView,1);  % di * d
for iView = 1:nView
    Ws{iView} = zeros(nFeas(iView), nProjection);
end
C = zeros(nProjection, nAnchor);  %  c * c
Z = zeros(nAnchor, nSmp); % c  * n
Z(:, 1:nAnchor) = eye(nAnchor);
alpha = ones(1, nView)/nView;

converges = false;
iter = 0;
maxIter = 50;
Hs = cell(1, maxIter);
objHistory = [];
options = optimoptions('quadprog','Display','off');
while ~converges
    iter = iter + 1;
    
    %*******************************************
    % Optimize W_i
    %*******************************************
    ZC = Z' * C'; % nSmp * nProjection
    AXBetas = cell(1, nView);
    for iView=1:nView
        beta = Betas{iView};
        AXs = AXss{iView};
        AXbeta = zeros(size(Bs{iView}));
        for iOrder = 1:length(AXs)
            AXbeta = AXbeta + beta(iOrder) * AXs{iOrder};
        end
        AXBetas{iView} = AXbeta;
        AXZC = AXbeta' * ZC(missInd2{iView}, :);
        [U1, ~, V1] = svd(AXZC, 'econ');
        Ws{iView} = U1 * V1';
    end
    
    %*******************************************
    % Optimize C
    %*******************************************
    WXZ = zeros(nProjection, nAnchor);
    for iView = 1:nView
        WXZ = WXZ + alpha(iView)^2 * Ws{iView}' * AXBetas{iView}' * Z(:, missInd2{iView})';
    end
    [U2, ~, V2] = svd(WXZ, 'econ');
    C = U2 * V2';
    
    %*******************************************
    % Optimize Z
    %*******************************************
    C1 = sum(bsxfun(@times, missInd, alpha.^2), 2);
    C2 = zeros(nSmp, nAnchor);
    for iView = 1:nView
        XWC = alpha(iView)^2 * AXBetas{iView} * Ws{iView} * C;  %ni*mi  mi*c c*c
        C2(missInd2{iView}, :) = C2(missInd2{iView}, :) +  XWC;     %%%%%
    end
    F = bsxfun(@rdivide, C2, max(C1, eps));
    for iSmp = 1:nSmp
        Z(:, iSmp) = EProjSimplex_new(F(iSmp, :))';
    end
    
    %*******************************************
    % Optimize beta
    %*******************************************
    WCZs = cell(1, nView);
    for iView = 1:nView
        WCZs{iView} = Ws{iView} * C * Z(:, missInd2{iView});
    end
    Betas = cell(1, nView);
    for iView = 1:nView
        Mi  = Ms{iView};
        AXs = AXss{iView};
        f = zeros(nOrders(iView)+1, 1);
        for iOrder = 1:nOrders(iView)+1
            f(iOrder) = sum(sum(AXs{iOrder} .* WCZs{iView}'));
        end
        beta = quadprog(Mi, -f, [], [], ones(1, nOrders(iView)+1), 1, zeros(nOrders(iView)+1, 1), ones(nOrders(iView)+1, 1), [], options);
        Betas{iView} = beta;
    end
    
    %*******************************************
    % Optimize alpha
    %*******************************************
    es = zeros(1, nView);
    for iView = 1:nView
        Ei = AXBetas{iView} - WCZs{iView}';
        es(iView) = sum(sum(Ei.^2));
    end
    alpha = 1./sqrt(es) / sum(1./sqrt(es));
    
    %*******************************************
    % Compute obj
    %*******************************************
    obj = sum(sum(alpha.^2 .* es));
    objHistory = [objHistory; obj]; %#ok
    
    if (iter>1) && (abs((objHistory(iter-1)-objHistory(iter))/(objHistory(iter-1)))<1e-5 || iter>maxIter || objHistory(iter) < 1e-10)
        converges = true;
    end
    
    if iter > maxIter
        converges = true;
    end
end
[H, ~, ~] = svd(Z', 'econ');
H = H(:,1:nCluster);
H_normalized = bsxfun(@rdivide, H, max(sum(H.^2, 2), eps));
end