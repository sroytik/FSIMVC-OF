clear;
clc;

addpath(genpath('./'));

datadir='./datasets/';
dataname = {'MNIST_10000n_Xs'};

numdata = length(dataname); % number of the test datasets
numname = {'_Per0.1','_Per0.3','_Per0.5','_Per0.7','_Per0.9'};

for iDataIdx = 1:length(dataname)
    nSplits = length(numname);
    nRepeat_km = 1;    
    nMeasures = 10;
    orders = 6;    %\bar{t}
    ks = 5;
    mus = 0.5;
    nfolds = 10;
    datafile = [datadir, cell2mat(dataname(iDataIdx)), cell2mat(numname(1)), '.mat'];
    load(datafile);
    nCluster = length(unique(truelabel));
    nView = length(data);
    ms= nCluster * [2:2:8]; % anchor size
    clear data truelabel;
    
    paramCell = Ours_build_param(orders, ks, mus, ms);
    nParam = length(paramCell);
    
    Ours_ResAIO = zeros(nSplits, nfolds, nParam, nRepeat_km, nMeasures);
    Ours_ts = zeros(nSplits, nfolds, nParam);
    
    for iSplit = 1:1:nSplits
        datafile = [datadir, cell2mat(dataname(iDataIdx)), cell2mat(numname(iSplit)), '.mat'];
        load(datafile);
        for ifold = 1:nfolds
            missInd = folds{ifold};
            nView = length(data);
            gt = truelabel;
            nSmp = length(gt);
            nCluster = length(unique(gt));
            tic;
            Xs = cell(1, nView);
            for iView = 1:nView
                Xi = data{iView};
                Xs{iView} = Xi(:, missInd(:,iView) > 0);
                Xs{iView} = Xs{iView}';
                Xs{iView} = double(zscore(double(Xs{iView})));
            end
            t1 = toc;
            
            for iParam = 1 : nParam
                param = paramCell{iParam};
                disp(['iDataIdx=', num2str(iDataIdx), '    ', char(dataname(iDataIdx)), '    index=', num2str(iSplit), '    ', char(numname(iSplit)), '    fold=', num2str(ifold),  '    iParam=', num2str(iParam)])
                t2_s = tic;
                Bs = cell(1, nView);
                nAnchor = param.m;
                for iView = 1:nView
                    [~, Xa] = litekmeans(Xs{iView}, param.m, 'Replicates', 1);   
                    Bs{iView} = ConstructBP_pkn(Xs{iView}, Xa, 'nNeighbor', param.k);
                end
                [H_normalized, alpha, Ws, C, Z, objHistory, Hs] = Ours_v3(Bs, missInd, nCluster, param.nOrder * ones(1, nView), nCluster, nCluster, param.mu);
                t2 = toc(t2_s);
               
                stream = RandStream.getGlobalStream;
                reset(stream);
                t3_s = tic;
                res_km = zeros(nRepeat_km, nMeasures);
                for iRepeat = 1 : nRepeat_km
                    [label, center] = kmeans(H_normalized, nCluster, 'MaxIter', 1000, 'Replicates', 10, 'emptyaction', 'singleton');
                    r = my_eval_y(gt, label);
                    res_km(iRepeat, :) = r;
                    Ours_ResAIO(iSplit, ifold, iParam, iRepeat, :) = r;
                end
                t3 = toc(t3_s);
                Ours_ts(iSplit, ifold, iParam) = t1 + t2 + t3/nRepeat_km;
            end
        end
    end
            
    a=mean(Ours_ResAIO,4);
    [per_num,fold_num,para_num,rep_km,measure_num] = size(Ours_ResAIO);
    
    final_results = zeros(per_num,measure_num);     %rows --- missing ratios (0.1:0.2:0.9) , columns --- measures (ACC NMI PUR AR RI MI HI Fscore precision recall)
    final_std = zeros(per_num,measure_num);         %rows --- missing ratios (0.1:0.2:0.9) , columns --- measures (ACC NMI PUR AR RI MI HI Fscore precision recall)
    for iper=1:per_num
        all_results_singleper = zeros(fold_num,para_num,measure_num);  
        for imeasure = 1:measure_num
            for ifold=1:fold_num 
                all_results_singleper(ifold,:,imeasure)= reshape(a(iper,ifold,:,1,imeasure),1,4)';
            end
            [bst_result,indx] = max(mean(all_results_singleper(:,:,imeasure)));
            final_results(iper,imeasure) = bst_result;
            b=all_results_singleper(:,:,imeasure);
            final_std(iper,imeasure) = std(b(:,indx));
        end
    end
    
    average_time = mean(mean(mean(Ours_ts)));
    
    disp("10 average clustering evaluation metrics at different missing ratios:");  
    disp(final_results);      
    
    disp("std of 10 clustering evaluation metrics at different missing ratios:");  
    disp(final_std);      
    
    disp("average running time:");  
    disp(average_time);   
end
