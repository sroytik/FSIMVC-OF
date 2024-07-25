function Z = ConstructBP_pkn(X, anchors, varargin)
% Input
%         X: nSmp * nFea
%         anchors: nAnchor * nFea
%         nNeighbor: row sparsity
% Output
%         Z: nSmp * nAnchor
%
% [1] Clustering and Projected Clustering with Adaptive Neighbors, KDD,
% 2014
%

[nSmp, nFea] = size(X);
[nAnchor, nFea] = size(anchors);

param_names = {'nNeighbor'};
param_default =  {5};
[eid, errmsg, nNeighbor] = getargs(param_names, param_default, varargin{:});
if ~isempty(eid)
    error(sprintf('ConstructBP_pkn:%s', eid), errmsg);
end


D = EuDist2(X, anchors, 0); % O(nmd)
D (D<0) = 0;
[D2, Idx] = sort(D, 2); % sort each row
v1 = D2(:, nNeighbor+1);
v2 = D2(:, 1:nNeighbor);
v3 = bsxfun(@minus, v1, v2);
v4 = 1./max(nNeighbor * v1 - sum(v2, 2), eps);
v5 = bsxfun(@times, v3, v4);
row_idx = repmat((1:nSmp)', nNeighbor, 1);
idx_k = Idx(:, 1:nNeighbor);
Z = sparse(row_idx, idx_k(:), v5(:), nSmp, nAnchor, nSmp * nNeighbor);
end