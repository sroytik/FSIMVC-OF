function paramCell = Ours_build_param(orders, ks, mus, ms)
if ~exist('orders', 'var')
    orders = 5;
end

if ~exist('ks', 'var')
    ks = 5;
end

if ~exist('mus', 'var')
    mus = 0.5;
end


if ~exist('ms', 'var')
    ms = 1000;
end

nParam = length(orders) * length(ks) * length(mus) * length(ms);
paramCell = cell(nParam, 1);
idx = 0;
for i1 = 1:length(orders)
    for i2 = 1:length(ks)
        for i3 = 1:length(mus)
            for i4 = 1:length(ms)
                param = [];
                param.nOrder = orders(i1);
                param.k = ks(i2);
                param.mu = mus(i3);
                param.m = ms(i4);
                idx = idx + 1;
                paramCell{idx,1} = param;
            end
        end
    end
end