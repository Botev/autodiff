function [err, grad] = evalBackprop(x,Wh1,W1,Wh2,W2,gradOn)
% Standard backprop
T = 20;
sigm = @(x) 1 ./ (1 + exp(-x));
filterH = zeros(size(Wh1,1),T);
filterPhi = zeros(size(W1,1),T);
solipH = zeros(size(Wh2,1),T-1);
solipPhi = zeros(size(W2,1),T-1);
filterH(:,1) = sigm(Wh1 * [x; 1]);
filterPhi(:,1) = sigm(W1 * [filterH(:,1); 1]);
for i=2:T
    solipH(:,i-1) = sigm(Wh2 * [filterPhi(:,i-1);1]);
    solipPhi(:,i-1) = sigm(W2 * [solipH(:,i-1);1]);
    filterH(:,i) = sigm(Wh1 * [filterPhi(:,i-1); 1]);
    filterPhi(:,i) = sigm(W1 * [filterH(:,i); 1]);
end
err = sum(sum(filterPhi.^2))/2;
err = err + sum(sum((solipPhi-filterPhi(:,2:end)).^2))/2;
if(gradOn)
    solipPhiEr = (solipPhi-filterPhi(:,2:end)).*solipPhi.*(1-solipPhi);
    filterPhiEr = (2*filterPhi-[filterPhi(:,1) solipPhi]);
    solipHEr = (W2(:,1:end-1)'*solipPhiEr) .* solipH .* (1-solipH);
    filterPhiEr(:,1:end-1) = (filterPhiEr(:,1:end-1) + Wh2(:,1:end-1)'*solipHEr);
    filterPhiEr(:,end) = filterPhiEr(:,end) .* filterPhi(:,end) .* (1-filterPhi(:,end));
    filterHEr = zeros(size(filterH));
    for i=T:-1:1
        filterHEr(:,i) = (W1(:,1:end-1)'*filterPhiEr(:,i)) .* filterH(:,i) .* (1-filterH(:,i));
        if(i==1)
            break;
        end
        filterPhiEr(:,i-1) = (filterPhiEr(:,i-1) + Wh1(:,1:end-1)'*filterHEr(:,i)) .* filterPhi(:,i-1) .* (1-filterPhi(:,i-1));
    end
    dWh1 = filterHEr * [x filterPhi(:,1:end-1); ones(1,T)]';
    dW1 = filterPhiEr * [filterH; ones(1,T)]';
    dWh2 = solipHEr * [filterPhi(:,1:end-1); ones(1,T-1)]';
    dW2 = solipPhiEr * [solipH; ones(1,T-1)]';
    grad = [reshape(dWh1,[numel(dWh1),1]);reshape(dW1,[numel(dW1),1]);reshape(dWh2,[numel(dWh2),1]);reshape(dW2,[numel(dW2),1])];
else
    grad = 0;
end
end