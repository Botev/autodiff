function [err,grad] = adV1(xv,Wh1,W1,Wh2,W2,gradOn)
T = 20;
x = ADNode(0,xv);
Wh1 = ADNode(1,Wh1);
W1 = ADNode(1,W1);
Wh2 = ADNode(1,Wh2);
W2 = ADNode(1,W2);
filterH = sigmoid(Wh1 * [x; 1]);
filterPhi = sigmoid(W1 * [filterH; 1]);
err = sum(filterPhi.^2)/2;
for i=2:T
    solipH = sigmoid(Wh2 * [filterPhi;1]);
    solipPhi = sigmoid(W2 * [solipH;1]);
    filterH = sigmoid(Wh1 * [filterPhi; 1]);
    filterPhi = sigmoid(W1 * [filterH; 1]);
    err = err + sum(filterPhi.^2)/2;
    err = err + sum((solipPhi - filterPhi).^2)/2;
end
if(gradOn)
    calculateGradient(err,1);
    grad = [reshape(Wh1.grad,[numel(Wh1.grad),1]);reshape(W1.grad,[numel(W1.grad),1]);
        reshape(Wh2.grad,[numel(Wh2.grad),1]);reshape(W2.grad,[numel(W2.grad),1])];
else
    grad = 0;
end
err = err.value;
end

