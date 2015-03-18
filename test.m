D = 10;
x = rand(D,20);
Wh1 = rand(D,D+1);
W1 = rand(D,D+1);
Wh2 = rand(D,D+1);
W2 = rand(D,D+1);
N = numel(Wh1) + numel(Wh2) + numel(W1) + numel(W2);
w1 = randn(N,1);
f = @(w,g) evalBackprop(x(:,1),reshape(w(1:D^2+D),[D,D+1]),reshape(w(1+D^2+D:2*D^2+2*D),[D,D+1]),...
reshape(w(1+2*D^2+2*D:3*D^2+3*D),[D,D+1]),reshape(w(1+3*D^2+3*D:4*D^2+4*D),[D,D+1]),g);
f2 = @(w,g) adV1(x(:,1),reshape(w(1:D^2+D),[D,D+1]),reshape(w(1+D^2+D:2*D^2+2*D),[D,D+1]),...
reshape(w(1+2*D^2+2*D:3*D^2+3*D),[D,D+1]),reshape(w(1+3*D^2+3*D:4*D^2+4*D),[D,D+1]),g);
N = 1000;
optT = zeros(1,1000);
for i=1:N
    t = tic;
    [ev,grad] = f(w1,1);
    optT(i) = toc(t);
end
fprintf('%.3f,%.3f\n',ev,norm(grad));
adVT = zeros(1,1000);
for i=1:N
    t = tic;
    [ev,grad] = f2(w1,1);
    adVT(i) = toc(t);
end
fprintf('%.3f,%.3f\n',ev,norm(grad));