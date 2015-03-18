classdef ADNode < handle
    % Represents a node in the compute graph
    properties
        % Reference to its parents
        parents
        % Children count
        childCount
        % Reference to the gradient functions with respect to each parent,
        % each takes 3 arguments - gradient of the node and value of
        % parents
        gf
        % Types of nodes values are 0 - constants, 1 - parameter, 2 - parameter
        % dependent
        valueType
        % Value stored on the forward pass of the algorithm
        value  
        % Stores whether the node is virtual or not
        virtual
        % Operator type to derive this node. Note that this is used for
        % optimisation purposes of memories in order to be able skip the
        % creation of lots of virtual nodes which can be bypassed
        % Example: a = b + c + d creates nodes a1 = b + c, a2 = a1 + d and
        % stores that in a. In fact if these are different more complicated
        % functions it is possible to remove the storage of a,a1 and a2
        % Types are in same order as defined here
        % 1 - plus
        % 2 - minus
        % 3 - uplus
        % 4 - uminus
        operatorType
        % Gradient value stored on the backward pass
        grad
        % Count of children messages 
        childMsg
    end
    methods(Static)
        function mask = extrapolateSubmatrix(mask,values)
            mask(mask == 1) = values;
        end
    end
    methods
        function obj = ADNode(valueType,value)
            obj.valueType = valueType;
            obj.value = value;
            obj.grad = 0;
            obj.parents = [];
            obj.gf = [];
            obj.childCount = 0;
            obj.childMsg = 0;
        end
        function s = size(a)
            s = size(a.value);
        end
        %% Algebric operators
        function obj = plus(a,b)
            if(isa(a,'ADNode') && isa(b,'ADNode'))
                if(a.valueType + b.valueType > 0)
                    obj = ADNode(2,a.value+b.value);
                else
                    obj = ADNode(0,a.value+b.value);
                end
                obj.parents{1} = a;
                obj.gf{1} = @(dv,v,p) dv;
                obj.parents{2} = b;
                obj.gf{2} = @(dv,v,p) dv;
                a.childCount = a.childCount + 1;
                b.childCount = b.childCount + 1;
            else
                if(isa(a,'ADNode'))
                    parent = a;
                    val = a.value + b;
                else
                    parent = b;
                    val = a + b.value;
                end
                if(parent.valueType > 0)
                    obj = ADNode(2,val);
                else
                    obj = ADNode(0,val);
                end
                obj.parents{1} = parent;
                obj.gf{1} = @(dv,v,p) dv;
                parent.childCount = parent.childCount + 1;
            end
        end
        function obj = minus(a,b)
            if(isa(a,'ADNode') && isa(b,'ADNode'))
                if(a.valueType + b.valueType > 0)
                    obj = ADNode(2,a.value-b.value);
                else
                    obj = ADNode(0,a.value-b.value);
                end
                obj.parents{1} = a;
                obj.gf{1} = @(dv,v,p) dv;
                obj.parents{2} = b;
                obj.gf{2} = @(dv,v,p) -dv;
                a.childCount = a.childCount + 1;
                b.childCount = b.childCount + 1;
            else
                if(isa(a,'ADNode'))
                    val = a.value - b;
                    if(a.valueType > 0)
                        obj = ADNode(2,val);
                    else
                        obj = ADNode(0,val);
                    end
                    obj.parents{1} = a;
                    obj.gf{1} = @(dv,v,p) dv;
                    a.childCount = a.childCount + 1;
                else
                    val = a - b.value;
                    if(b.valueType > 0)
                        obj = ADNode(2,val);
                    else
                        obj = ADNode(0,val);
                    end
                    obj.parents{1} = b;
                    obj.gf{1} = @(dv,v,p) dv;
                    b.childCount = b.childCount + 1;
                end
            end
        end
        function obj = uplus(a)
            if(a.valueType > 0)
                obj = ADNode(2,a.value);
            else
                obj = ADNode(0,a.value);
            end
            obj.parents{1} = a;
            obj.gf{1} = @(dv,v,p) dv;
            a.childCount = a.childCount + 1;
        end
        function obj = uminus(a)
            if(a.valueType > 0)
                obj = ADNode(2,a.value);
            else
                obj = ADNode(0,a.value);
            end
            obj.parents = a;
            obj.parents{1} = a;
            obj.gf{1} = @(dv,v,p) -dv;
            a.childCount = a.childCount + 1;
        end
        function obj = times(a,b)
            if(isa(a,'ADNode') && isa(b,'ADNode'))
                if(a.valueType + b.valueType > 0)
                    obj = ADNode(2,a.value.*b.value);
                else
                    obj = ADNode(0,a.value.*b.value);
                end
                obj.parents{1} = a;
                obj.gf{1} = @(dv,v,p) dv.*p{2}.value;
                obj.parents{2} = b;
                obj.gf{2} = @(dv,v,p) dv.*p{1}.value;
                a.childCount = a.childCount + 1;
                b.childCount = b.childCount + 1;
            else
                if(isa(a,'ADNode'))
                    val = a.value .* b;
                    if(a.valueType > 0)
                        obj = ADNode(2,val);
                    else
                        obj = ADNode(0,val);
                    end
                    obj.parents{1} = a;
                    obj.gf{1} = @(dv,v,p) dv.*b;
                    a.childCount = a.childCount + 1;
                else
                    val = a .* b.value;
                    if(b.valueType > 0)
                        obj = ADNode(2,val);
                    else
                        obj = ADNode(0,val);
                    end
                    obj.parents{1} = b;
                    obj.gf{1} = @(dv,v,p) dv.*a;
                    b.childCount = b.childCount + 1;
                end
            end
        end
        function obj = mtimes(a,b)
            if(isa(a,'ADNode') && isa(b,'ADNode'))
                if(a.valueType + b.valueType > 0)
                    obj = ADNode(2,a.value*b.value);
                else
                    obj = ADNode(0,a.value*b.value);
                end
                obj.parents{1} = a;
                obj.gf{1} = @(dv,v,p) dv * p{2}.value';
                obj.parents{2} = b;
                obj.gf{2} = @(dv,v,p) p{1}.value'*dv;
                a.childCount = a.childCount + 1;
                b.childCount = b.childCount + 1;
            else
               if(isa(a,'ADNode'))
                    val = a.value * b;
                    if(a.valueType > 0)
                        obj = ADNode(2,val);
                    else
                        obj = ADNode(0,val);
                    end
                    obj.parents{1} = a;
                    obj.gf{1} = @(dv,v,p) dv*b';
                    a.childCount = a.childCount + 1;
                else
                    val = a * b.value;
                    if(b.valueType > 0)
                        obj = ADNode(2,val);
                    else
                        obj = ADNode(0,val);
                    end
                    obj.parents{1} = b;
                    obj.gf{1} = @(dv,v,p) a'*dv;
                    b.childCount = b.childCount + 1;
                end
            end
        end
        function obj = rdivide(a,b)
            if(isa(a,'ADNode') && isa(b,'ADNode'))
                if(a.valueType + b.valueType > 0)
                    obj = ADNode(2,a.value./b.value);
                else
                    obj = ADNode(0,a.value./b.value);
                end
                obj.parents{1} = a;
                obj.gf{1} = @(dv,v,p) dv./p{2}.value;
                obj.parents{2} = b;
                obj.gf{2} = @(dv,v,p) -dv.*p{1}.value./p{2}.value.^2;
                a.childCount = a.childCount + 1;
                b.childCount = b.childCount + 1;
            else
               if(isa(a,'ADNode'))
                    val = a.value ./ b;
                    if(a.valueType > 0)
                        obj = ADNode(2,val);
                    else
                        obj = ADNode(0,val);
                    end
                    obj.parents{1} = a;
                    obj.gf{1} = @(dv,v,p) dv./b;
                    a.childCount = a.childCount + 1;
                else
                    val = a ./ b.value;
                    if(b.valueType > 0)
                        obj = ADNode(2,val);
                    else
                        obj = ADNode(0,val);
                    end
                    obj.parents{1} = b;
                    obj.gf{1} = @(dv,v,p) -dv.*a./p{1}.value.^2;
                    b.childCount = b.childCount + 1;
                end
            end           
        end
        function obj = ldivide(a,b)
            obj = rdivide(b,a);
        end
        function obj = mrdivide(a,b)
            if(isa(a,'ADNode') && isa(b,'ADNode'))
                if(all(size(b.value) == [1 1]))
                    val = a.value / b.value;
                    if(a.valueType + b.valueType > 0)
                        obj = ADNode(2,val);
                    else
                        obj = ADNode(0,val);
                    end
                    obj.parents{1} = a;
                    obj.gf{1} = @(dv,v,p) dv./p{2}.value;
                    obj.parents{2} = b;
                    obj.gf{2} = @(dv,v,p) sum(sum(-dv.*p{1}.value))/p{2}.value.^2;
                else
                    error('ADNode does not support matrix division');
                end
            elseif(isa(a,'ADNode'))
                val = a.value / b;
                if(a.valueType > 0)
                    obj = ADNode(2,val);
                else
                    obj = ADNode(0,val);
                end
                obj.parents{1} = a;
                obj.gf{1} = @(dv,v,p) dv./b;
            else
                error('ADNode does not support matrix division');
            end
        end
        function obj = power(a,b)
            if(isa(a,'ADNode') && isa(b,'ADNode'))
                if(a.valueType + b.valueType > 0)
                    obj = ADNode(2,a.value.^b.value);
                else
                    obj = ADNode(0,a.value.^b.value);
                end
                obj.parents{1} = a;
                obj.gf{1} = @(dv,v,p) dv.*p{2}.value.*v./p{1}.value;
                obj.parents{2} = b;
                obj.gf{2} = @(dv,v,p) dv.*log(p{1}.value).*v;
                a.childCount = a.childCount + 1;
                b.childCount = b.childCount + 1;
            else
               if(isa(a,'ADNode'))
                    val = a.value .^ b;
                    if(a.valueType > 0)
                        obj = ADNode(2,val);
                    else
                        obj = ADNode(0,val);
                    end
                    obj.parents{1} = a;
                    if(b == 2)
                        obj.gf{1} = @(dv,v,p) dv.*b.*p{1}.value;
                    else
                        obj.gf{1} = @(dv,v,p) dv.*b.*v./p{1}.value;
                    end
                    a.childCount = a.childCount + 1;
                else
                    val = a .^ b.value;
                    if(b.valueType > 0)
                        obj = ADNode(2,val);
                    else
                        obj = ADNode(0,val);
                    end
                    obj.parents{1} = b;
                    obj.gf{1} = @(dv,v,p) dv.*log(a).*v;
                    b.childCount = b.childCount + 1;
                end
            end
        end
        function obj = ctranspose(a)
            if(a.valueType > 0)
                obj = ADNode(2,a.value');
            else
                obj = ADNode(0,a.value');
            end
            obj.parents{1} = a;
            obj.gf{1} = @(dv,v,p) dv';
            a.childCount = a.childCount + 1;
        end
        function obj = transpose(a)
            if(a.valueType > 0)
                obj = ADNode(2,a.value.');
            else
                obj = ADNode(0,a.value.');
            end
            obj.parents{1} = a;
            obj.gf{1} = @(dv,v,p) dv.';
            a.childCount = a.childCount + 1;
        end
        function obj = sum(varargin)
            if(nargin == 1)
                a = varargin{1};
                if(a.valueType > 0)
                    obj = ADNode(2,sum(a.value));
                else
                    obj = ADNode(0,sum(a.value));
                end
                obj.parents{1} = a;
                if(size(a.value,1) > 1)
                    obj.gf{1} = @(dv,v,p) repmat(dv,[size(a.value,1),1]);
                else
                    obj.gf{1} = @(dv,v,p) repmat(dv,[1, size(a.value,2)]);
                end
                a.childCount = a.childCount + 1;
            else
                a = varargin{1};
                dim = varargin{2};
                if(dim == 0)
                    val = sum(sum(a.value));
                elseif(dim == 1)
                    val = sum(a.value,1);
                else
                    val = sum(a.value,2);
                end
                if(a.valueType > 0)
                    obj = ADNode(2,val);
                else
                    obj = ADNode(0,val);
                end
                obj.parents{1} = a;
                if(dim == 0)
                    obj.gf{1} = @(dv,v,p) repmat(dv,[size(a.value,1),size(a.value,2)]);
                elseif(dim == 1)
                    obj.gf{1} = @(dv,v,p) repmat(dv,[size(a.value,1),1]);
                else
                    obj.gf{1} = @(dv,v,p) repmat(dv,[1, size(a.value,2)]);
                end
                a.childCount = a.childCount + 1;
            end
        end
        %% Known elemntwise functions
        function obj = exp(a)
            if(a.valueType > 0)
                obj = ADNode(2,exp(a.value));
            else
                obj = ADNode(0,exp(a.value));
            end
            obj.parents{1} = a;
            obj.gf{1} = @(dv,v,p) dv.*v;
            a.childCount = a.childCount + 1;
        end
        function obj = log(a)
            if(a.valueType > 0)
                obj = ADNode(2,log(a.value));
            else
                obj = ADNode(0,log(a.value));
            end
            obj.parents{1} = a;
            obj.gf{1} = @(dv,v,p) dv./p{1}.value;
            a.childCount = a.childCount + 1;
        end
        function obj = sigmoid(a)
            if(a.valueType > 0)
                obj = ADNode(2,1./(1+exp(-a.value)));
            else
                obj = ADNode(0,1./(1+exp(-a.value)));
            end
            obj.parents{1} = a;
            obj.gf{1} = @(dv,v,p) dv.*v.*(1-v);
            a.childCount = a.childCount + 1;
        end
        function obj = abs(a)
            if(a.valueType > 0)
                obj = ADNode(2,abs(a.value));
            else
                obj = ADNode(0,abs(a.value));
            end
            obj.parents{1} = a;
            obj.gf{1} = @(dv,v,p) dv.*sign(v);
            a.childCount = a.childCount + 1;
        end
        %% Logical opeartors
        function obj = lt(a,b)
            if(isa(a,'ADNode') && isa(b,'ADNode'))
                obj = a.value < b.value;
            elseif(isa(a,'ADNode'))
                obj = a.value < b;
            else
                obj = a < b.value;
            end
        end
        function obj = gt(a,b)
            if(isa(a,'ADNode') && isa(b,'ADNode'))
                obj = a.value > b.value;
            elseif(isa(a,'ADNode'))
                obj = a.value > b;
            else
                obj = a > b.value;
            end
        end
        function obj = le(a,b)
            if(isa(a,'ADNode') && isa(b,'ADNode'))
                obj = a.value <= b.value;
            elseif(isa(a,'ADNode'))
                obj = a.value <= b;
            else
                obj = a <= b.value;
            end
        end
        function obj = ge(a,b)
            if(isa(a,'ADNode') && isa(b,'ADNode'))
                obj = a.value >= b.value;
            elseif(isa(a,'ADNode'))
                obj = a.value >= b;
            else
                obj = a >= b.value;
            end
        end
        function obj = ne(a,b)
            if(isa(a,'ADNode') && isa(b,'ADNode'))
                obj = a.value ~= b.value;
            elseif(isa(a,'ADNode'))
                obj = a.value ~= b;
            else
                obj = a ~= b.value;
            end
        end
        function obj = eq(a,b)
            if(isa(a,'ADNode') && isa(b,'ADNode'))
                obj = a.value == b.value;
            elseif(isa(a,'ADNode'))
                obj = a.value == b;
            else
                obj = a == b.value;
            end
        end
        function obj = and(a,b)
            if(isa(a,'ADNode') && isa(b,'ADNode'))
                obj = a.value && b.value;
            elseif(isa(a,'ADNode'))
                obj = a.value && b;
            else
                obj = a && b.value;
            end
        end
        function obj = or(a,b)
            if(isa(a,'ADNode') && isa(b,'ADNode'))
                obj = a.value || b.value;
            elseif(isa(a,'ADNode'))
                obj = a.value || b;
            else
                obj = a || b.value;
            end
        end
        function obj = not(a)
            obj = ~a.value;
        end
        %% Indexing operators
        function obj = horzcat(varargin)
            if(nargin == 1)
                obj = varargin{1};
            else
                p = zeros(size(varargin));
                s = p;
                t = 0;
                v = [];
                for i=1:nargin
                    if(isa(varargin{i},'ADNode'))
                        if(varargin{i}.valueType > 0)
                            t = 2;
                        end
                        p(i) = 1;
                        v = [v, varargin{i}.value];
                        s(i) = size(varargin{i}.value,2);
                    else
                        v = [v, varargin{i}];
                        s(i) = size(varargin{i},2);
                    end
                    
                end
                obj = ADNode(t,v);
                j = 1;
                for i=1:nargin
                    if(p(i) == 1)
                        obj.parents{j} = varargin{i};
                        varargin{i}.childCount = varargin{i}.childCount + 1;
                        obj.gf{j} = @(dv,v,p) dv(:,sum(s(1:i-1))+1:sum(s(1:i)));
                        j = j + 1;
                    end
                end
            end
        end    
        function obj = vertcat(varargin)
            if(nargin == 1)
                obj = varargin{1};
            else
                p = zeros(size(varargin));
                s = p;
                t = 0;
                v = [];
                for i=1:nargin
                    if(isa(varargin{i},'ADNode'))
                        if(varargin{i}.valueType > 0)
                            t = 2;
                        end
                        p(i) = 1;
                        v = [v; varargin{i}.value];
                        s(i) = size(varargin{i}.value,1);
                    else
                        v = [v; varargin{i}];
                        s(i) = size(varargin{i},1);
                    end
                    
                end
                obj = ADNode(t,v);
                j = 1;
                for i=1:nargin
                    if(p(i) == 1)
                        obj.parents{j} = varargin{i};
                        varargin{i}.childCount = varargin{i}.childCount + 1;
                        obj.gf{j} = @(dv,v,p) dv(sum(s(1:i-1))+1:sum(s(1:i)),:);
                        j = j + 1;
                    end
                end
            end
        end
        %{
        function obj = subsref(a,s)
            if(all(size(s) == [1 1]) && strcmp(s.type,'()'))
                if(a.valueType == 0)
                    t = 0;
                else
                    t = 2;
                end
                mask = zeros(size(a.value));
                if(size(s.subs,2) == 1)
                    obj = ADNode(t,a.value(s.subs{1}));
                    mask(s.subs{1}) = 1;
                else
                    obj = ADNode(t,a.value(s.subs{1},s.subs{2}));
                    mask(s.subs{1},s.subs{2}) = 1;
                end
                obj.parents{1} = a;
                obj.gf{1} = @(dv,v,p) ADNode.extrapolateSubmatrix(mask,dv);
            else
                obj = builtin('subsref',a,s);
            end
        end
        %}
        function obj = subsasgn(a,s,b)
            if(all(size(s) == [1 1]) && strcmp(s.type,'()'))
                if(a.valueType == 2)
                    error('Only constant and parametr nodes support subassign');
                else
                    obj = builtin('subsasgn',a.value,s,b);
                end
            else
                obj = builtin('subsasgn',a,s,b);
            end
        end
        %% Gradient calculation
        function calculateGradient(a,dv)
            if(any(size(a.grad) ~= size(dv)))
                a.grad = dv;
            else
                a.grad = a.grad + dv;
            end
            a.childMsg = a.childMsg + 1;
            if(a.childMsg >= a.childCount)
                for i=1:size(a.parents,2)
                    if(a.parents{i}.valueType > 0)
                        calculateGradient(a.parents{i},a.gf{i}(a.grad,a.value,a.parents));
                    end
                end
                a.childMsg = 0;
            end
        end
        function clearGradient(a)
            a.grad = zeros(size(a.value));
            for i=1:size(a.parents,2)
                clearGradient(a.parents{i});
            end
        end
    end
end
